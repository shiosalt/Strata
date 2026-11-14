#!/usr/bin/env python3
"""tools/nemotron/calibrate.py - measure Nemotron-3.5's two speed settings on this computer and write the faster ones into its run config.

    python tools/nemotron/calibrate.py --config <strata-nemotron35-ud-iq3_xxs.json> [--mtp 0,2,3,4] [--chunk 2048,4096] [--tokens 256]
                                       [--repeat 2] [--port 8199] [--dry-run] [--output json]

The settings (docs/NEMOTRON35.md):
    --mtp N      MTP drafts a step. Measured on 2 x RTX 5060 Ti 16 GB, Ryzen 7 5700X, 96 GB, temperature 1.0: 0/2/3/4 drafts 10.5/13.6/14.7/13.1 tok/s.
    --prefill N  prompt rows per forward. Same PC: a 3,218-token prompt 156 tok/s at 4096 against 93 at 1024.
The PCIe share (how many experts each card reads over its link) is measured by the engine itself at every start, so it is not swept here.

Each setting is one start of the model (about 1 to 3 minutes: the experts are copied into pinned memory) and a few requests at
temperature 1.0 (the model card's text setting, where the MTP drafts are sampled): a story of --tokens tokens for the answer speed, and
a ~3,000-token prompt for the prompt speed. The decode sweep runs first at the config's chunk; the chunk sweep then runs at the best MTP.
A value is kept only when it is at least --margin (3%) faster than the config's own; otherwise the config is left as it is.
The old config is kept beside it as <name>.before-calibrate.json. Nothing is changed with --dry-run.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

STORY = "Write a long, detailed story about a lighthouse keeper who finds a message in a bottle. Keep writing until you are stopped."
# about 3,000 tokens of plain text: a numbered list the model has to read whole (the prompt speed counts every row)
PROMPT_TEXT = "\n".join(f"Item {i}: the shipment of part number {1000 + i * 7} left warehouse {i % 13} on day {i % 28 + 1} with {i * 3 % 97} boxes."
                        for i in range(1, 121)) + "\n\nHow many items are listed above? Answer with the number only."


def say(msg: str) -> None:
    print(msg, flush=True)


def arg_value(args: list, flag: str) -> str | None:
    return str(args[args.index(flag) + 1]) if flag in args and args.index(flag) + 1 < len(args) else None


def with_value(args: list, flag: str, value) -> list:
    out = [str(a) for a in args]
    if flag in out and out.index(flag) + 1 < len(out):
        out[out.index(flag) + 1] = str(value)
    else:
        out += [flag, str(value)]
    return out


def numbers(text: str, lo: int, hi: int, what: str) -> list[int]:
    try:
        out = [int(x) for x in text.replace(" ", "").split(",") if x]
    except ValueError:
        raise SystemExit(f"--{what}: numbers like 2,3,4, not {text!r}") from None
    if not out or any(not lo <= n <= hi for n in out):
        raise SystemExit(f"--{what}: each value {lo}..{hi}")
    return list(dict.fromkeys(out))


def post(url: str, body: dict, timeout: float = 1800) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Server:
    """serve/server.py on one variant of the config, on a port of its own; stopped (POST /unload, then the process tree) when done."""

    def __init__(self, cfg: dict, port: int, python: str, workdir: Path):
        self.cfg, self.port, self.python, self.workdir = cfg, port, python, workdir
        self.url = f"http://127.0.0.1:{port}"
        self.proc = None
        self.thinking = True

    def __enter__(self):
        path = self.workdir / f"variant-{self.port}.json"
        cfg = {k: v for k, v in self.cfg.items() if k not in ("expert_profile_save",)}   # a sweep never rewrites the learned profile
        cfg["log"] = str(self.workdir / "engine.log")
        cfg["open_browser"] = False
        path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        self.out = open(self.workdir / "server.log", "ab")
        self.proc = subprocess.Popen([self.python, "-m", "serve.server", "--engine", "strata", "--config", str(path), "--host", "127.0.0.1",
                                      "--port", str(self.port)], cwd=str(ROOT), stdout=self.out, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the server stopped while loading (see {self.workdir / 'server.log'})")
            try:
                urllib.request.urlopen(self.url + "/v1/models", timeout=5).read()
                post(self.url + "/v1/chat/completions", {"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}, 900)
                self.load_s = time.time() - t0
                return self
            except (urllib.error.URLError, ConnectionError, OSError, ValueError):
                time.sleep(3)
        raise RuntimeError("the model did not load in 15 minutes")

    def __exit__(self, *exc):
        try:
            post(self.url + "/unload", {}, 120)
        except Exception:                                    # noqa: BLE001 - the process is stopped below either way
            pass
        if self.proc is not None:
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
                else:
                    self.proc.kill()
                self.proc.wait(timeout=30)
        self.out.close()
        time.sleep(3)                                        # the cards' memory is given back before the next start

    def decode(self, tokens: int) -> dict:
        r = post(self.url + "/v1/chat/completions", {"model": "m", "temperature": 1.0, "top_p": 0.95, "max_tokens": tokens, "seed": 7,
                                                     "chat_template_kwargs": {"enable_thinking": self.thinking},
                                                     "messages": [{"role": "user", "content": STORY}]})
        t = r.get("timings") or {}
        return {"tok_s": float(t.get("predicted_per_second") or 0), "n": int(t.get("predicted_n") or 0),
                "drafts": f"{t.get('draft_n_accepted', '-')}/{t.get('draft_n', '-')}"}

    def prompt(self, salt: int) -> dict:
        # a different first line every time, so the conversation cache cannot answer it
        text = f"Run {salt} at {time.time():.3f}.\n" + PROMPT_TEXT
        r = post(self.url + "/v1/chat/completions", {"model": "m", "temperature": 1.0, "max_tokens": 1,
                                                     "chat_template_kwargs": {"enable_thinking": False},
                                                     "messages": [{"role": "user", "content": text}]})
        t = r.get("timings") or {}
        return {"tok_s": float(t.get("prompt_per_second") or 0), "n": int(t.get("prompt_n") or 0)}


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0.0


def measure(cfg: dict, a, workdir: Path, label: str, want: str) -> float:
    """One start of `cfg`: the median decode or prompt speed over --repeat runs (the first, warm-up run is not counted)."""
    srv0 = Server(cfg, a.port, a.python, workdir)
    srv0.thinking = a.thinking == "on"
    with srv0 as srv:
        runs = []
        for i in range(a.repeat + 1):
            got = srv.decode(a.tokens) if want == "decode" else srv.prompt(i)
            if i > 0:
                runs.append(got["tok_s"])
            extra = f", drafts {got['drafts']}" if want == "decode" else ""
            say(f"  {label}: run {i}{' (warm-up)' if i == 0 else ''} {got['n']} tokens at {got['tok_s']:.1f} tok/s{extra}")
        say(f"  {label}: {median(runs):.1f} tok/s (loaded in {srv.load_s:.0f} s)")
        return median(runs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, type=Path, help="the run config (strata-nemotron35-<size>.json, or a tools/nemotron/make_config.py one)")
    ap.add_argument("--mtp", default="0,2,3,4", help="MTP drafts to try (0..7)")
    ap.add_argument("--chunk", default="2048,4096", help="prompt rows per forward to try")
    ap.add_argument("--tokens", type=int, default=256, help="tokens per decode run")
    ap.add_argument("--repeat", type=int, default=2, help="counted runs per setting (after one warm-up run)")
    ap.add_argument("--thinking", choices=("on", "off"), default="on",
                    help="the decode runs with the model's thinking on (the default, as agents use it) or off: the best MTP differs "
                         "(2026-11-13, this PC, a story with thinking off: 3 drafts 9.9 tok/s, 2 drafts 11.7)")
    ap.add_argument("--margin", type=float, default=0.03, help="how much faster a value must be than the config's own to be kept")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--python", default=sys.executable, help="the Python that runs serve/server.py")
    ap.add_argument("--dry-run", action="store_true", help="say what would be measured; start nothing, change nothing")
    ap.add_argument("--output", choices=("text", "json"), default="text")
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text(encoding="utf-8-sig"))
    if cfg.get("arch") != "nemotron_h_moe":
        raise SystemExit(f"{a.config.name} is not a Nemotron-3.5 run config (its \"arch\" is {cfg.get('arch')!r})")
    args = [str(x) for x in cfg.get("args") or []]
    mtps, chunks = numbers(a.mtp, 0, 7, "mtp"), numbers(a.chunk, 9, 65536, "chunk")
    cur_mtp, cur_chunk = int(arg_value(args, "--mtp") or 0), int(arg_value(args, "--prefill") or 4096)
    if a.repeat < 1 or a.tokens < 16:
        raise SystemExit("--repeat 1 or more, --tokens 16 or more")
    mtps = list(dict.fromkeys([cur_mtp, *mtps]))
    chunks = list(dict.fromkeys([cur_chunk, *chunks]))
    starts = len(mtps) + len(chunks)
    say(f"Nemotron-3.5 calibration of {a.config.name}: now --mtp {cur_mtp} --prefill {cur_chunk}.")
    say(f"  {starts} starts of the model (MTP {', '.join(map(str, mtps))}; then prompt chunks {', '.join(map(str, chunks))} at the best MTP), "
        f"{a.repeat} counted runs each. Close other programs that use the graphics cards.")
    if a.dry_run:
        say("Nothing was started or changed (--dry-run).")
        return 0
    workdir = Path(tempfile.mkdtemp(prefix="nemotron-calibrate-"))
    result = {"config": str(a.config), "before": {"mtp": cur_mtp, "chunk": cur_chunk}, "decode": {}, "prompt": {}}
    try:
        for m in mtps:
            result["decode"][m] = measure({**cfg, "args": with_value(args, "--mtp", m)}, a, workdir, f"--mtp {m}", "decode")
        base = result["decode"][cur_mtp]
        best_mtp = max(mtps, key=lambda m: result["decode"][m])
        if result["decode"][best_mtp] < base * (1 + a.margin):
            best_mtp = cur_mtp
        args_m = with_value(args, "--mtp", best_mtp)
        for c in chunks:
            result["prompt"][c] = measure({**cfg, "args": with_value(args_m, "--prefill", c)}, a, workdir, f"--prefill {c}", "prompt")
        best_chunk = max(chunks, key=lambda c: result["prompt"][c])
        if result["prompt"][best_chunk] < result["prompt"][cur_chunk] * (1 + a.margin):
            best_chunk = cur_chunk
    except (RuntimeError, urllib.error.URLError, OSError) as e:
        say(f"Calibration stopped: {e}. The config was not changed.")
        return 1
    result["after"] = {"mtp": best_mtp, "chunk": best_chunk}
    if (best_mtp, best_chunk) != (cur_mtp, cur_chunk):
        keep = a.config.with_name(a.config.stem + ".before-calibrate.json")
        shutil.copyfile(a.config, keep)
        cfg["args"] = with_value(with_value(args, "--mtp", best_mtp), "--prefill", best_chunk)
        cfg["nemotron_calibrated"] = {"at": time.strftime("%Y-%m-%d %H:%M"), "decode_tok_s": result["decode"],
                                      "prompt_tok_s": result["prompt"]}
        tmp = a.config.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, a.config)
        say(f"Wrote --mtp {best_mtp} --prefill {best_chunk} into {a.config.name} (the old one: {keep.name}).")
    else:
        say(f"The config's own --mtp {cur_mtp} --prefill {cur_chunk} were the fastest (within {a.margin:.0%}); nothing was changed.")
    shutil.rmtree(workdir, ignore_errors=True)
    if a.output == "json":
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
