#!/usr/bin/env python3
"""tools/nemotron/smoke.py - the first run of the real model: one prompt through strata-nemotron.exe (no server), greedy, decoded.

    python tools/nemotron/smoke.py [--tokenizer build/nemotron-run/tokenizer] [--prompt "..."] [--n 48] [--ctx 8192] [--mtp 0]
                                   [--check] [-- <more engine flags>]

The prompt is rendered with the model's chat template (thinking off, so the answer starts at once), tokenized with the tekken rule,
and passed as --tokens; the engine prints the ids it made, which are decoded here.  --check sets STRATA_NEMO_CHECK=1 (the engine
stops at the first block whose output is not finite and says which).  It loads the model: run it when the GPUs are free.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
MODEL = os.environ.get("STRATA_NEMOTRON_GGUF", "")   # the first shard; --model overrides
CUDA_BIN = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1")) / "bin"


def load_tokenizer(tdir: Path):
    import strata_tokenizer as ST
    vocab = json.loads((tdir / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tdir / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tdir / "token_type.json").read_text())
    tcfg = json.loads((tdir / "tokenizer.json").read_text(encoding="utf-8"))
    return ST.Tokenizer(tokens, merges, types, tcfg["pre"], tcfg.get("special_ids"))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, default=Path(MODEL) if MODEL else None,
                    help="the GGUF's first shard (default: STRATA_NEMOTRON_GGUF)")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "build" / "nemotron-run" / "tokenizer")
    ap.add_argument("--exe", type=Path, default=ROOT / "build" / "nemotron-cuda" / "strata-nemotron.exe")
    ap.add_argument("--prompt", default="What is the capital of Japan? Answer in one sentence.")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--mtp", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.model is None:
        ap.error("--model <first shard> (or STRATA_NEMOTRON_GGUF) is needed")
    from serve.frontend import ChatTemplate
    tok = load_tokenizer(a.tokenizer)
    text = ChatTemplate(a.tokenizer / "chat_template.jinja").render([{"role": "user", "content": a.prompt}],
                                                                    add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(text, parse_special=True)
    print(f"prompt: {len(ids)} tokens", flush=True)
    env = dict(os.environ)
    env["PATH"] = str(CUDA_BIN) + os.pathsep + env.get("PATH", "")
    if a.check:
        env["STRATA_NEMO_CHECK"] = "1"
    cmd = [str(a.exe), "--model", str(a.model), "--max-context", str(a.ctx), "--mtp", str(a.mtp), "--tokens",
           ",".join(map(str, ids)), "--n", str(a.n), *extra]
    t0 = time.monotonic()
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, text=True)
    print(f"engine: exit {r.returncode} after {time.monotonic() - t0:.1f} s (loading included; its log is above)")
    line = (r.stdout or "").strip().splitlines()[-1] if (r.stdout or "").strip() else ""
    if r.returncode != 0 or not line:
        return r.returncode or 1
    out = [int(x) for x in line.split(",") if x.strip().lstrip("-").isdigit()]
    print("ids:", out)
    print("text:", repr(tok.decode(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
