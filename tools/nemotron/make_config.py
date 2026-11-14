#!/usr/bin/env python3
"""tools/nemotron/make_config.py - a run config for NVIDIA-Nemotron-3.5-Super-VL-120B-A12B (nemotron_h_moe) on this checkout's own
build, for serve/server.py.  It reads only the GGUF's first shard (the metadata shard, about 10 MB: the tokenizer and the chat
template) and writes:

    <out>/tokenizer/          vocab, merges, token types, the tekken pre-tokenizer, chat_template.jinja
    <out>/strata-nemotron-<ctx>.json   the run config

    python tools/nemotron/make_config.py --model-dir E:/Strata-data/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF/UD-IQ3_XXS \
        [--context 128k|256k] [--mtp N] [--chunk 4096] [--gpu 0,1] [--vision cpu|gpu|off] [--port 8080] [--out build/nemotron-run]
    python serve/server.py --engine strata --config build/nemotron-run/strata-nemotron-128k.json --port 8080

The Terminal writes the same config by itself (`strata launch --family nemotron35`, setup.py --family nemotron35); both use
strata_control/nemotron35.py.  Nothing here loads the weights; the engine does when the server starts.  Pictures need strata-vision built
against a llama.cpp that knows the nemotron_omni projector (the pinned 3cf03257 does not): tools/vision with -DLLAMA_DIR=<newer llama.cpp>,
see docs/NEMOTRON35.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_control import nemotron35 as N  # noqa: E402

CONTEXTS = N.CONTEXTS
MODEL_NAME = N.MODEL_NAME


def shards(model_dir: Path) -> tuple[Path, Path | None]:
    """(the first shard, the mmproj file or None) in `model_dir`."""
    first = sorted(model_dir.glob("*-00001-of-*.gguf"))
    if not first:
        single = [p for p in sorted(model_dir.glob("*.gguf")) if not p.name.lower().startswith("mmproj")]
        if not single:
            raise SystemExit(f"no GGUF in {model_dir}")
        first = single[:1]
    mmproj = sorted(model_dir.glob("mmproj*.gguf")) or sorted(model_dir.parent.glob("mmproj*.gguf"))
    return first[0], (mmproj[0] if mmproj else None)


def gpu_numbers(text: str) -> list[int]:
    out = [int(x) for x in text.replace(" ", "").split(",") if x != ""]
    if not out or any(g < 0 for g in out) or len(set(out)) != len(out):
        raise SystemExit(f"--gpu: expected numbers like 0 or 0,1, not {text!r}")
    return out


def build_config(a, first: Path, mmproj: Path | None, tok_dir: Path, out: Path) -> dict:
    vision_exe = None
    if a.vision != "off":
        if mmproj is None:
            raise SystemExit("--vision: no mmproj*.gguf beside the model (or --vision off)")
        vision_exe = Path(a.vision_exe or (ROOT / "build" / ("vision-cuda" if a.vision == "gpu" else "vision-cpu") / "bin" / "strata-vision.exe"))
        if not vision_exe.is_file():
            raise SystemExit(f"--vision {a.vision}: {vision_exe} is not built (docs/NEMOTRON35.md, 'Pictures'), or pass --vision-exe / --vision off")
    try:
        return N.build_config(first=first, exe=Path(a.exe), tokenizer=tok_dir, out=out, context=CONTEXTS[a.context], mtp=a.mtp, chunk=a.chunk,
                              threads=a.threads, gpus=a.gpus, vision=a.vision, vision_exe=vision_exe, mmproj=mmproj,
                              vision_max_tokens=a.vision_max_tokens, vision_reserve_mib=a.vision_reserve_mib,
                              log=out / f"strata-nemotron-{a.context}.log", mtp_slots=a.mtp_slots)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", required=True, type=Path, help="the folder of the GGUF shards (and the mmproj)")
    ap.add_argument("--context", choices=sorted(CONTEXTS), default="128k")
    ap.add_argument("--mtp", type=int, default=N.DEFAULT_MTP, help="MTP drafts per step, 0..7 (measured at temperature 1.0: 0/2/3/4 drafts 10.5/13.6/14.7/13.1 tok/s)")
    ap.add_argument("--chunk", type=int, default=N.DEFAULT_CHUNK, help="prompt rows per forward (4096: a 3,218-token prompt 156 tok/s against 93 at 1024, decode 7.5 against 7.8 tok/s; measured 2026-11-10)")
    ap.add_argument("--mtp-slots", type=int, default=N.DEFAULT_MTP_SLOTS,
                    help="VRAM slots for the MTP block's own experts, taken from its GPU's other slots (0: none; measured 2026-11-13: a "
                         "verify window and its drafts 254 ms with 64, 270 ms with none, 253 ms with 128)")
    ap.add_argument("--threads", type=int, default=0, help="CPU expert threads (0: the engine's default, all but one)")
    ap.add_argument("--gpu", default="0,1", help="the cards (nvidia-smi numbers): 0,1 = a layer split over both")
    ap.add_argument("--vision", choices=("cpu", "gpu", "off"), default="cpu",
                    help="the picture reader: on the CPU (no VRAM, slower), on the GPU (about 1.5 GB of VRAM), or none")
    ap.add_argument("--vision-exe", help="strata-vision.exe (default: build/vision-cpu or build/vision-cuda)")
    ap.add_argument("--vision-max-tokens", type=int, help="image tokens per picture at most (default 3328, the projector's own top)")
    ap.add_argument("--vision-reserve-mib", type=int, default=N.VISION_GPU_RESERVE_MIB, help="--vision gpu: VRAM the engine leaves free on every card")
    ap.add_argument("--port", type=int, default=8080, help="the port the start command line names")
    ap.add_argument("--exe", default=str(ROOT / "build" / "nemotron-cuda" / "strata.exe"))
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "nemotron-run")
    a = ap.parse_args(argv)
    if not 0 <= a.mtp <= 7:
        ap.error("--mtp: 0..7")
    if a.chunk < 9:
        ap.error("--chunk: at least 9")
    a.gpus = gpu_numbers(a.gpu)
    first, mmproj = shards(a.model_dir)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        tok_dir = N.write_tokenizer(first, out, ROOT)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    cfg = build_config(a, first, mmproj, tok_dir, out)
    path = N.write(out / f"strata-nemotron-{a.context}.json", cfg)
    print(f"wrote {path}")
    print(f"start: python serve/server.py --engine strata --config \"{path}\" --port {a.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
