"""NVIDIA-Nemotron-3.5-Super-VL-120B-A12B (GGUF architecture nemotron_h_moe): the one table and the one config writer that setup.py
(`--family nemotron35`), `strata launch --family nemotron35`, the MCP server's model list and tools/nemotron/make_config.py share.

The model runs on the engine in src/nemotron (`strata-nemotron`, built from a source checkout: docs/NEMOTRON35.md). The release
packages carry no build of it yet, so nothing here downloads an engine: it is looked for in the places below and, when it is not there,
setup says so in one plain line and stops before the 60 GB download.

    the files      Unsloth's UD-IQ3_XXS, three shards (59.8 GB) and the picture reader mmproj-BF16.gguf (1.7 GB), at a pinned revision,
                   each file with its size and SHA-256 (the Hub's LFS etag; the mmproj hashed here, 2026-11-10)
    the engine     STRATA_NEMOTRON_ENGINE, <install>/engine-nemotron/strata.exe, <checkout>/build/nemotron-cuda/strata.exe (a build),
                   <checkout>/engine-nemotron/strata.exe (the fork's ready-made Windows engine, unzipped into the checkout)
    the reader     strata-vision built against a llama.cpp that knows the nemotron_omni projector (the pinned one does not):
                   STRATA_NEMOTRON_VISION, <install>/engine-nemotron/strata-vision.exe, <checkout>/build/vision-cpu|vision-cuda/bin,
                   <checkout>/engine-nemotron/strata-vision.exe

Measured on 2 x RTX 5060 Ti 16 GB, Ryzen 7 5700X, 96 GB RAM, Windows (docs/NEMOTRON35_PROGRESS.md): decode 10.2 tok/s greedy and 14.7 tok/s at
temperature 1 with 3 MTP drafts, a 3,218-token prompt 166.8 tok/s.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

FAMILY = "nemotron35"
ARCH = "nemotron_h_moe"
TITLE = "Nemotron-3.5-Super-VL"
REPO = "unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF"
REVISION = "75a5829055f69a0359085ae6bbd0a81b60bc05c6"          # the commit the port was checked against (2026-11-10)
MODEL_NAME = "nemotron-3.5-super-vl-120b-a12b"
TAG = "nemotron35-"
MMPROJ = "mmproj-BF16.gguf"
REPO_FOLDER = "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF"    # the folder under <data>/models (the repository's name, as a Hub download names it)
LICENSE = "NVIDIA's license for the model: its model card says which (the repository " + REPO + ")"

_SHARD = "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-UD-IQ3_XXS-{i:05d}-of-00003.gguf"
FILES = {                                                      # name: (bytes, sha256)
    _SHARD.format(i=1): (7874568, "01150188b68c1ebf4528d095eab5fa12b7ae18acfe7cb9fcc5563c0772badd94"),
    _SHARD.format(i=2): (49997462976, "fbf889020bf9effe03da67985d24a1b5d58ad382fcdd02fa390c09687abc0957"),
    _SHARD.format(i=3): (9758873024, "e69358032031aa6dac6838dffb03ad2d54e510bad2fea10513db7dd27dfbd70b"),
    MMPROJ: (1733695840, "9c09381a5744d0ea9a30048c94922f6f25ee0d63842de9c45779eca4fb5924d9"),
}
MODELS = {
    "UD-IQ3_XXS": {
        "about": "3-bit (Unsloth Dynamic IQ3_XXS), 59.8 GB: two 16 GB cards and 64 GB of RAM or more",
        "subdir": "UD-IQ3_XXS",                                # the folder at the repository; the mmproj is at its top
        "shards": [_SHARD.format(i=i) for i in (1, 2, 3)],
        # RAM: the 49.2 GiB of routed experts are copied into pinned memory at load (--no-pin turns it off); the 96 GB PC it was measured on
        # had room to spare, a smaller one was not tried
        "ram_gb": 64,
        "arena_gb": 52.8,
    },
}
CONTEXTS = {"128k": 131072, "256k": 262144}
DEFAULT_CONTEXT = 131072
# drafts per step: temperature 1.0, 0/2/3/4 drafts 10.5/13.6/14.7/13.1 tok/s (2026-11-10, before the MTP slots); with the MTP slots and the
# later decode changes 3 drafts 15 tok/s against 6 drafts 21 tok/s (2026-11-13, 2 x RTX 5060 Ti, 九九の表 with thinking off, 128K)
DEFAULT_MTP = 6
DEFAULT_CHUNK = 4096       # prompt rows per forward: a 3,218-token prompt 156 tok/s against 93 at 1024
# VRAM slots for the MTP block's own experts (its GPU's other slots give the room): a verify window and its drafts 254 ms against 270 ms
# without, 253 ms with 128 slots (2026-11-13, 2 x RTX 5060 Ti, 九九の表 with thinking off, temperature 1.0, MTP 6)
DEFAULT_MTP_SLOTS = 64
VISION_MAX_TOKENS = 3328   # the projector's own top (mtmd does not scale a picture down to a lower limit for it)
VISION_GPU_RESERVE_MIB = 2048
NOT_BUILT = ("the Nemotron engine (strata-nemotron) is not on this computer: upstream's release packages do not carry it. Unzip the fork's "
             "ready-made Windows engine (https://github.com/shiosalt/Strata/releases) into this checkout (it makes engine-nemotron), "
             "build it (tools/nemotron/build.ps1, docs/NEMOTRON35.md), or put its strata.exe in {where}")


def entry() -> dict:
    """The family as setup.py's tables hold a family (title, tag, models, the pinned files)."""
    return {"title": TITLE, "by": "NVIDIA; GGUFs by Unsloth", "arch": ARCH, "tag": TAG, "name": MODEL_NAME, "license": LICENSE,
            "about": "120B hybrid Mamba-attention mixture of experts with 12B active that reads pictures, on two 16 GB cards and the RAM",
            "repo": REPO, "revision": REVISION, "models": MODELS, "sha256": FILES, "vision": True, "nemotron": True}


def size_row(model: str) -> dict | None:
    """One size as a model-table row (tools/strata_mcp.py own_size): the download is the shards and the picture reader."""
    d = MODELS.get(model)
    if d is None:
        return None
    gb = sum(FILES[n][0] for n in d["shards"]) / 1e9
    return {"about": d["about"], "download_gb": round(gb, 1), "ram_gb": d["ram_gb"], "arena_gb": d["arena_gb"], "vision": True,
            "engine": None, "families": (FAMILY,), "nemotron": True}


def is_config(cfg: dict) -> bool:
    return isinstance(cfg, dict) and cfg.get("arch") == ARCH


# ---- the files
def folder_for(models_dir: Path, model: str) -> Path:
    """Where setup puts this size's files: <models>/<repository>/<size> (the layout a Hub download makes)."""
    return Path(models_dir) / REPO_FOLDER / MODELS[model]["subdir"]


def find_set(path) -> tuple[str, Path] | None:
    """(size, folder) when `path` (a folder, the repository folder above the size's own folder, or one shard) holds shards of a pinned set."""
    if not path:
        return None
    p = Path(path)
    for model, d in MODELS.items():
        folders = [p, p / d["subdir"]] if p.is_dir() else [p.parent]
        for f in folders:
            if any((f / n).is_file() for n in d["shards"]):
                return model, f
    return None


def missing_parts(folder: Path, model: str) -> list[str]:
    """The shards of `model` that are not in `folder` (a file of the wrong size counts as missing)."""
    out = []
    for n in MODELS[model]["shards"]:
        f = Path(folder) / n
        if not f.is_file() or f.stat().st_size != FILES[n][0]:
            out.append(n)
    return out


def mmproj_in(folder: Path) -> Path | None:
    """The picture reader beside the shards, or at the repository folder's top (where the Hub keeps it)."""
    for f in (Path(folder) / MMPROJ, Path(folder).parent / MMPROJ):
        if f.is_file() and f.stat().st_size == FILES[MMPROJ][0]:
            return f
    return None


# ---- the engine and the picture reader
def _exe(name: str) -> str:
    return name + (".exe" if os.name == "nt" else "")


def engine_candidates(install_root: Path | None, src: Path | None) -> list[Path]:
    out = []
    if os.environ.get("STRATA_NEMOTRON_ENGINE"):
        out.append(Path(os.environ["STRATA_NEMOTRON_ENGINE"]))
    if install_root:
        out.append(Path(install_root) / "engine-nemotron" / _exe("strata"))
    if src:
        out.append(Path(src) / "build" / "nemotron-cuda" / _exe("strata"))
        out.append(Path(src) / "engine-nemotron" / _exe("strata"))   # the ready-made engine (the fork's Releases), unzipped here
    return out


def find_engine(install_root: Path | None, src: Path | None) -> Path | None:
    return next((p for p in engine_candidates(install_root, src) if p.is_file()), None)


def find_vision(install_root: Path | None, src: Path | None, gpu: bool) -> Path | None:
    """strata-vision that reads the nemotron_omni projector: the one named, the one beside the engine, else this checkout's build."""
    cands = []
    if os.environ.get("STRATA_NEMOTRON_VISION"):
        cands.append(Path(os.environ["STRATA_NEMOTRON_VISION"]))
    if install_root:
        cands.append(Path(install_root) / "engine-nemotron" / _exe("strata-vision"))
    if src:
        first, second = ("vision-cuda", "vision-cpu") if gpu else ("vision-cpu", "vision-cuda")
        cands += [Path(src) / "build" / first / "bin" / _exe("strata-vision")]
        if gpu:
            cands += [Path(src) / "build" / second / "bin" / _exe("strata-vision")]
        cands.append(Path(src) / "engine-nemotron" / _exe("strata-vision"))   # (the ready-made package's: the CPU reader)
    return next((p for p in cands if p.is_file()), None)


def cuda_lib_dirs() -> list[str]:
    """The CUDA runtime folder the engine's DLLs come from (Windows; Linux finds them through the loader)."""
    if os.name != "nt":
        return []
    cands = []
    if os.environ.get("CUDA_PATH"):
        cands.append(Path(os.environ["CUDA_PATH"]) / "bin")
    base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if base.is_dir():
        cands += sorted((p / "bin" for p in base.iterdir() if p.name.startswith("v13")), reverse=True)
    return [str(p) for p in cands if p.is_dir()][:1]


# ---- the run config
def context_of(value) -> int:
    """128k / 256k / a number -> tokens; anything else the default (128K). The engine was checked at both (docs/NEMOTRON35_PROGRESS.md)."""
    if value is None:
        return DEFAULT_CONTEXT
    s = str(value).strip().lower()
    if s in CONTEXTS:
        return CONTEXTS[s]
    n = int(s)
    if n < 4096 or n > CONTEXTS["256k"]:
        raise ValueError(f"a context of {n} tokens: Nemotron-3.5 is set up for 4096 to 262144 (128k or 256k)")
    return n


def write_tokenizer(first: Path, out: Path, src: Path | None = None) -> Path:
    """The tokenizer folder (vocab, merges, the tekken pre-tokenizer, the chat template) from the metadata shard: tools/strata_tokenizer.py."""
    tools = Path(src or Path(__file__).resolve().parents[1]) / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import strata_tokenizer as ST
    tcfg = ST.extract(str(first), str(out))
    if tcfg.get("pre") != "tekken":
        raise ValueError(f"{Path(first).name}: its pre-tokenizer is {tcfg.get('pre')!r}, not Nemotron-3.5's tekken")
    tok = Path(out) / "tokenizer"
    if not (tok / "chat_template.jinja").is_file():
        raise ValueError(f"{Path(first).name} has no chat template")
    return tok


def build_config(*, first: Path, exe: Path, tokenizer: Path, out: Path, context: int = DEFAULT_CONTEXT, mtp: int = DEFAULT_MTP,
                 chunk: int = DEFAULT_CHUNK, threads: int = 0, gpus: list[int] | None = None, vision: str = "cpu",
                 vision_exe: Path | None = None, mmproj: Path | None = None, vision_max_tokens: int | None = None,
                 vision_reserve_mib: int = VISION_GPU_RESERVE_MIB, log: Path | None = None, lib_dirs: list[str] | None = None,
                 profile: Path | None = None, mtp_slots: int = DEFAULT_MTP_SLOTS) -> dict:
    """The run config serve/server.py starts the engine with. vision: cpu | gpu | off (cpu and gpu need `vision_exe` and `mmproj`)."""
    if not 0 <= int(mtp) <= 7:
        raise ValueError("--mtp: 0..7 drafts")
    if int(chunk) < 9:
        raise ValueError("--chunk: at least 9 rows")
    gpus = list(gpus) if gpus else [0]
    args = ["--native", str(first), "--max-context", str(int(context)), "--prefill", str(int(chunk)), "--mtp", str(int(mtp))]
    if int(mtp) > 0 and mtp_slots:
        args += ["--mtp-slots", str(int(mtp_slots))]
    if threads:
        args += ["--threads", str(int(threads))]
    # the expert profile: the slots start from what earlier runs routed most, and the engine saves what it routes there (between requests every
    # 10 minutes, and when the server stops); serve/server.py's learned_profile_args handles the key
    profile = Path(profile) if profile else Path(out) / "expert-profile.bin"
    args += ["--expert-profile", str(profile)]
    cfg = {"exe": str(Path(exe).resolve()), "args": args, "tokenizer": str(tokenizer), "arch": ARCH, "model_name": MODEL_NAME,
           "log": str(log or Path(out) / "strata-nemotron.log"), "gpu": gpus if len(gpus) > 1 else gpus[0], "expert_profile_save": str(profile)}
    libs = cuda_lib_dirs() if lib_dirs is None else lib_dirs
    if libs:
        cfg["lib_dirs"] = list(libs)
    if vision != "off":
        if mmproj is None or vision_exe is None:
            raise ValueError("pictures need the picture reader (mmproj-BF16.gguf) and a strata-vision that reads it")
        vis = {"exe": str(Path(vision_exe).resolve()), "mmproj": str(mmproj), "model": str(first), "gpu": vision == "gpu",
               "max_tokens": int(vision_max_tokens) if vision_max_tokens else VISION_MAX_TOKENS}
        args.append("--vision")
        if vision == "gpu":
            vis["cuda_device"] = gpus[-1]                  # the reader's own card (the engine's last stage card): the engine leaves room for it
            args += ["--vram-reserve-mib", str(int(vision_reserve_mib))]
        cfg["vision"] = vis
    return cfg


def write(path: Path, cfg: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def fit_lines(ram_gb: float | None, cards: list[dict]) -> list[str]:
    """Plain warnings (never a refusal) for this computer: an NVIDIA card is needed (the engine is CUDA only), and the measured setup had two
    16 GB cards and 96 GB of RAM."""
    out = []
    nvidia = [c for c in cards or [] if str(c.get("vendor", "nvidia")).lower() == "nvidia"]
    if not nvidia:
        out.append("Nemotron-3.5 runs on NVIDIA cards only (its engine is CUDA); no NVIDIA card was found")
    vram = sum(float(c.get("vram_gb") or 0) for c in nvidia)
    if nvidia and vram < 30:
        out.append(f"the cards have {vram:.0f} GB of graphics memory together; it was measured on two 16 GB cards, so less may not start")
    if ram_gb is not None and float(ram_gb) < MODELS["UD-IQ3_XXS"]["ram_gb"]:
        out.append(f"this computer has {float(ram_gb):.0f} GB of RAM; the experts take about 53 GB of it (measured on 96 GB), so it may not start "
                   "(the engine's --no-pin leaves them in the file cache instead, slower)")
    return out
