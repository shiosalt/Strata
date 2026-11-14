"""tests/nemotron/test_proto.py - the nemotron_h_moe engine path through Strata's engine protocol (what serve/server.py speaks), on the
synthetic model nemotron_engine_parity writes (STRATA_NEMO_KEEP_GGUF), run by `strata --serve` (its family switch) on one GPU.

    STRATA_NEMO_KEEP_GGUF=D:/tmp/tiny.gguf build/.../nemotron_engine_parity
    STRATA_NEMO_GGUF=D:/tmp/tiny.gguf STRATA_NEMO_EXE=build/.../strata.exe python -m pytest tests/nemotron/test_proto.py

Skipped without the two variables (no model, a GPU with a few hundred MB free).
"""
import os
import subprocess
import threading
import queue
import time

import pytest

GGUF = os.environ.get("STRATA_NEMO_GGUF")
EXE = os.environ.get("STRATA_NEMO_EXE")
pytestmark = pytest.mark.skipif(not (GGUF and EXE and os.path.exists(GGUF) and os.path.exists(EXE)),
                                reason="needs STRATA_NEMO_GGUF (the synthetic model) and STRATA_NEMO_EXE (strata.exe)")


class Engine:
    def __init__(self, *extra):
        args = [EXE, "--serve", "--native", GGUF, "--max-context", "512", "--chunk", "32", "--no-warm", "--gpus", "1",
                "--vram-reserve-mib", "0", "--prefill", "auto", "--pool-workers", "4", "--vision", "--vram-elastic", *extra]
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.q = queue.Queue()
        self.err = []
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=lambda: self.err.extend(self.p.stderr), daemon=True).start()
        self.info = {}
        while True:
            line = self.get(120)
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = v
            if line.startswith("READY"):
                self.ready = line.split()
                break

    def _pump(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def get(self, timeout=60):
        line = self.q.get(timeout=timeout)
        assert line is not None, "the engine ended: " + "".join(self.err[-20:])
        return line

    def send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def gen(self, ids, max_new, keys="", stop_after=None, emb=None):
        head = f"GENI {max_new}{(' ' + keys) if keys else ''} {emb}" if emb else f"GEN {max_new}{(' ' + keys) if keys else ''}"
        self.send(f"{head} {','.join(map(str, ids))}")
        toks, pp, done = [], [], None
        while done is None:
            line = self.get()
            if line.startswith("T "):
                toks.append(int(line[2:]))
                if stop_after is not None and len(toks) == stop_after:
                    self.send("STOP")
            elif line.startswith("PP "):
                pp.append(line)
            elif line.startswith("DONE"):
                done = line.split()
            elif line.startswith("ERR"):
                return toks, pp, line
        return toks, pp, done

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


PROMPT = [20 + (i * 37) % 480 for i in range(45)]


@pytest.fixture(scope="module")
def eng():
    e = Engine()
    yield e
    e.close()


def test_ready_and_unknown_flags(eng):
    assert eng.ready[0] == "READY" and eng.ready[1] == "512" and "stop" in eng.ready[2:]
    assert eng.info.get("family") == "nemotron_h_moe"
    assert any("ignoring --vram-elastic" in l for l in eng.err), "the server's own flags are said and skipped"
    assert not any("ignoring --vision" in l for l in eng.err), "--vision is taken (GENI)"


def test_greedy_generation_and_done_fields(eng):
    toks, pp, done = eng.gen(PROMPT, 8)
    assert len(toks) == 8 and pp, (toks, pp, done)
    assert done[0] == "DONE" and int(done[1]) == 8 and int(done[2]) == len(PROMPT) and done[5] == "length"
    # a prompt of two chunks: two PP lines, the last one complete
    last = pp[-1].split()
    assert last[1] == last[2] == str(len(PROMPT))


def test_continuation_reuses_the_held_state(eng):
    toks, _, _ = eng.gen(PROMPT, 5)
    nxt = PROMPT + toks[:-1] + [101, 102, 103]   # the last token was picked, not read
    _, _, done = eng.gen(nxt, 3)
    assert done[0] == "DONE" and int(done[8]) == len(PROMPT) + len(toks) - 1, done   # reused
    assert int(done[14]) == 3                                                          # read


def test_same_request_twice_gives_the_same_tokens(eng):
    a, _, _ = eng.gen(PROMPT, 6)
    b, _, _ = eng.gen(PROMPT, 6)
    assert a == b


def test_sampling_keys_and_seed(eng):
    a, _, d = eng.gen(PROMPT, 6, "temperature=0.8 top_p=0.9 top_k=40 seed=7 penalty_repeat=1.1 penalty_last_n=64")
    b, _, _ = eng.gen(PROMPT, 6, "temperature=0.8 top_p=0.9 top_k=40 seed=7 penalty_repeat=1.1 penalty_last_n=64")
    assert d[0] == "DONE" and a == b, "the same seed, the same tokens"


def test_stop_mid_answer(eng):
    toks, _, done = eng.gen(PROMPT, 100, stop_after=3)
    assert done[0] == "DONE" and done[5] == "cancel" and 3 <= len(toks) <= 5


def test_errors_keep_the_engine_in_step(eng):
    _, _, err = eng.gen([1, 2, 99999], 4)
    assert err.startswith("ERR") and "vocabulary" in err
    _, _, err = eng.gen(PROMPT, 1000)
    assert err.startswith("ERR") and "context" in err
    _, _, err = eng.gen(PROMPT, 4, "logprobs=3")
    assert err.startswith("ERR") and "logprobs" in err
    toks, _, done = eng.gen(PROMPT, 2)
    assert done[0] == "DONE" and len(toks) == 2


def test_mtp_drafts_give_the_same_tokens():
    plain = Engine()
    spec = Engine("--mtp", "2")
    try:
        a, _, _ = plain.gen(PROMPT, 10)
        b, _, d = spec.gen(PROMPT, 10)
        assert a == b
        assert int(d[7]) > 0, "drafts were offered"
    finally:
        plain.close()
        spec.close()


# ---- pictures (GENI): strata-vision records, int32 'SVE1', n, nx, ny, n_embd, then n x n_embd float32, in place of <image> (id 18)
IMG, IMG_BEGIN, IMG_END = 18, 19, 20


def sve(path, nx, ny, width, seed=1):
    import random
    import struct
    rnd = random.Random(seed)
    with open(path, "wb") as f:
        f.write(struct.pack("<5i", 0x31455653, nx * ny, nx, ny, width))
        f.write(struct.pack(f"<{nx * ny * width}f", *[rnd.uniform(-0.05, 0.05) for _ in range(nx * ny * width)]))


def picture_prompt(n):
    """The server's expansion of one template <image>: <img>, n x <image>, </img>, with text around it."""
    return PROMPT[:10] + [IMG_BEGIN] + [IMG] * n + [IMG_END] + PROMPT[10:20]


@pytest.fixture(scope="module")
def width(eng, tmp_path_factory):
    p = tmp_path_factory.mktemp("geni") / "probe.sve"
    sve(p, 1, 1, 1)
    _, _, err = eng.gen(picture_prompt(1), 2, emb=p)
    assert err.startswith("ERR") and "of width" in err, err
    return int(err.rsplit("of width", 1)[1].strip(" )"))


def test_picture_request_reads_and_answers(eng, width, tmp_path):
    p = tmp_path / "a.sve"
    sve(p, 3, 2, width)
    toks, _, done = eng.gen(picture_prompt(6), 4, emb=p, keys="img_keys=00000000000000aa")
    assert done[0] == "DONE" and len(toks) == 4 and int(done[2]) == len(picture_prompt(6)), done
    again, _, _ = eng.gen(picture_prompt(6), 4, emb=p, keys="img_keys=00000000000000aa")
    assert again == toks, "the same picture, the same greedy tokens"


def test_picture_errors_keep_the_engine_in_step(eng, width, tmp_path):
    p = tmp_path / "b.sve"
    sve(p, 3, 2, width)
    _, _, err = eng.gen(picture_prompt(5), 2, emb=p)
    assert err.startswith("ERR") and "fewer <image> tokens" in err, err
    _, _, err = eng.gen(PROMPT[:10] + [IMG_BEGIN] + [IMG] * 6 + [IMG_END] + [IMG] * 6 + [IMG_END] + PROMPT[10:20], 2, emb=p)
    assert err.startswith("ERR") and "more images than the embeddings file" in err, err
    _, _, err = eng.gen(PROMPT, 2, emb=p)
    assert err.startswith("ERR") and "more images than the prompt" in err, err
    _, _, err = eng.gen(PROMPT[:10] + [IMG] * 6, 2, emb=p)
    assert err.startswith("ERR") and "cannot end in an image" in err, err
    _, _, err = eng.gen(picture_prompt(6), 2, emb=tmp_path / "missing.sve")
    assert err.startswith("ERR") and "cannot open" in err, err
    toks, _, done = eng.gen(PROMPT, 2)
    assert done[0] == "DONE" and len(toks) == 2


def test_picture_prefix_is_reused_only_for_the_same_picture(eng, width, tmp_path):
    p = tmp_path / "c.sve"
    sve(p, 3, 2, width, seed=5)
    first = picture_prompt(6)
    toks, _, _ = eng.gen(first, 1, emb=p, keys="img_keys=00000000000000c1")   # one token: the held state is the prompt
    nxt = first + toks + [101, 102]
    _, _, done = eng.gen(nxt, 1, emb=p, keys="img_keys=00000000000000c1")
    assert done[0] == "DONE" and int(done[8]) == len(first), done             # the same picture: the prompt is reused
    other = first + toks + [101, 103]
    _, _, done = eng.gen(other, 1, emb=p, keys="img_keys=00000000000000c2")
    assert done[0] == "DONE" and int(done[8]) <= 11, done                     # another picture: at most the text before it


def test_sampled_drafts_keep_the_distribution():
    """Speculative sampling: the 2nd and 3rd tokens after a prompt are distributed as without drafts.  The yardstick is sampling noise
    itself: a second plain run with other seeds; the drafted run must be as close to the first plain run as that one is (with a margin)."""
    from collections import Counter
    n = 1500
    keys = "temperature=0.4 top_p=0.95 top_k=40"
    plain, spec = Engine(), Engine("--mtp", "2")
    try:
        def run(eng, seeds):
            c2, c3, acc, off = Counter(), Counter(), 0, 0
            for seed in seeds:
                toks, _, done = eng.gen(PROMPT, 3, f"{keys} seed={seed}")
                assert done[0] == "DONE" and len(toks) == 3, done
                c2[toks[1]] += 1
                c3[(toks[1], toks[2])] += 1
                acc += int(done[6])
                off += int(done[7])
            return c2, c3, acc, off

        def tv(a, b):
            return 0.5 * sum(abs(a[k] / n - b[k] / n) for k in set(a) | set(b))

        a2, a3, _, _ = run(plain, range(n))
        b2, b3, _, _ = run(plain, range(n, 2 * n))
        s2, s3, acc, off = run(spec, range(2 * n, 3 * n))
        noise2, noise3 = tv(a2, b2), tv(a3, b3)
        assert tv(a2, s2) < 1.5 * noise2 + 0.03, (tv(a2, s2), noise2)
        assert tv(a3, s3) < 1.5 * noise3 + 0.03, (tv(a3, s3), noise3)
        assert off > 0 and acc > 0, (acc, off)
        print(f"TV 2nd token: drafted {tv(a2, s2):.3f}, noise {noise2:.3f}; pairs: {tv(a3, s3):.3f}, noise {noise3:.3f}; drafts {acc}/{off}")
    finally:
        plain.close()
        spec.close()


def test_session_file_save_and_restore(eng, tmp_path):
    """SAVE <path> / RESTORE <path> (serve/server.py's session files): the conversation comes back after another one, and continuing it
    reads only its new tokens, with the tokens it gives without the interruption."""
    path = str(tmp_path / "chat1.bin")
    toks, _, _ = eng.gen(PROMPT, 2)
    held = PROMPT + toks[:1]                       # what the engine holds: the prompt and the first token (the second was picked, not read)
    eng.send(f"SAVE {path}")
    line = eng.get()
    assert line.startswith("SAVED "), line
    n_saved, size = int(line.split()[1]), int(line.split()[2])
    assert n_saved == len(held) and size > 0 and os.path.getsize(path) > 0
    nxt = held + [toks[1], 101, 102]
    ref, _, _ = eng.gen(nxt, 3)                    # the reference continuation (the held state continues)
    eng.gen([5, 6, 7] + PROMPT[3:30], 2)          # another conversation over the cells
    eng.send(f"RESTORE {path}")
    line = eng.get()
    assert line.startswith("RESTORED ") and int(line.split()[1]) == n_saved, line
    got, _, done = eng.gen(nxt, 3)
    assert int(done[8]) == n_saved, done            # only the new tokens were read
    assert got == ref
    eng.send(f"RESTORE {tmp_path / 'missing.bin'}")
    line = eng.get()
    assert line.startswith("SERR invalid 0"), line
