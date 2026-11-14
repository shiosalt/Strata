// src/nemotron/engine.hpp - the nemotron_h_moe engine (NVIDIA-Nemotron-3.5-Super-VL-120B-A12B): one or more NVIDIA GPUs in a layer
// split, the routed experts in VRAM slots (the most-used ones, adapted while it runs) and the rest computed on the CPU from the mapped
// GGUF, prompts in chunks with the experts streamed to the GPU, the GGUF's own MTP block for greedy drafts, Strata's engine protocol.
//
// The arithmetic is include/strata/nemotron/reference.hpp's (the fp32 CPU reference), with Strata's quantized products: q8_1
// activations against the GGUF blocks (decode, native MMVQ), FP16 GEMMs on dequantized weights (prompt dense), llama.cpp's MMQ (prompt
// experts).  tests/nemotron/engine_parity.cu checks it against the reference on a synthetic quantized model.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::nemotron {

struct EngineOptions {
    int ctx = 131072;              ///< context cells (KV and MTP caches)
    int chunk = 1024;              ///< prompt rows per forward
    int mtp = 0;                   ///< MTP drafts per step (0: no drafting)
    int checkpoints = 6;           ///< plain checkpoints kept (the least recently used go); one more at the end of a long system prompt
    int64_t ckpt_every = 16384;    ///< a checkpoint every N tokens of a long prompt (0: none)
    int64_t root_min = 1024;       ///< the system prompt's end is kept for good when it is at least this many tokens
    int64_t park_min = 1024;       ///< a conversation of at least this many tokens is parked in RAM before another one writes over it
    int64_t park_mib = 8192;       ///< RAM for parked conversations (0: no parking)
    bool lookup = true;            ///< prompt lookup: when the context repeats itself, its continuation is the window's drafts (with mtp > 0)
    int threads = 0;               ///< CPU expert threads (0: all but one)
    std::string pool_cores;        ///< pin the CPU expert threads: "2,4,6", "2-14:2" or "physical" (the first is the engine's own thread)
    double expert_vram_gib = -1;   ///< per GPU, the most VRAM for expert slots (-1: what is free)
    int expert_slots = -1;         ///< tests: exactly this many slots per MoE layer (-1: from the VRAM)
    int reserve_mib = 1024;        ///< VRAM left free on every GPU (512 left Windows paging GPU memory: decode 7.1 against 8.3 tok/s)
    int max_gpus = 0;              ///< use at most this many visible GPUs (0: all)
    int layer_split = -1;          ///< the first block of the second GPU (-1: placed by the free VRAM)
    bool warm = true;              ///< read the routed experts once at load (into the page cache)
    bool adapt = true;             ///< move the most-used experts into the slots as it runs
    int adapt_every = 16;          ///< decode: every N tokens, up to adapt_swaps of the most-routed missing experts move into the slots (0: only after requests)
    int adapt_swaps = 40;          ///< (at most one per MoE layer each time)
    std::string profile;           ///< an expert profile (STRP, tools/make_profile.py's format): the slots start with its hottest experts
    std::string profile_save;      ///< where to save what was routed (the next start's profile): between requests and at the end
    double profile_save_min = 10;  ///< minutes between saves (0: at the end only)
    int prompt_cpu_rows = -1;      ///< a prompt chunk's non-resident expert with at most this many rows runs on the CPU (0: off)
    ///< -1: scaled per GPU by its measured link (a slow link sends more to the CPU)
    bool pin_experts = true;       ///< copy the routed experts into page-locked RAM at load (faster copies, the decode's PCIe share)
    double pcie_frac = -1;         ///< decode: the share of missed experts the GPU copies (0..1; -1: from the link and the CPU's speed)
    double pin_gib = -1;           ///< the most GiB of experts page-locked (registered), the fast-link GPU's layers first (-1: 36 on Windows; the rest
                                   ///< stay in the mapped file, read by the CPU as fast, without a PCIe share)
    int mtp_slots = 0;             ///< VRAM slots for the MTP block's own routed experts, on its GPU (0: none, every draft's experts on the
                                   ///< CPU or over the link); they come out of that GPU's room for the other layers' slots
    bool gpu_sample = true;        ///< sampled requests: the top-k logits picked on the GPU, only those copied (false: every logit copied)
    bool share_adapt = true;       ///< decode: each GPU's PCIe share follows the measured waits (the GPU later than the CPU: less), unless
                                   ///< pcie_frac fixes it
    bool prefill_pipe = true;      ///< a prompt of several chunks on several GPUs: each GPU reads its layers of the next chunk while the
                                   ///< next GPU reads the chunk before (false, or STRATA_NEMO_PIPE=0: one chunk through all GPUs at a time)
    bool verbose = true;
};

struct EngineImpl;

class Engine {
public:
    static std::unique_ptr<Engine> load(const std::string& gguf, const EngineOptions& o, std::string& err);
    ~Engine();

    /// Forget the held state (position 0).
    void reset();
    /// Read `ids` (the whole prompt: the held state continues when it is a prefix, else a checkpoint or a reset); `last_logits`
    /// gets the logits of its last token.  Returns the prompt tokens actually read (not reused).
    int64_t feed(const std::vector<int32_t>& ids, std::vector<float>* last_logits);
    /// The same with pictures (GENI): `rows[p]` is the n_embd floats that position p reads instead of a token embedding (nullptr: the
    /// token), and ids[p] is then the cell's picture identity (image_cell_id), never a token.  The rows are read during the call only.
    int64_t feed(const std::vector<int32_t>& ids, const std::vector<const float*>& rows, std::vector<float>* last_logits);
    /// The token embeddings of `ids` (ids.size() x n_embd, dequantized as the engine reads them): tests.
    std::vector<float> token_rows(const std::vector<int32_t>& ids);
    int64_t n_embd() const;
    /// Greedy decoding of `n` tokens after `first` (the token the last logits picked, not yet read): the tokens, with the MTP drafts
    /// when EngineOptions::mtp > 0 (the same tokens either way).  `logits`, when given, gets every step's logits (n x vocab).
    std::vector<int32_t> greedy(int32_t first, int n, std::vector<float>* logits = nullptr);
    int64_t pos() const;
    int64_t vocab() const;
    /// The serve loop on stdin / stdout (READY, GEN, PP, T, DONE, STOP); returns the exit code.
    int serve();
    /// --bench-cpu: the CPU's read rate of the experts in RAM against the expert kernel's (stderr); returns the exit code.
    int bench_cpu();

private:
    Engine() = default;
    std::unique_ptr<EngineImpl> p_;
};

/// The id an image cell carries in the held state: negative (never a token), from the picture's hash `h` (its grid and its key or
/// rows) and the cell's index `j` in it, so two prompts share a prefix only where their pictures are the same.
inline int32_t image_cell_id(uint64_t h, int64_t j) {
    uint64_t x = h + 0x9E3779B97F4A7C15ull * (uint64_t) (j + 1);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x ^= x >> 31;
    return -1 - (int32_t) (x & 0x3FFFFFFFull);
}

/// The command line of strata-nemotron (and of `strata` when it switches to this family): returns the exit code.
int run(int argc, char** argv);

}  // namespace strata::nemotron
