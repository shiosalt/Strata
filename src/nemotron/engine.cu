// src/nemotron/engine.cu - see engine.hpp.
#include "engine.hpp"

#include "cpu_experts.hpp"
#include "grouped_mmvq.cuh"
#include "kernels.cuh"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/nemotron/geometry.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <array>
#include <climits>
#include <filesystem>
#include <future>
#include <thread>
#include <vector>
#if defined(_WIN32)
#include <malloc.h>   // _aligned_malloc (the expert arena)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>  // SetProcessWorkingSetSizeEx (room to lock the arena)
#endif

namespace strata::nemotron {

namespace K = strata::kernels;
namespace MMQ = strata::prefill::mmq;
namespace G = strata::nemotron::gpu;

#define CK(x)                                                                                                            \
    do {                                                                                                                 \
        cudaError_t e_ = (x);                                                                                            \
        if (e_ != cudaSuccess) {                                                                                         \
            std::fprintf(stderr, "strata-nemotron: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                                \
        }                                                                                                                \
    } while (0)

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int kDecMax = 8;            // rows of the decode path (a verify window); more rows take the prompt path
constexpr int kGroupMax = 32;         // experts of one streamed MMQ group
constexpr double kShareWaitMs = 0.1;  // decode: the GPU still busy this long after the CPU's experts means its PCIe share was too large
constexpr int kGroupRows = 8192;      // the most sorted rows of one streamed group
constexpr int kSmallRows = 256;       // the rows of the permanent buffer set (decode, short prompt chunks); a longer chunk borrows
constexpr size_t kMmqTail = 4096;     // zeroed bytes after a slot / staging buffer (MMQ reads a little past)
constexpr size_t kGemmScratch = (size_t) 32 << 20;   // FP16 elements of the prompt GEMMs' dequant scratch

size_t block_bytes(int type) {
    int be = 0, bb = 0;
    return strata::block_geometry((uint32_t) type, be, bb) ? (size_t) bb : 1;
}
size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb) || n % be) return 0;
    return (size_t) (n / be) * (size_t) bb;
}
size_t gcd_sz(size_t a, size_t b) { return b ? gcd_sz(b, a % b) : a; }
size_t lcm_sz(size_t a, size_t b) { return a / gcd_sz(a, b) * b; }
size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

template <class T> T* dalloc(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
    CK(cudaMemset(p, 0, std::max<size_t>(n, 1) * sizeof(T)));
    // the engine's streams are non-blocking: the legacy stream's memset is not ordered before their work (MMQ::iota on s.ident
    // right after its allocation lost the race now and then - zeroed after it - and the prompt experts read the wrong rows)
    CK(cudaDeviceSynchronize());
    return p;
}
template <class T> T* halloc(size_t n) {
    T* p = nullptr;
    CK(cudaHostAlloc((void**) &p, std::max<size_t>(n, 1) * sizeof(T), cudaHostAllocPortable));
    std::memset(p, 0, std::max<size_t>(n, 1) * sizeof(T));
    return p;
}

struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

/// One block's weights on its GPU, and its routed experts' places.
struct Layer {
    BlockKind kind = BlockKind::Mamba;
    int stage = 0;
    const float* attn_norm = nullptr;
    // Mamba2
    Mat ssm_in, ssm_out;
    const float *conv_w = nullptr, *conv_b = nullptr, *dt_b = nullptr, *A = nullptr, *Dv = nullptr, *ssm_norm = nullptr;
    // attention
    Mat wq, wk, wv, wo;
    // MoE
    const float *router = nullptr, *probs_b = nullptr;
    Mat lat_down, lat_up, up_sh, down_sh;
    ExpertFmt fmt;                // the CPU's view of the experts
    const uint8_t* host_up = nullptr;     // expert x's up rows at host_up + x * fmt.up_bytes: the pinned arena, else the mapped GGUF
    const uint8_t* host_down = nullptr;
    size_t down_off = 0, stride = 0;      // a slot / staging blob: [up rows | down rows at down_off], `stride` apart
    bool mmq = false;                     // the prompt path's experts on MMQ
    int n_slots = 0;                      // slot rows of the stage's pool (exp_in_slot[k] = -1: empty or lent)
    size_t pool_off = 0;                  // this layer's blob in each pool row
    uint8_t* own_pool = nullptr;          // the MTP block's slots (--mtp-slots): an allocation of its own, slot k at own_pool + k * stride
    uint8_t pin_kind = 0;                 // where host_up / host_down are: 0 the mapped GGUF, 1 registered memory, 2 cudaHostAlloc's
    std::vector<int> slot_of, exp_in_slot;
    std::vector<double> score;            // decayed routing counts (the adaptive tier)
    std::vector<double> total;            // all routing since the start (the saved profile)
    std::vector<double> dec;              // the decode's routing since the start (STRATA_NEMO_ROUTE_DUMP)
    int moe_idx = -1;                     // this MoE layer's ordinal (the profile's layer)
    G::GroupedEntry* h_ent = nullptr;     // decode: this layer's GPU entries for the grouped products (pinned), and their device copy
    G::GroupedEntry* d_ent = nullptr;
    // the MTP block
    const float *post_norm = nullptr, *enorm = nullptr, *hnorm = nullptr, *shnorm = nullptr;
    Mat eh_proj;
    // state (on the stage)
    float *conv_state = nullptr, *ssm_state = nullptr;
    float* vproj = nullptr;               // a verify window's ssm_in rows (kDecMax), for the commit
    __half *kc = nullptr, *vc = nullptr;  // attention: [Hk][ctx][D]
};

/// The buffers whose size is a number of rows (a prompt chunk or a decode window): one set for up to `rows` rows.
struct Bufs {
    float *x = nullptr, *h = nullptr, *y = nullptr, *proj = nullptr, *xbc = nullptr, *ym = nullptr, *q = nullptr, *k = nullptr,
          *v = nullptr, *att = nullptr, *logits_r = nullptr, *wts = nullptr, *lat = nullptr, *shu = nullptr, *shy = nullptr,
          *acc = nullptr, *moe = nullptr, *parts = nullptr, *ecpu = nullptr, *mq_up = nullptr, *hmain = nullptr, *hprev = nullptr,
          *cat = nullptr, *xm = nullptr, *emb = nullptr;
    int32_t *ids = nullptr, *tok = nullptr, *mask = nullptr, *rdst = nullptr, *rsrc = nullptr, *rinv = nullptr, *picks = nullptr;
    __half* x16 = nullptr;
    void *mq_x = nullptr, *mq_h = nullptr;
    int rows = 0, mq_rows = 0;
};

/// One GPU and the consecutive blocks it runs.
struct Stage {
    int dev = 0;
    int first = 0, last = 0;   // blocks [first, last)
    double h2d_gbps = 0;       // the measured host -> this GPU copy rate from pinned memory (the link)
    double pcie_frac = -1;     // decode: a fixed share of a layer's missed experts this GPU copies over the link (-1: per layer, pcie_share)
    double share_k = 1.0;      // decode: the adaptive factor on pcie_share's per-layer share (the waits move it; 1 at the start)
    long long share_down = 0, share_up = 0;   // (how often it moved each way: the detail line)
    int cpu_rows = 0;          // the prompt path: a non-resident expert with at most this many rows is computed on the CPU
    cudaStream_t st = nullptr;
    std::unique_ptr<strata::prefill::Gemm> gemm;
    std::unique_ptr<MMQ::Context> mctx;
    // activations, [tmax][...]
    float *x = nullptr, *h = nullptr, *y = nullptr, *proj = nullptr, *xbc = nullptr, *ym = nullptr, *q = nullptr, *k = nullptr,
          *v = nullptr, *att = nullptr, *logits_r = nullptr, *wts = nullptr, *lat = nullptr, *shu = nullptr, *shy = nullptr,
          *acc = nullptr, *moe = nullptr, *eu = nullptr, *parts = nullptr, *ecpu = nullptr, *attn_scr = nullptr;
    int32_t *ids = nullptr, *tok = nullptr, *mask = nullptr;
    void* q8 = nullptr;        // q8_1 images of the decode path's inputs
    void* q8b = nullptr;       // a second one (the experts' hidden rows)
    __half* x16 = nullptr;     // the prompt GEMMs' FP16 inputs
    // the prompt experts
    int32_t *rcnt = nullptr, *rcur = nullptr, *rbounds = nullptr, *rdst = nullptr, *rsrc = nullptr, *rinv = nullptr, *rmax = nullptr,
            *ident = nullptr, *gbounds = nullptr;
    void *mq_x = nullptr, *mq_h = nullptr;
    float* mq_up = nullptr;
    uint8_t* staging = nullptr;
    size_t staging_bytes = 0;
    int mq_rows = 0;           // the rows of one streamed MMQ group the current buffer set holds
    // The two buffer sets: `small` (up to t_small rows, its own allocation) for decode and short prompt chunks, `big` (the chunk) for
    // longer ones.  With a loan, `big` lies in the top lend_rows rows of the expert pool: a long chunk evicts those rows' experts
    // (they are computed elsewhere meanwhile) and the copies that put them back run on `st2` after the prompt.
    Bufs small, big;
    int t_small = 0;
    size_t big_bytes = 0;
    bool big_separate = false;  // `big` has its own allocation (no loan)
    // the expert pool: slot k of MoE layer l at pool + k * pool_row + layers[l].pool_off
    uint8_t* pool = nullptr;
    size_t pool_row = 0;
    int pool_rows = 0, lend_rows = 0;
    bool lent = false, restore_pending = false;
    std::vector<std::array<int, 3>> lent_list;   // (layer, slot, expert) evicted by the loan
    cudaStream_t st2 = nullptr;
    cudaEvent_t ev_done = nullptr, ev_restore = nullptr;
    cudaEvent_t ev_copied = nullptr;   // decode: the PCIe copies of a layer's missed experts (on st2) have landed
    int32_t* tok1 = nullptr;    // the MTP draft's token
    // the last stage: the head and the MTP block
    float *hmain = nullptr, *hlast = nullptr, *logits = nullptr, *cat = nullptr, *xm = nullptr, *emb = nullptr, *hprev = nullptr;
    int32_t* picks = nullptr;
    __half *mtp_kc = nullptr, *mtp_vc = nullptr;
    const uint8_t* embd = nullptr;   // token_embd (stage 0 and the last stage)
};

struct Checkpoint {
    std::vector<int32_t> tokens;
    std::vector<std::vector<float>> conv, ssm;   // per block (Mamba blocks only)
    std::vector<float> hlast;
    uint64_t used = 0;
    bool root = false;     // the end of a long system prompt: kept for good (a new chat with the same start begins there)
    // A parked conversation (or a session file) also holds the attention K / V of its cells [0, tokens.size()), per attention block and
    // for the MTP block, [Hk][L][D] each: it can come back after another conversation wrote over those cells.  A plain checkpoint
    // does not: it is usable only while the cells still hold its tokens (kv_tokens).
    bool parked = false;
    std::vector<std::vector<__half>> k, v;       // per block (attention blocks only), then the MTP block's at index n_layers
    size_t bytes() const {
        size_t b = tokens.size() * 4 + hlast.size() * 4;
        for (const auto& x : conv) b += x.size() * 4;
        for (const auto& x : ssm) b += x.size() * 4;
        for (const auto& x : k) b += x.size() * 2;
        for (const auto& x : v) b += x.size() * 2;
        return b;
    }
};

struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 0, last_n = 64;
    uint64_t seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};
}  // namespace

std::vector<int32_t> lookup_drafts(const std::vector<int32_t>& ctx, int max_n, int min_match);

struct EngineImpl {
    EngineOptions o;
    Geometry g;
    std::shared_ptr<strata::GgufModel> file;
    std::vector<Layer> layers;   // main blocks + the MTP block
    std::vector<Stage> stages;
    Mat head;
    const float* out_norm = nullptr;   // on the last stage
    int embd_type = -1;
    size_t embd_row = 0;
    int tmax = 0;
    int ctx = 0;
    int eos = -1, im_start = -1;
    uint64_t model_bytes = 0;   // the first shard's size (a session file's fingerprint of the model)
    int img_pad = -1;   // "<image>": the token a picture's rows take the place of (GENI)
    // GENI: the picture rows of the request being read, by absolute position (nullptr: a text cell).  Image cells carry negative
    // ids in `live` and the checkpoints (a picture's identity, see image_cell_id), so a prefix with another picture never matches.
    const std::vector<const float*>* rows = nullptr;
    bool has_mtp = false;
    std::unique_ptr<CpuExperts> pool;
    bool pinned = false;              // routed experts are in page-locked RAM (the arena), not read from the mapped file (each layer: pin_kind)
    bool all_pinned = false;          // every layer's
    std::vector<uint8_t*> arena;      // the arena's allocations
    long long pcie_ent = 0;           // decode: missed experts the GPU copied over the link
    int last_offered = 0;             // the drafts the last speculative step put in its window
    long long lk_offered = 0, lk_accepted = 0;   // prompt-lookup drafts
    double lk_rate = 0.5;             // the recent share of prompt-lookup drafts kept (decides how long a lookup window is)
    long long lk_probe = 0;
    /// the drafts for a window after x: prompt lookup when the context repeats itself (at least D tokens), else empty (the MTP drafts)
    std::vector<int32_t> lookup_window(int32_t x, int D) {
        if (!o.lookup || D <= 0) return {};
        // prose rarely repeats: while the recent lookup drafts are mostly rejected, only every 8th chance tries one
        if (lk_rate < 0.3 && (++lk_probe % 8) != 0) return {};
        std::vector<int32_t> ctx = live;
        ctx.push_back(x);
        std::vector<int32_t> lk = lookup_drafts(ctx, kDecMax - 1, 4);
        const int want = lk_rate > 0.7 ? kDecMax - 1 : D;   // a long window only while lookup drafts are mostly kept
        if ((int) lk.size() < D) return {};
        if ((int) lk.size() > want) lk.resize((size_t) want);
        return lk;
    }
    void lookup_kept(int offered, int kept) {
        lk_offered += offered;
        lk_accepted += kept;
        lk_rate = 0.8 * lk_rate + 0.2 * (offered > 0 ? (double) kept / offered : 0.0);
    }
    // host staging of the decode experts (pinned)
    int32_t* h_ids = nullptr;
    float *h_lat = nullptr, *h_cpu = nullptr;
    int32_t* h_mask = nullptr;
    int32_t* h_cnt = nullptr;    // the prompt experts' counts
    int32_t* h_rsrc = nullptr;   // the prompt path's sorted rows' token rows
    int32_t* h_gb = nullptr;     // a group's bounds
    int32_t* h_picks = nullptr;
    int32_t* h_tok = nullptr;
    float* h_logits = nullptr;
    cudaEvent_t ev = nullptr;
    // the MoE paths' host staging per stage (the prefill pipeline runs the stages at once): stage 0's are the buffers above, the others
    // have their own when the pipeline is on (else every stage uses stage 0's, one at a time)
    struct HostSet {
        int32_t *ids = nullptr, *mask = nullptr, *cnt = nullptr, *rsrc = nullptr, *gb = nullptr, *tok = nullptr;
        float *lat = nullptr, *cpu = nullptr;
    };
    std::vector<HostSet> hsets;
    HostSet& hs(int stage) { return hsets[(size_t) std::min<int>(std::max(stage, 0), (int) hsets.size() - 1)]; }
    std::vector<float*> hand_x;   // the prefill pipeline: stage i's output rows for stage i + 1 (host, tmax x n_embd)
    bool pipe_ok = false;         // the prefill pipeline can run (several stages, its buffers allocated)
    std::mutex pool_mu;           // the CPU expert pool: one run at a time
    std::mutex stat_mu;           // the counters below, added from the stages' threads
    // the counters a MoE call adds to: its own while it runs, added to the engine's at its end (under stat_mu)
    struct StatAdd {
        EngineImpl* e;
        double route = 0, cpu = 0, tail = 0;
        long long gpu = 0, cpu_n = 0, pcie = 0;
        ~StatAdd() {
            std::lock_guard<std::mutex> lk(e->stat_mu);
            e->route_ms += route;
            e->cpu_ms += cpu;
            e->tail_ms += tail;
            e->gpu_ent += gpu;
            e->cpu_ent += cpu_n;
            e->pcie_ent += pcie;
        }
    };
    void run_blocks(Stage& s, int T, int64_t p0, bool verify);
    void forward_tail(int T, int64_t p0, int heads);
    bool pipe_on() const {
        static const bool env_on = !(std::getenv("STRATA_NEMO_PIPE") && std::atoi(std::getenv("STRATA_NEMO_PIPE")) == 0);
        return pipe_ok && o.prefill_pipe && env_on;
    }
    bool feed_piped(const std::vector<int32_t>& ids, const std::vector<std::array<int64_t, 2>>& plan, size_t c0, size_t c1,
                    const std::function<bool(int64_t, int64_t)>& progress, int64_t& done, int64_t total);
    // the held state
    int64_t pos = 0;
    std::vector<int32_t> live;
    std::deque<Checkpoint> ckpts;
    uint64_t use_clock = 0;
    // stats
    long long gpu_ent = 0, cpu_ent = 0, swaps = 0;
    double cpu_ms = 0, route_ms = 0, tail_ms = 0;   // decode: CPU experts, waiting for the GPU before them / after them
    // one request's decode in detail (reset when its decoding starts): the MoE layers of each GPU, and the MTP block's experts
    struct DecStat {
        double route = 0, cpu = 0, tail = 0;
        long long gpu = 0, link = 0, cpu_n = 0, calls = 0, res = 0;   // (res: the entries whose expert was in a slot)
    };
    std::vector<DecStat> dstat;   // [stage], then the MTP block at the back
    double draft_ms = 0, verify_ms = 0, samp_ms = 0;
    long long draft_n = 0, verify_n = 0;
    void reset_detail() {
        dstat.assign(stages.size() + 1, DecStat{});
        draft_ms = verify_ms = samp_ms = 0;
        draft_n = verify_n = 0;
        prof_ms.clear();
        for (auto& p : prof_st) p.n = 0;   // (the prompt's events are not the decode's)
    }
    void print_detail(double decode_ms);
    // STRATA_NEMO_PROFILE=1: GPU time per operation during decoding.  prof(s, "op") records an event on s's stream after op; at the end
    // of each step (prof_flush, everything synchronised) the time between consecutive events of a stage goes to the later one's op.
    bool prof_on = false, prof_quiet = false;
    struct ProfStage {
        std::vector<cudaEvent_t> ev;
        std::vector<const char*> label;
        size_t n = 0;
    };
    std::vector<ProfStage> prof_st;
    std::vector<std::pair<std::string, std::vector<double>>> prof_ms;   // label -> ms per stage
    void prof(Stage& s, const char* label) {
        if (!prof_on || prof_quiet) return;
        const size_t si = (size_t) (&s - stages.data());
        if (prof_st.size() < stages.size()) prof_st.resize(stages.size());
        ProfStage& p = prof_st[si];
        use(s);   // (an event belongs to the device it is made on)
        if (p.n == p.ev.size()) {
            cudaEvent_t e;
            CK(cudaEventCreate(&e));
            p.ev.push_back(e);
            p.label.push_back(nullptr);
        }
        CK(cudaEventRecord(p.ev[p.n], s.st));
        p.label[p.n++] = label;
    }
    void prof_flush() {
        if (!prof_on) return;
        for (size_t si = 0; si < prof_st.size(); ++si) {
            ProfStage& p = prof_st[si];
            for (size_t i = 1; i < p.n; ++i) {
                float ms = 0;
                CK(cudaEventSynchronize(p.ev[i]));
                CK(cudaEventElapsedTime(&ms, p.ev[i - 1], p.ev[i]));
                auto it = std::find_if(prof_ms.begin(), prof_ms.end(), [&](const auto& x) { return x.first == p.label[i]; });
                if (it == prof_ms.end()) {
                    prof_ms.push_back({p.label[i], std::vector<double>(stages.size(), 0.0)});
                    it = prof_ms.end() - 1;
                }
                it->second[si] += ms;
            }
            p.n = 0;   // (each step starts with its own marker: the time between steps is not an operation's)
        }
    }
    // sampled requests: the top-k logits picked on the GPU (the last stage), and their host copies (pinned)
    int32_t* d_topi = nullptr;
    float* d_topv = nullptr;
    int32_t* h_topi = nullptr;
    float* h_topv = nullptr;
    std::vector<int> mtp_rank;     // the MTP block's experts, hottest first (the profile's side file)
    int mtp_layer() const {
        for (size_t l = 0; l < layers.size(); ++l)
            if (layers[l].kind == BlockKind::MtpAttentionMoe && layers[l].host_up) return (int) l;
        return -1;
    }
    int bench_cpu();
    std::atomic<bool> stop{false};

    Stage& stage_of(int l) { return stages[(size_t) layers[(size_t) l].stage]; }
    Stage& last() { return stages.back(); }
    void use(const Stage& s) { CK(cudaSetDevice(s.dev)); }

    bool load(const std::string& path, std::string& err);
    void alloc_stage(Stage& s, bool is_last);
    void place_slots();
    void warm();
    bool pin_experts();
    void probe_links();
    size_t carve(Stage& s, Bufs& b, int T, bool is_last, uint8_t* base, bool measure = false);
    void set_bufs(Stage& s, const Bufs& b);
    void lend(Stage& s, int rows_needed);
    void give_back(Stage& s);
    void land_restore(Stage& s, bool wait);
    uint8_t* slot_ptr(const Layer& L, int k) {
        if (L.own_pool) return L.own_pool + (size_t) k * L.stride;
        Stage& s = stages[(size_t) L.stage];
        return s.pool + (size_t) k * s.pool_row + L.pool_off;
    }
    /// the decode's PCIe share of GPU s follows the measured waits (Stage::share_k): not with a fixed share (--pcie-frac, tests),
    /// nor with --no-share-adapt / STRATA_NEMO_SHARE_ADAPT=0
    bool share_adapting(const Stage& s) const {
        static const bool env_on = !(std::getenv("STRATA_NEMO_SHARE_ADAPT") && std::atoi(std::getenv("STRATA_NEMO_SHARE_ADAPT")) == 0);
        return o.share_adapt && env_on && pinned && s.pcie_frac < 0;
    }
    /// the decode's share of missed experts of blob size stride that GPU s copies: the link's time per blob against the CPU's
    double pcie_share(const Stage& s, size_t stride) const {
        if (!pinned) return 0.0;
        if (s.pcie_frac >= 0) return s.pcie_frac;
        const double link_ms = (double) stride / (std::max(s.h2d_gbps, 0.1) * 1e9) * 1000.0, cpu_expert_ms = 0.125;
        const double f = cpu_expert_ms / (cpu_expert_ms + link_ms);
        return f < 0.05 ? 0.0 : f;
    }
    // forward
    void forward(const int32_t* toks, int T, int64_t p0, bool verify, int heads);
    void block(int l, int T, int64_t p0, bool verify);
    void mm(Stage& s, const Mat& w, const float* x, float* y, int T, bool quantized_input = false);
    void quant(Stage& s, const float* x, int n, int T, void* q) { K::native_quantize_q8_1(x, q, n * T, 1, s.st); }
    void mmq_q(Stage& s, const Mat& w, const void* q, float* y, int T);
    void moe(int l, int T, const float* h, float* out);
    void moe_decode(int l, int T, const float* h);
    void moe_prompt(int l, int T, const float* h);
    void mtp_fill(const int32_t* d_toks, int T, int64_t p0);
    void put_rows(float* dst, int T, int64_t p0, cudaStream_t st);
    /// `lg`: the draft's logits (all of them); `topk` > 0 instead: its top-k (ids / logits) into h_topi / h_topv
    int32_t mtp_draft(int32_t tok, int64_t cell, const float* hseed, float* hout, std::vector<float>* lg = nullptr, int topk = 0);
    void commit(int n_keep);
    void rebalance(int max_swaps, double decay = 0.5, int per_layer = 1 << 30);
    int n_moe = 0;
    std::vector<std::vector<int>> prof_rank;   // per MoE layer: the loaded profile's experts, hottest first
    bool load_profile(const std::string& path);
    bool save_profile(const std::string& path);
    void head_rows(const float* hn, int n, float* lg, int32_t* pk);
    // sessions
    void reset_state();
    void save_checkpoint(bool root = false);
    bool restore_checkpoint(const std::vector<int32_t>& ids, size_t limit);
    void capture_state(Checkpoint& c);
    void capture_kv(Checkpoint& c);
    void restore_state(const Checkpoint& c);
    void restore_kv(const Checkpoint& c);
    void park_live();
    void trim_checkpoints();
    std::string save_session(const std::string& path, size_t& bytes);
    std::string restore_session(const std::string& path, size_t& bytes);
    // the tokens whose K / V the attention cells hold now (cell i: kv_tokens[i]; kNoToken where a cell holds nothing known)
    std::vector<int32_t> kv_tokens;
    static constexpr int32_t kNoToken = INT32_MIN;
    bool kv_holds(const std::vector<int32_t>& t) const {
        return kv_tokens.size() >= t.size() && std::equal(t.begin(), t.end(), kv_tokens.begin());
    }
    int64_t feed(const std::vector<int32_t>& ids, std::vector<float>* last_logits,
                 const std::function<bool(int64_t, int64_t)>& progress);
    // decoding
    int32_t decode_step(int32_t tok, std::vector<float>* lg, int topk = 0);
    int spec_step(int32_t x, int D, std::vector<int32_t>& accepted, int32_t& next);
    int spec_step_sampled(int32_t x, int D, const Sampling& smp, const std::vector<int32_t>& hist, std::mt19937_64& rng,
                          std::vector<int32_t>& accepted, int32_t& next);
    int serve();
};

// ================================================================================================================ loading
bool EngineImpl::load(const std::string& path, std::string& err) {
    const auto t0 = Clock::now();
    file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    {
        std::error_code fe;
        model_bytes = (uint64_t) std::filesystem::file_size(path, fe);
    }
    const strata::GgufModel& gm = *file;
    if (!geometry_from_gguf(gm.meta(), g, err)) return false;
    size_t checked = 0;
    if (!check_gguf_layout(gm, g, err, &checked)) return false;
    if (!G::mamba_scan_supported((int) g.ssm_n_head, (int) g.ssm_head_dim(), (int) g.ssm_d_state, (int) g.ssm_n_group)) {
        err = "the Mamba2 sizes are not ones the scan kernel takes";
        return false;
    }
    if (g.n_head / g.n_head_kv > 16 || g.head_dim > 128) {
        err = "the attention heads are not ones the attention kernel takes (<= 16 query heads per KV head, head <= 128)";
        return false;
    }
    if (g.n_expert > 1024 || g.n_expert_used > 32) {
        err = "more than 1024 experts or 32 per token";
        return false;
    }
    has_mtp = g.n_mtp > 0;
    if (const strata::MetaValue* v = gm.meta().get("tokenizer.ggml.eos_token_id")) eos = (int) v->u;
    if (const strata::MetaValue* v = gm.meta().get("tokenizer.ggml.tokens"))
        for (size_t i = 0; i < v->items.size(); ++i)
            if (v->items[i].s == "<|im_start|>") im_start = (int) i;
            else if (v->items[i].s == "<image>") img_pad = (int) i;
    ctx = o.ctx;
    tmax = std::max(o.chunk, kDecMax);

    // ---- the GPUs and the layer split
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        err = "no CUDA GPU";
        return false;
    }
    if (o.max_gpus > 0) n_dev = std::min(n_dev, o.max_gpus);
    // the dense bytes of each block (what a stage must hold for it)
    std::vector<size_t> dense((size_t) g.n_blocks(), 0);
    for (int64_t l = 0; l < g.n_blocks(); ++l)
        for (const TensorWant& w : layer_tensors(g, l)) {
            if (w.name.find("_exps.") != std::string::npos) continue;
            size_t s = 0;
            const strata::TensorInfo* t = gm.find("blk." + std::to_string(l) + "." + w.name, &s);
            dense[(size_t) l] += strata::tensor_payload_bytes(*t);
        }
    int split = o.layer_split;
    if (n_dev >= 2 && split < 0) {
        // the split that leaves the two cards' free VRAM after their dense weights most even (the slots fill what is left)
        size_t fr[2] = {0, 0};
        for (int d = 0; d < 2; ++d) {
            CK(cudaSetDevice(d));
            size_t tot = 0;
            CK(cudaMemGetInfo(&fr[d], &tot));
        }
        double best = -1e300;
        for (int k = 1; k < g.n_layers; ++k) {
            size_t d0 = 0, d1 = 0;
            for (int l = 0; l < k; ++l) d0 += dense[(size_t) l];
            for (int l = k; l < g.n_blocks(); ++l) d1 += dense[(size_t) l];
            int m0 = 0, m1 = 0;
            for (int l = 0; l < g.n_layers; ++l) (l < k ? m0 : m1) += g.kind[(size_t) l] == BlockKind::Moe;
            const double r0 = (double) fr[0] - (double) d0, r1 = (double) fr[1] - (double) d1;
            if (r0 <= 0 || r1 <= 0 || m0 == 0 || m1 == 0) continue;
            // the slots per MoE layer are min(room / layers) on both: maximise the experts resident in all
            const double score = std::min(r0 / m0, r1 / m1) * (m0 + m1) + 1e-6 * std::min(r0, r1);
            if (score > best) {
                best = score;
                split = k;
            }
        }
        if (split < 0) split = (int) (g.n_layers / 2);
    }
    const int n_stage = n_dev >= 2 && split > 0 && split < g.n_layers ? 2 : 1;
    stages.resize((size_t) n_stage);
    stages[0].dev = 0;
    stages[0].first = 0;
    stages[0].last = n_stage == 2 ? split : (int) g.n_blocks();
    if (n_stage == 2) {
        stages[1].dev = 1;
        stages[1].first = split;
        stages[1].last = (int) g.n_blocks();
    }
    layers.resize((size_t) g.n_blocks());
    for (int l = 0; l < g.n_blocks(); ++l) {
        layers[(size_t) l].kind = g.kind[(size_t) l];
        layers[(size_t) l].stage = n_stage == 2 && l >= split ? 1 : 0;
    }

    // ---- the weights
    uint64_t bytes_dev[2] = {0, 0};
    auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) throw std::runtime_error("missing " + name);
        return t;
    };
    auto f32 = [&](int stg, const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        if (t->type != 0) throw std::runtime_error(name + " is " + t->type_name() + ", expected F32");
        CK(cudaSetDevice(stages[(size_t) stg].dev));
        float* d = dalloc<float>(t->elements());
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), t->elements() * 4, cudaMemcpyHostToDevice));
        bytes_dev[stg] += t->elements() * 4;
        return d;
    };
    auto mat = [&](int stg, const std::string& name) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        if (!K::native_mmvq_supported(w.type)) throw std::runtime_error(name + " is " + t->type_name() + ", which the native MMVQ does not take");
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        CK(cudaSetDevice(stages[(size_t) stg].dev));
        uint8_t* d = dalloc<uint8_t>(nb);
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        bytes_dev[stg] += nb;
        w.d = d;
        return w;
    };
    try {
        auto L = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
        for (int l = 0; l < g.n_blocks(); ++l) {
            Layer& y = layers[(size_t) l];
            const int s = y.stage;
            y.attn_norm = f32(s, L(l, "attn_norm.weight"));
            if (y.kind == BlockKind::Mamba) {
                y.ssm_in = mat(s, L(l, "ssm_in.weight"));
                y.ssm_out = mat(s, L(l, "ssm_out.weight"));
                y.conv_w = f32(s, L(l, "ssm_conv1d.weight"));
                y.conv_b = f32(s, L(l, "ssm_conv1d.bias"));
                y.dt_b = f32(s, L(l, "ssm_dt.bias"));
                y.A = f32(s, L(l, "ssm_a"));
                y.Dv = f32(s, L(l, "ssm_d"));
                y.ssm_norm = f32(s, L(l, "ssm_norm.weight"));
            }
            if (y.kind == BlockKind::Attention || y.kind == BlockKind::MtpAttentionMoe) {
                y.wq = mat(s, L(l, "attn_q.weight"));
                y.wk = mat(s, L(l, "attn_k.weight"));
                y.wv = mat(s, L(l, "attn_v.weight"));
                y.wo = mat(s, L(l, "attn_output.weight"));
            }
            if (y.kind == BlockKind::Moe || y.kind == BlockKind::MtpAttentionMoe) {
                y.router = f32(s, L(l, "ffn_gate_inp.weight"));
                y.probs_b = f32(s, L(l, "exp_probs_b.bias"));
                y.lat_down = mat(s, L(l, "ffn_latent_down.weight"));
                y.lat_up = mat(s, L(l, "ffn_latent_up.weight"));
                y.up_sh = mat(s, L(l, "ffn_up_shexp.weight"));
                y.down_sh = mat(s, L(l, "ffn_down_shexp.weight"));
                size_t su = 0, sd = 0;
                const strata::TensorInfo* tu = find(L(l, "ffn_up_exps.weight"), su);
                const strata::TensorInfo* td = find(L(l, "ffn_down_exps.weight"), sd);
                if (!expert_fmt((int) tu->type, (int) td->type, g.moe_latent, g.n_ff_exp, y.fmt, err)) {
                    err = L(l, "experts: ") + err;
                    return false;
                }
                if (!K::native_mmvq_supported(y.fmt.up_type) || !K::native_mmvq_supported(y.fmt.down_type)) {
                    err = L(l, "experts are ") + tu->type_name() + " / " + td->type_name() + ", which the native MMVQ does not take";
                    return false;
                }
                if (strata::tensor_payload_bytes(*tu) != y.fmt.up_bytes * (size_t) g.n_expert ||
                    strata::tensor_payload_bytes(*td) != y.fmt.down_bytes * (size_t) g.n_expert) {
                    err = L(l, "expert tensors are not n_expert whole matrices");
                    return false;
                }
                y.host_up = gm.shard(su).tensor_data(*tu);
                y.host_down = gm.shard(sd).tensor_data(*td);
                // a blob: up rows, zeroed guard, down rows, zeroed guard.  MMQ's last K tile of a matrix's last row reads past the
                // matrix (MMQ::overread_bytes); past an expert's matrix is the next one's other-typed rows, whose bytes read as this
                // type's blocks can make a NaN scale that the zero activation padding does not cancel (NaN x 0).  The guards are never
                // written: zero from the allocation on.
                const size_t al = lcm_sz(lcm_sz(block_bytes(y.fmt.up_type), block_bytes(y.fmt.down_type)), 16);
                size_t guard = 256;
                if (MMQ::built())
                    guard = std::max({guard, MMQ::overread_bytes(y.fmt.up_type, g.moe_latent), MMQ::overread_bytes(y.fmt.down_type, g.n_ff_exp)});
                y.down_off = round_up(y.fmt.up_bytes + guard, al);
                y.stride = round_up(y.down_off + y.fmt.down_bytes + guard, al);
                y.mmq = MMQ::built() && MMQ::supported(y.fmt.up_type) && MMQ::supported(y.fmt.down_type) &&
                        MMQ::fits(y.fmt.up_type, g.n_ff_exp) && MMQ::fits(y.fmt.down_type, g.moe_latent);
                y.slot_of.assign((size_t) g.n_expert, -1);
                y.score.assign((size_t) g.n_expert, 0.0);
                y.total.assign((size_t) g.n_expert, 0.0);
                y.dec.assign((size_t) g.n_expert, 0.0);
                if (y.kind == BlockKind::Moe) y.moe_idx = n_moe++;
            }
            if (y.kind == BlockKind::MtpAttentionMoe) {
                y.post_norm = f32(s, L(l, "post_attention_norm.weight"));
                y.eh_proj = mat(s, L(l, "nextn.eh_proj.weight"));
                y.enorm = f32(s, L(l, "nextn.enorm.weight"));
                y.hnorm = f32(s, L(l, "nextn.hnorm.weight"));
                y.shnorm = f32(s, L(l, "nextn.shared_head_norm.weight"));
            }
        }
        const int ls = n_stage - 1;
        out_norm = f32(ls, "output_norm.weight");
        head = mat(ls, "output.weight");
        size_t s = 0;
        const strata::TensorInfo* t = find("token_embd.weight", s);
        embd_type = (int) t->type;
        embd_row = row_bytes(embd_type, g.n_embd);
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        for (int st = 0; st < n_stage; ++st) {
            if (st != 0 && st != ls) continue;
            CK(cudaSetDevice(stages[(size_t) st].dev));
            uint8_t* d = dalloc<uint8_t>(nb);
            CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
            stages[(size_t) st].embd = d;
            bytes_dev[st] += nb;
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    for (int st = 0; st < n_stage; ++st) alloc_stage(stages[(size_t) st], st == n_stage - 1);
    // host staging
    // (a prompt chunk's rows: the prompt path's CPU experts use the same buffers, rows in expert-sorted order)
    const size_t ent = (size_t) std::max(kDecMax, tmax) * g.n_expert_used;
    h_ids = halloc<int32_t>(ent);
    h_lat = halloc<float>((size_t) std::max(kDecMax, tmax) * g.moe_latent);
    h_rsrc = halloc<int32_t>(ent);
    h_cpu = halloc<float>(ent * g.moe_latent);
    h_mask = halloc<int32_t>(ent);
    h_cnt = halloc<int32_t>((size_t) g.n_expert + 1);
    h_gb = halloc<int32_t>(kGroupMax + 1);
    h_picks = halloc<int32_t>(kDecMax);
    h_tok = halloc<int32_t>((size_t) tmax + 1);
    h_logits = halloc<float>((size_t) g.n_vocab);
    hsets.assign(1, HostSet{h_ids, h_mask, h_cnt, h_rsrc, h_gb, h_tok, h_lat, h_cpu});
    if (n_stage >= 2 && tmax > kDecMax && o.prefill_pipe) {
        // the prefill pipeline's: every other stage's MoE staging, and a hand-over of one chunk's rows between neighbours (pinned when
        // it can be had; pageable works, slower)
        bool all_pinned_h = true;
        size_t bytes = 0;
        auto hb = [&](size_t n, size_t elt) -> void* {
            void* p = nullptr;
            const size_t b = std::max<size_t>(n, 1) * elt;
            bytes += b;
            if (cudaHostAlloc(&p, b, cudaHostAllocPortable) == cudaSuccess) return p;
            cudaGetLastError();
            all_pinned_h = false;
            p = std::malloc(b);
            if (!p) throw std::bad_alloc();
            return p;
        };
        const size_t rows = (size_t) std::max(kDecMax, tmax);
        for (int st = 1; st < n_stage; ++st) {
            HostSet h;
            h.ids = (int32_t*) hb(ent, 4);
            h.lat = (float*) hb(rows * g.moe_latent, 4);
            h.rsrc = (int32_t*) hb(ent, 4);
            h.cpu = (float*) hb(ent * g.moe_latent, 4);
            h.mask = (int32_t*) hb(ent, 4);
            h.cnt = (int32_t*) hb((size_t) g.n_expert + 1, 4);
            h.gb = (int32_t*) hb(kGroupMax + 1, 4);
            h.tok = (int32_t*) hb((size_t) tmax + 1, 4);
            hsets.push_back(h);
        }
        for (int st = 0; st + 1 < n_stage; ++st) hand_x.push_back((float*) hb(rows * g.n_embd, 4));
        pipe_ok = true;
        if (o.verbose)
            std::fprintf(stderr, "strata-nemotron: prefill pipeline over %d GPUs: %.0f MiB of host buffers (%s)\n", n_stage, bytes / 1048576.0,
                         all_pinned_h ? "pinned" : "partly pageable: slower");
    }
    CK(cudaSetDevice(last().dev));
    CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    prof_on = std::getenv("STRATA_NEMO_PROFILE") && std::atoi(std::getenv("STRATA_NEMO_PROFILE")) != 0;
    d_topi = dalloc<int32_t>((size_t) kDecMax * G::kTopKMax);   // (a verify window's rows, kTopKMax each: 16 KiB)
    d_topv = dalloc<float>((size_t) kDecMax * G::kTopKMax);
    h_topi = halloc<int32_t>((size_t) kDecMax * G::kTopKMax);
    h_topv = halloc<float>((size_t) kDecMax * G::kTopKMax);
    {
        std::vector<int> cores;
        if (!o.pool_cores.empty()) {
            std::string ce;
            cores = CpuExperts::parse_cores(o.pool_cores, ce);
            if (!ce.empty()) std::fprintf(stderr, "strata-nemotron: --pool-cores %s: %s; the CPU threads are not pinned\n", o.pool_cores.c_str(), ce.c_str());
            else if (o.verbose) std::fprintf(stderr, "strata-nemotron: CPU expert threads pinned to %zu logical processors (%s)\n", cores.size(), o.pool_cores.c_str());
        }
        pool = std::make_unique<CpuExperts>(o.threads, cores);
    }
    {
        const char* pe = std::getenv("STRATA_NEMO_PIN");
        pinned = o.pin_experts && !(pe && std::atoi(pe) == 0) && pin_experts();
    }
    probe_links();
    if (!o.profile.empty() && !load_profile(o.profile) && o.verbose)
        std::fprintf(stderr, "strata-nemotron: no usable expert profile at %s: the slots start with the lowest expert ids\n", o.profile.c_str());
    place_slots();
    for (Stage& s : stages) {   // every load-time copy (the legacy stream's, pageable) has landed before the streams' first work
        use(s);
        CK(cudaDeviceSynchronize());
    }
    if (o.warm && !all_pinned) warm();   // (the arena's copy read its layers' pages already; the layers left in the file are read here)
    if (o.verbose) {
        std::fprintf(stderr, "strata-nemotron: %s: %zu tensors (layout checked); %lld blocks (+%lld MTP) on %d GPU%s", path.c_str(), checked,
                     (long long) g.n_layers, (long long) g.n_mtp, n_stage, n_stage > 1 ? "s" : "");
        if (n_stage > 1) std::fprintf(stderr, " (blocks 0-%d on GPU 0, %d-%lld on GPU 1)", split - 1, split, (long long) g.n_blocks() - 1);
        std::fprintf(stderr, "; dense weights %.2f / %.2f GiB; %.1f s\n", bytes_dev[0] / 1073741824.0, bytes_dev[1] / 1073741824.0,
                     ms_since(t0) / 1000.0);
    }
    return true;
}

void EngineImpl::alloc_stage(Stage& s, bool is_last) {
    CK(cudaSetDevice(s.dev));
    CK(cudaStreamCreateWithFlags(&s.st, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s.st2, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&s.ev_done, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&s.ev_restore, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&s.ev_copied, cudaEventDisableTiming));
    s.tok1 = dalloc<int32_t>(1);
    const size_t T = (size_t) tmax, N = (size_t) g.n_embd, Kp = (size_t) g.n_expert_used, L = (size_t) g.moe_latent;
    const size_t F = (size_t) g.n_ff_exp, FS = (size_t) g.n_ff_shexp, DI = (size_t) g.ssm_d_inner, C = (size_t) g.ssm_conv_dim();
    const size_t W_IN = (size_t) g.ssm_in_dim(), HQ = (size_t) (g.n_head * g.head_dim), HK = (size_t) (g.n_head_kv * g.head_dim);
    s.eu = dalloc<float>((size_t) kDecMax * Kp * F);
    s.attn_scr = dalloc<float>(G::attn_scratch_floats(kDecMax, (int) g.n_head, (int) g.head_dim, G::kAttnMaxSplits + 1));
    const int64_t max_in = std::max<int64_t>({g.n_embd, g.ssm_d_inner, g.n_head * g.head_dim, g.n_ff_shexp, 2 * g.n_embd, g.moe_latent});
    // (native_q8_1_bytes takes at most 8 columns: one column's image times the column count)
    const size_t q8 = std::max(K::native_q8_1_bytes((int) max_in, 1) * kDecMax, K::native_q8_1_bytes((int) F, 1) * (size_t) kDecMax * Kp);
    s.q8 = dalloc<uint8_t>(q8);
    s.q8b = dalloc<uint8_t>(q8);
    s.gemm = std::make_unique<strata::prefill::Gemm>();
    std::string gerr;
    // the scratch: the largest dense matrix of the model, at most kGemmScratch elements (a bigger one is dequantized in slices)
    int64_t biggest = 1 << 20;
    for (const Layer& y : layers)
        for (const Mat* m : {&y.ssm_in, &y.ssm_out, &y.wq, &y.wk, &y.wv, &y.wo, &y.lat_down, &y.lat_up, &y.up_sh, &y.down_sh, &y.eh_proj})
            if (m->d) biggest = std::max<int64_t>(biggest, (int64_t) m->n_in * m->n_out);
    if (!s.gemm->init(s.st, std::min<int64_t>((int64_t) kGemmScratch, biggest), gerr)) {
        std::fprintf(stderr, "strata-nemotron: the prompt GEMM: %s\n", gerr.c_str());
        std::exit(1);
    }
    // the prompt experts
    const size_t ne = (size_t) g.n_expert;
    s.rcnt = dalloc<int32_t>(ne);
    s.rcur = dalloc<int32_t>(ne);
    s.rbounds = dalloc<int32_t>(ne + 1);
    s.rmax = dalloc<int32_t>(1);
    s.ident = dalloc<int32_t>(kGroupRows);
    MMQ::iota(s.ident, kGroupRows, s.st);
    s.gbounds = dalloc<int32_t>(kGroupMax + 1);
    if (MMQ::built()) {
        s.mctx = std::make_unique<MMQ::Context>();
        size_t stride = 0;
        for (int l = s.first; l < s.last; ++l)
            if (layers[(size_t) l].kind == BlockKind::Moe) stride = std::max(stride, layers[(size_t) l].stride);
        s.staging_bytes = stride * kGroupMax;
        if (s.staging_bytes) s.staging = dalloc<uint8_t>(s.staging_bytes + kMmqTail);
    }
    // state
    for (int l = s.first; l < s.last; ++l) {
        Layer& y = layers[(size_t) l];
        if (y.kind == BlockKind::Mamba) {
            y.conv_state = dalloc<float>((size_t) (g.ssm_d_conv - 1) * C);
            y.ssm_state = dalloc<float>(DI * (size_t) g.ssm_d_state);
            y.vproj = dalloc<float>((size_t) kDecMax * W_IN);
        }
        if (y.host_up) {   // a MoE layer (or the MTP block): the decode's grouped expert entries
            y.h_ent = halloc<G::GroupedEntry>((size_t) 2 * kDecMax * Kp);   // (the up products' table, then the down products')
            y.d_ent = dalloc<G::GroupedEntry>((size_t) 2 * kDecMax * Kp);
        }
        if (y.kind == BlockKind::Attention) {
            y.kc = dalloc<__half>((size_t) g.n_head_kv * ctx * g.head_dim);
            y.vc = dalloc<__half>((size_t) g.n_head_kv * ctx * g.head_dim);
        }
    }
    if (is_last) {
        s.hlast = dalloc<float>(N);
        s.logits = dalloc<float>((size_t) kDecMax * g.n_vocab);
        if (has_mtp) {
            s.mtp_kc = dalloc<__half>((size_t) g.n_head_kv * ctx * g.head_dim);
            s.mtp_vc = dalloc<__half>((size_t) g.n_head_kv * ctx * g.head_dim);
        }
    }
    // the row-sized buffers: the small set now (decode, chunks of up to t_small rows), the chunk's set where place_slots puts it
    int small_rows = kSmallRows;
    if (const char* v = std::getenv("STRATA_NEMO_SMALL_ROWS")) small_rows = std::max(kDecMax, std::atoi(v));   // (tests)
    s.t_small = std::min(tmax, std::max(kDecMax, small_rows));
    carve(s, s.small, s.t_small, is_last, nullptr);
    set_bufs(s, s.small);
    s.big_bytes = s.t_small < tmax ? carve(s, s.big, tmax, is_last, nullptr, true) : 0;
    if (s.t_small >= tmax) s.big = s.small;
    CK(cudaStreamSynchronize(s.st));
    (void) T; (void) N; (void) FS; (void) DI; (void) C; (void) W_IN; (void) HQ; (void) HK;
}

/// The row-sized buffers of a set for T rows: each allocated (base == nullptr), laid out from `base` (256-byte aligned), or only counted
/// (`measure`).  Returns the bytes.
size_t EngineImpl::carve(Stage& s, Bufs& b, int Ti, bool is_last, uint8_t* base, bool measure) {
    const size_t T = (size_t) Ti, N = (size_t) g.n_embd, Kp = (size_t) g.n_expert_used, L = (size_t) g.moe_latent;
    const size_t F = (size_t) g.n_ff_exp, FS = (size_t) g.n_ff_shexp, DI = (size_t) g.ssm_d_inner, C = (size_t) g.ssm_conv_dim();
    const size_t W_IN = (size_t) g.ssm_in_dim(), HQ = (size_t) (g.n_head * g.head_dim), HK = (size_t) (g.n_head_kv * g.head_dim);
    const int64_t max_in = std::max<int64_t>({g.n_embd, g.ssm_d_inner, g.n_head * g.head_dim, g.n_ff_shexp, 2 * g.n_embd, g.moe_latent});
    size_t used = 0;
    auto take = [&](size_t bytes) -> void* {
        bytes = round_up(std::max<size_t>(bytes, 1), 256);
        void* p = nullptr;
        if (!measure) p = base ? (void*) (base + used) : (void*) dalloc<uint8_t>(bytes);
        used += bytes;
        return p;
    };
    auto f = [&](size_t n) { return (float*) take(n * 4); };
    auto i32 = [&](size_t n) { return (int32_t*) take(n * 4); };
    b.rows = Ti;
    b.mq_rows = (int) std::min<size_t>((size_t) kGroupRows, T * Kp);
    b.x = f(T * N); b.h = f(T * N); b.y = f(T * N); b.proj = f(T * W_IN); b.xbc = f(T * C); b.ym = f(T * DI);
    b.q = f(T * HQ); b.k = f(T * HK); b.v = f(T * HK); b.att = f(T * HQ); b.logits_r = f(T * g.n_expert);
    b.ids = i32(T * Kp); b.wts = f(T * Kp); b.lat = f(T * L); b.shu = f(T * FS); b.shy = f(T * N); b.acc = f(T * L); b.moe = f(T * N);
    b.parts = f(T * Kp * L);
    b.ecpu = f(std::max<size_t>(kDecMax, T) * Kp * L);
    b.mask = i32(std::max<size_t>(kDecMax, T) * Kp);
    b.tok = i32(T + 1);
    b.x16 = (__half*) take(T * (size_t) max_in * 2);
    b.rdst = i32(T * Kp); b.rsrc = i32(T * Kp); b.rinv = i32(T * Kp);
    if (MMQ::built()) {
        b.mq_x = take(MMQ::q8_bytes(b.mq_rows, (int64_t) L));
        b.mq_h = take(MMQ::q8_bytes(b.mq_rows, (int64_t) F));
        b.mq_up = f((size_t) b.mq_rows * F);
    }
    if (is_last) {
        b.hmain = f(T * N); b.hprev = f(T * N); b.picks = i32(T); b.cat = f(T * 2 * N); b.xm = f(T * N); b.emb = f(T * N);
    }
    (void) s;
    return used;
}

void EngineImpl::set_bufs(Stage& s, const Bufs& b) {
    s.x = b.x; s.h = b.h; s.y = b.y; s.proj = b.proj; s.xbc = b.xbc; s.ym = b.ym; s.q = b.q; s.k = b.k; s.v = b.v; s.att = b.att;
    s.logits_r = b.logits_r; s.ids = b.ids; s.wts = b.wts; s.lat = b.lat; s.shu = b.shu; s.shy = b.shy; s.acc = b.acc; s.moe = b.moe;
    s.parts = b.parts; s.ecpu = b.ecpu; s.mask = b.mask; s.tok = b.tok; s.x16 = b.x16; s.rdst = b.rdst; s.rsrc = b.rsrc; s.rinv = b.rinv;
    s.mq_x = b.mq_x; s.mq_h = b.mq_h; s.mq_up = b.mq_up; s.mq_rows = b.mq_rows;
    s.hmain = b.hmain; s.hprev = b.hprev; s.picks = b.picks; s.cat = b.cat; s.xm = b.xm; s.emb = b.emb;
}

/// A chunk longer than the small set: the stage switches to the big set; with a loan, the experts in the pool's top lend_rows rows leave
/// their slots first (they are computed from RAM until give_back puts them back).
void EngineImpl::lend(Stage& s, int rows_needed) {
    if (s.lent || s.t_small >= tmax || rows_needed <= s.t_small) return;
    use(s);
    land_restore(s, true);   // (the last loan's copies write the rows this one lends)
    if (!s.big_separate) {
        // only the pool rows this chunk's buffers take (a 305-token prompt: 4 rows, not the 46 a 4096-token chunk needs)
        const int T = std::min(tmax, std::max(rows_needed, s.t_small + 1));
        const size_t need = carve(s, s.big, T, &s == &stages.back(), nullptr, true);
        const int rows = std::min(s.lend_rows, (int) ((need + s.pool_row - 1) / s.pool_row));
        carve(s, s.big, T, &s == &stages.back(), s.pool + (size_t) (s.pool_rows - rows) * s.pool_row);
        s.lent_list.clear();
        for (int l = s.first; l < s.last; ++l) {
            Layer& L = layers[(size_t) l];
            if (L.kind != BlockKind::Moe) continue;
            for (int k = s.pool_rows - rows; k < L.n_slots; ++k) {
                const int x = L.exp_in_slot[(size_t) k];
                if (x < 0) continue;
                s.lent_list.push_back(std::array<int, 3>{l, k, x});
                L.slot_of[(size_t) x] = -1;
                L.exp_in_slot[(size_t) k] = -1;
            }
        }
    }
    set_bufs(s, s.big);
    s.lent = true;
}

/// After the prompt: the small set again, and the lent rows' experts copied back on st2 once the prompt's kernels are done with the
/// rows; they count as resident when the copies have landed (land_restore), so decoding starts at once.
void EngineImpl::give_back(Stage& s) {
    if (!s.lent) return;
    use(s);
    set_bufs(s, s.small);
    s.lent = false;
    if (s.big_separate || s.lent_list.empty()) return;
    CK(cudaEventRecord(s.ev_done, s.st));
    CK(cudaStreamWaitEvent(s.st2, s.ev_done, 0));
    for (const auto& [l, k, x] : s.lent_list) {
        const Layer& L = layers[(size_t) l];
        uint8_t* dst = slot_ptr(L, k);
        CK(cudaMemcpyAsync(dst, L.host_up + (size_t) x * L.fmt.up_bytes, L.fmt.up_bytes, cudaMemcpyHostToDevice, s.st2));
        CK(cudaMemcpyAsync(dst + L.down_off, L.host_down + (size_t) x * L.fmt.down_bytes, L.fmt.down_bytes, cudaMemcpyHostToDevice, s.st2));
        // the guards between and after the matrices: the big set wrote over them
        CK(cudaMemsetAsync(dst + L.fmt.up_bytes, 0, L.down_off - L.fmt.up_bytes, s.st2));
        CK(cudaMemsetAsync(dst + L.down_off + L.fmt.down_bytes, 0, L.stride - L.down_off - L.fmt.down_bytes, s.st2));
    }
    CK(cudaEventRecord(s.ev_restore, s.st2));
    s.restore_pending = true;
}

/// The lent experts count as resident again once their copies landed (`wait`: block until they have).
void EngineImpl::land_restore(Stage& s, bool wait) {
    if (!s.restore_pending) return;
    if (wait) CK(cudaEventSynchronize(s.ev_restore));
    else {
        const cudaError_t q = cudaEventQuery(s.ev_restore);
        if (q == cudaErrorNotReady) return;
        CK(q);
    }
    for (const auto& [l, k, x] : s.lent_list) {
        Layer& L = layers[(size_t) l];
        if (L.slot_of[(size_t) x] >= 0 || L.exp_in_slot[(size_t) k] >= 0) continue;   // (moved meanwhile: the copy is stale)
        L.slot_of[(size_t) x] = k;
        L.exp_in_slot[(size_t) k] = x;
    }
    s.lent_list.clear();
    s.restore_pending = false;
}

void EngineImpl::place_slots() {
    const int ml = mtp_layer();
    for (Stage& s : stages) {
        use(s);
        // --mtp-slots: the MTP block's own slots first, so the room measured below (the other layers' pool) is what is left.  Every
        // draft runs that block's experts one row at a time, one draft after the other: what it misses is on the critical path.
        if (ml >= 0 && o.mtp_slots > 0 && &stages[(size_t) layers[(size_t) ml].stage] == &s && !layers[(size_t) ml].own_pool) {
            Layer& y = layers[(size_t) ml];
            const int S = (int) std::min<int64_t>(o.mtp_slots, g.n_expert);
            const size_t bytes = (size_t) S * y.stride + kMmqTail;
            if (cudaMalloc(&y.own_pool, bytes) != cudaSuccess) {
                cudaGetLastError();
                y.own_pool = nullptr;
                if (o.verbose) std::fprintf(stderr, "strata-nemotron: --mtp-slots %d: %.0f MiB of VRAM could not be had; the MTP block keeps no slots\n",
                                            S, bytes / 1048576.0);
            } else {
                CK(cudaMemset(y.own_pool, 0, bytes));
                CK(cudaDeviceSynchronize());
                y.n_slots = S;
                y.exp_in_slot.assign((size_t) S, -1);
                std::vector<int> order;
                std::vector<char> taken((size_t) g.n_expert, 0);
                for (int x : mtp_rank)
                    if (x >= 0 && x < g.n_expert && !taken[(size_t) x]) { order.push_back(x); taken[(size_t) x] = 1; }
                const bool from_profile = !order.empty();
                for (int x = 0; x < g.n_expert; ++x)
                    if (!taken[(size_t) x]) order.push_back(x);
                for (int k = 0; k < S; ++k) {
                    const int x = order[(size_t) k];
                    uint8_t* dst = slot_ptr(y, k);
                    CK(cudaMemcpy(dst, y.host_up + (size_t) x * y.fmt.up_bytes, y.fmt.up_bytes, cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(dst + y.down_off, y.host_down + (size_t) x * y.fmt.down_bytes, y.fmt.down_bytes, cudaMemcpyHostToDevice));
                    y.slot_of[(size_t) x] = k;
                    y.exp_in_slot[(size_t) k] = x;
                }
                if (o.verbose)
                    std::fprintf(stderr, "strata-nemotron: GPU %d: the MTP block's %d of %lld experts in VRAM (%.0f MiB, %s)\n", s.dev, S,
                                 (long long) g.n_expert, bytes / 1048576.0, from_profile ? "its hottest from the profile" : "no profile of it yet: the lowest ids");
            }
        }
        std::vector<int> moe;
        for (int l = s.first; l < s.last; ++l)
            if (layers[(size_t) l].kind == BlockKind::Moe) moe.push_back(l);
        if (moe.empty()) continue;
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        double room = (double) fr - (double) o.reserve_mib * 1048576.0 - (double) moe.size() * kMmqTail;
        if (o.expert_vram_gib >= 0) room = std::min(room, o.expert_vram_gib * 1073741824.0);
        // one pool row = every MoE layer's blob of one slot, so the top rows (the loan) are one contiguous range
        s.pool_row = 0;
        for (int l : moe) {
            layers[(size_t) l].pool_off = s.pool_row;
            s.pool_row += round_up(layers[(size_t) l].stride, 256);
        }
        int S = room > 0 ? (int) std::min<double>((double) g.n_expert, room / (double) s.pool_row) : 0;
        if (o.expert_slots >= 0) S = (int) std::min<int64_t>(o.expert_slots, g.n_expert);
        // the big buffer set: in the pool's top rows when the pool is big enough to lend them and keep 16 rows, else on its own
        s.lend_rows = s.big_bytes ? (int) ((s.big_bytes + s.pool_row - 1) / s.pool_row) : 0;
        const bool loan_test = std::getenv("STRATA_NEMO_LOAN_TEST") != nullptr;   // tests: lend even with too few slots
        int rows = S;
        if (s.lend_rows > 0 && (S >= s.lend_rows + 16 || loan_test)) {
            rows = std::max(S, s.lend_rows);
        } else if (s.lend_rows > 0) {
            s.lend_rows = 0;
            s.big_separate = true;
            carve(s, s.big, tmax, &s == &stages.back(), nullptr);
            CK(cudaMemGetInfo(&fr, &tot));
            room = (double) fr - (double) o.reserve_mib * 1048576.0;
            if (o.expert_vram_gib >= 0) room = std::min(room, o.expert_vram_gib * 1073741824.0);
            S = room > 0 ? (int) std::min<double>((double) g.n_expert, room / (double) s.pool_row) : 0;
            if (o.expert_slots >= 0) S = (int) std::min<int64_t>(o.expert_slots, g.n_expert);
            rows = S;
        }
        s.pool_rows = rows;
        if (rows > 0) {
            CK(cudaMalloc(&s.pool, (size_t) rows * s.pool_row + kMmqTail));
            CK(cudaMemset(s.pool, 0, (size_t) rows * s.pool_row + kMmqTail));
            CK(cudaDeviceSynchronize());   // (before the copies below and the streams' work: see dalloc)
        }
        if (s.lend_rows > 0) carve(s, s.big, tmax, &s == &stages.back(), s.pool + (size_t) (rows - s.lend_rows) * s.pool_row);
        for (int l : moe) {
            Layer& y = layers[(size_t) l];
            y.n_slots = rows;
            y.exp_in_slot.assign((size_t) rows, -1);
            // the S experts to start with: the profile's hottest of this layer, then the lowest ids (the routing moves the most-used in)
            std::vector<int> order;
            std::vector<char> taken((size_t) g.n_expert, 0);
            if (y.moe_idx >= 0 && y.moe_idx < (int) prof_rank.size())
                for (int x : prof_rank[(size_t) y.moe_idx])
                    if (x >= 0 && x < g.n_expert && !taken[(size_t) x]) { order.push_back(x); taken[(size_t) x] = 1; }
            for (int x = 0; x < g.n_expert; ++x)
                if (!taken[(size_t) x]) order.push_back(x);
            for (int k = 0; k < std::min<int>(S, (int) g.n_expert); ++k) {
                const int x = order[(size_t) k];
                uint8_t* dst = slot_ptr(y, k);
                CK(cudaMemcpy(dst, y.host_up + (size_t) x * y.fmt.up_bytes, y.fmt.up_bytes, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(dst + y.down_off, y.host_down + (size_t) x * y.fmt.down_bytes, y.fmt.down_bytes, cudaMemcpyHostToDevice));
                y.slot_of[(size_t) x] = k;
                y.exp_in_slot[(size_t) k] = x;
            }
        }
        if (o.verbose)
            std::fprintf(stderr, "strata-nemotron: GPU %d: %d of %lld experts per MoE layer in VRAM (%zu layers, %.2f GiB), the rest on the CPU; "
                         "prompt chunks over %d rows %s\n", s.dev, S, (long long) g.n_expert, moe.size(),
                         (double) rows * s.pool_row / 1073741824.0, s.t_small,
                         s.lend_rows > 0 ? ("borrow the top " + std::to_string(s.lend_rows) + " of them").c_str()
                                         : s.big_bytes ? "have buffers of their own" : "do not occur");
    }
}

/// The routed experts into page-locked RAM, one allocation per MoE layer (its up and down matrices), copied from the mapped file by 8
/// threads.  A copy from pinned memory runs at the link's full rate and asynchronously (the decode's PCIe share needs that).  False
/// (the mapped file stays the source) when the RAM cannot be locked.
bool EngineImpl::pin_experts() {
    const auto t0 = Clock::now();
    std::vector<Layer*> moe;
    size_t total = 0;
    // the last GPU's layers first (with the MTP block): that card has the fast link here and copies the most experts over it (41% of the
    // misses against 13% on the other, measured 2026-11-13), so when not every layer can be locked those are the ones that should be
    for (int si = (int) stages.size() - 1; si >= 0; --si)
        for (Layer& y : layers)
            if (y.host_up && y.stage == si) {
                moe.push_back(&y);
                total += (y.fmt.up_bytes + y.fmt.down_bytes) * (size_t) g.n_expert;
            }
    // Ordinary memory locked afterwards (cudaHostRegister) by default: the CPU reads it at the RAM's rate (measured 2026-11-13 on this
    // PC: 27.2 GB/s, against 10-16 GB/s from cudaHostAlloc's memory; the copy to the GPU 13.6 against 14.2 GB/s).
    // STRATA_NEMO_PIN_ALLOC=1: cudaHostAlloc as before.
    const char* pa = std::getenv("STRATA_NEMO_PIN_ALLOC");
    const bool host_alloc = pa && std::atoi(pa) != 0;
#if defined(_WIN32)
    if (!host_alloc) {   // the pages a process locks count against its working set's maximum: make room for the arena and some
        SIZE_T mn = 0, mx = 0;
        DWORD fl = 0;
        GetProcessWorkingSetSizeEx(GetCurrentProcess(), &mn, &mx, &fl);
        const SIZE_T want = (SIZE_T) total + ((SIZE_T) 4 << 30);
        if (mx < want && !SetProcessWorkingSetSizeEx(GetCurrentProcess(), std::max<SIZE_T>(mn, want - ((SIZE_T) 2 << 30)), want,
                                                     QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE) && o.verbose)
            std::fprintf(stderr, "strata-nemotron: the working set could not be raised to %.1f GiB (error %lu)\n", want / 1073741824.0,
                         (unsigned long) GetLastError());
    }
#endif
    std::vector<char> from_host_alloc;   // per buffer: cudaHostAlloc's (a layer that could not be registered) or registered
    auto release = [&](size_t i, uint8_t* b) {
        if (from_host_alloc[i]) {
            cudaFreeHost(b);
        } else {
            cudaHostUnregister(b);
#if defined(_WIN32)
            _aligned_free(b);
#else
            std::free(b);
#endif
        }
    };
    std::vector<uint8_t*> bufs;
    int n_alloc = 0;
    // Registered memory has a ceiling of its own on Windows (this PC: 40.5 GiB went, the next layer failed, and after a failed
    // registration even small cudaHostAlloc buffers failed): register up to a budget below it and never try past it.
#if defined(_WIN32)
    const double budget_gib = o.pin_gib >= 0 ? o.pin_gib : 36.0;
#else
    const double budget_gib = o.pin_gib >= 0 ? o.pin_gib : 1e9;   // (the ceiling was seen on Windows; elsewhere everything is tried)
#endif
    size_t locked = 0;
    for (Layer* y : moe) {
        uint8_t* p = nullptr;
        const size_t bytes = (y->fmt.up_bytes + y->fmt.down_bytes) * (size_t) g.n_expert;
        if (!host_alloc && (double) (locked + bytes) > budget_gib * 1073741824.0) break;
        locked += bytes;
        bool ok = false, ha = host_alloc;
        if (!host_alloc) {
            const size_t al = (size_t) 2 << 20, rounded = (bytes + al - 1) / al * al;
#if defined(_WIN32)
            p = (uint8_t*) _aligned_malloc(rounded, al);
#else
            p = (uint8_t*) std::aligned_alloc(al, rounded);
#endif
            // registered for its own GPU only (the one whose slots, staging and prompt copies read it): a portable registration maps
            // every layer into every GPU, and Windows caps what one GPU maps (on this PC about 40 GiB was the end with both)
            use(stages[(size_t) y->stage]);
            ok = p != nullptr && cudaHostRegister(p, rounded, cudaHostRegisterDefault) == cudaSuccess;
            if (!ok) {
                cudaGetLastError();
                if (p) {
#if defined(_WIN32)
                    _aligned_free(p);
#else
                    std::free(p);
#endif
                }
                p = nullptr;
                ha = true;   // this layer from cudaHostAlloc (the CPU reads it slower; the rest stay registered)
            }
        }
        if (host_alloc) ok = cudaHostAlloc((void**) &p, bytes, cudaHostAllocPortable) == cudaSuccess;
        if (!ok) {
            // the locked-memory limit (on this PC about 40 GiB in all, registered or cudaHostAlloc'd): this layer and the rest stay in
            // the mapped file, where the CPU reads them at the RAM's rate; only their PCIe share is lost (a pageable copy blocks the host)
            const cudaError_t why = cudaGetLastError();
            if (bufs.empty()) {
                if (o.verbose)
                    std::fprintf(stderr, "strata-nemotron: the routed experts (%.1f GiB) could not be page-locked (%s); they are read from the "
                                 "mapped file (no PCIe share)\n", total / 1073741824.0, cudaGetErrorString(why));
                return false;
            }
            break;
        }
        (void) ha;
        bufs.push_back(p);
        from_host_alloc.push_back(host_alloc ? 1 : 0);
        n_alloc += host_alloc ? 1 : 0;
    }
    if (bufs.size() < moe.size()) {
        // the limit was reached: give back the last layers locked until 1 GiB is free again, for the engine's own page-locked buffers
        // (the link probe, staging) and the CUDA runtime's
        size_t freed = 0;
        while (!bufs.empty() && freed < ((size_t) 1 << 30)) {
            const size_t i = bufs.size() - 1;
            freed += (moe[i]->fmt.up_bytes + moe[i]->fmt.down_bytes) * (size_t) g.n_expert;
            release(i, bufs[i]);
            bufs.pop_back();
            from_host_alloc.pop_back();
        }
        if (bufs.empty()) return false;
    }
    const size_t n_pin = bufs.size();
    size_t pinned_bytes = 0;
    for (size_t i = 0; i < n_pin; ++i) pinned_bytes += (moe[i]->fmt.up_bytes + moe[i]->fmt.down_bytes) * (size_t) g.n_expert;
    // the copy: (layer, up|down) parts in 64 MiB pieces over 8 threads
    struct Piece { const uint8_t* src; uint8_t* dst; size_t n; };
    std::vector<Piece> pieces;
    for (size_t i = 0; i < n_pin; ++i) {
        const size_t ub = moe[i]->fmt.up_bytes * (size_t) g.n_expert, db = moe[i]->fmt.down_bytes * (size_t) g.n_expert;
        const std::pair<const uint8_t*, uint8_t*> parts[2] = {{moe[i]->host_up, bufs[i]}, {moe[i]->host_down, bufs[i] + ub}};
        const size_t sizes[2] = {ub, db};
        for (int k = 0; k < 2; ++k)
            for (size_t o2 = 0; o2 < sizes[k]; o2 += (size_t) 64 << 20)
                pieces.push_back({parts[k].first + o2, parts[k].second + o2, std::min<size_t>((size_t) 64 << 20, sizes[k] - o2)});
    }
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int t = 0; t < 8; ++t)
        th.emplace_back([&] {
            for (size_t j; (j = next.fetch_add(1)) < pieces.size();) std::memcpy(pieces[j].dst, pieces[j].src, pieces[j].n);
        });
    for (auto& t : th) t.join();
    for (size_t i = 0; i < n_pin; ++i) {
        moe[i]->host_up = bufs[i];
        moe[i]->host_down = bufs[i] + moe[i]->fmt.up_bytes * (size_t) g.n_expert;
        moe[i]->pin_kind = from_host_alloc[i] ? 2 : 1;
    }
    all_pinned = n_pin == moe.size();
    arena = bufs;
    if (o.verbose)
        std::fprintf(stderr, "strata-nemotron: the routed experts: %.1f of %.1f GiB copied into page-locked RAM in %.1f s (%zu of %zu layers, %s)%s\n",
                     pinned_bytes / 1073741824.0, total / 1073741824.0, ms_since(t0) / 1000.0, n_pin, moe.size(),
                     n_alloc ? "cudaHostAlloc" : "registered ordinary memory",
                     all_pinned ? "" : "; the others are read from the mapped file (no PCIe share for them; --pin-gib sets the budget)");
    return true;
}

/// Each GPU's host -> device rate (64 MiB from pinned memory, the second of two copies), and from it the decode's PCIe share and the
/// prompt path's CPU rows.  A missed expert costs the link stride / rate and the CPU about kCpuExpertMs (measured: 0.12-0.13 ms per
/// expert on a Ryzen 7 5700X, DDR4); the share balances the two, as Strata's own PCIe share does (L / (L + R)).
void EngineImpl::probe_links() {
    const size_t n = (size_t) 64 << 20;
    uint8_t* hb = halloc<uint8_t>(n);
    std::vector<double> fr_env;
    if (const char* v = std::getenv("STRATA_NEMO_PCIE_FRAC"))   // "f" or "f0,f1": per GPU, for measuring
        for (const char* q = v; *q;) {
            char* e = nullptr;
            fr_env.push_back(std::strtod(q, &e));
            q = (*e == ',') ? e + 1 : e;
            if (e == q && *e) break;
        }
    std::vector<int> rows_env;
    if (const char* v = std::getenv("STRATA_NEMO_CPU_ROWS"))   // "n" or "n0,n1"
        for (const char* q = v; *q;) {
            char* e = nullptr;
            rows_env.push_back((int) std::strtol(q, &e, 10));
            q = (*e == ',') ? e + 1 : e;
            if (e == q && *e) break;
        }
    size_t typical = 0;   // the median MoE layer's blob (the largest is a Q8_0 layer, twice the others)
    {
        std::vector<size_t> st;
        for (const Layer& y : layers)
            if (y.stride) st.push_back(y.stride);
        if (!st.empty()) { std::sort(st.begin(), st.end()); typical = st[st.size() / 2]; }
    }
    for (size_t si = 0; si < stages.size(); ++si) {
        Stage& s = stages[si];
        use(s);
        uint8_t* d = dalloc<uint8_t>(n);
        double gbps = 0;
        for (int rep = 0; rep < 6; ++rep) {   // the best of 6: an idle link starts at its lowest PCIe generation and steps up under load
            const auto t0 = Clock::now();
            CK(cudaMemcpyAsync(d, hb, n, cudaMemcpyHostToDevice, s.st));
            CK(cudaStreamSynchronize(s.st));
            gbps = std::max(gbps, (double) n / (ms_since(t0) / 1000.0) / 1e9);
        }
        CK(cudaFree(d));
        s.h2d_gbps = gbps;
        s.pcie_frac = -1;   // per layer (pcie_share)
        if (o.pcie_frac >= 0) s.pcie_frac = std::clamp(o.pcie_frac, 0.0, 1.0);
        if (!fr_env.empty()) s.pcie_frac = std::clamp(fr_env[std::min(si, fr_env.size() - 1)], 0.0, 1.0);
        if (!pinned) s.pcie_frac = 0.0;   // (a pageable copy blocks the host: the CPU could not work meanwhile)
        // the prompt path: an expert's rows on the CPU cost about kCpuExpertMs each; streaming it costs link_ms.  10 rows was the
        // measured best on a 9-14 GB/s link, 20 on a 3 GB/s one (721-token prompt: 10 / 20 / 40 rows 86.6 / 94.3 / 72.7 tok/s; docs/NEMOTRON35_PROGRESS.md)
        s.cpu_rows = o.prompt_cpu_rows >= 0 ? o.prompt_cpu_rows : 64;   // (the cap; moe_prompt balances the two sides below it)
        if (!rows_env.empty()) s.cpu_rows = std::max(0, rows_env[std::min(si, rows_env.size() - 1)]);
        if (o.verbose)
            std::fprintf(stderr, "strata-nemotron: GPU %d: host -> GPU %.1f GB/s; decode copies %.0f%% of the missed experts (a typical layer), "
                         "the prompt sends experts of up to %d rows to the CPU\n", s.dev, gbps, pcie_share(s, typical) * 100.0, s.cpu_rows);
    }
    CK(cudaFreeHost(hb));
}

void EngineImpl::warm() {
    // every routed expert's pages once, so the first tokens do not wait on the disk (the OS keeps them cached)
    const auto t0 = Clock::now();
    std::vector<std::pair<const uint8_t*, size_t>> parts;
    size_t total = 0;
    for (const Layer& y : layers)
        if (y.host_up && y.pin_kind == 0) {   // (a page-locked layer's copy is in RAM already)
            parts.emplace_back(y.host_up, y.fmt.up_bytes * (size_t) g.n_expert);
            parts.emplace_back(y.host_down, y.fmt.down_bytes * (size_t) g.n_expert);
            total += parts.back().second + parts[parts.size() - 2].second;
        }
    std::atomic<size_t> next{0};
    std::atomic<unsigned> sink{0};
    const int nt = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int i = 0; i < nt; ++i)
        th.emplace_back([&] {
            unsigned acc = 0;
            for (size_t j; (j = next.fetch_add(1)) < parts.size();)
                for (size_t b = 0; b < parts[j].second; b += 4096) acc += parts[j].first[b];
            sink += acc;
        });
    for (auto& t : th) t.join();
    if (o.verbose)
        std::fprintf(stderr, "strata-nemotron: the routed experts in the mapped file (%.1f GiB) read once in %.1f s\n", total / 1073741824.0,
                     ms_since(t0) / 1000.0);
}

// ================================================================================================================ products
void EngineImpl::mmq_q(Stage& s, const Mat& w, const void* q, float* y, int T) {
    const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
    for (int c0 = 0; c0 < T; c0 += 8) {
        const int nc = std::min(8, T - c0);
        K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, y + (size_t) c0 * w.n_out, w.n_in, w.n_out, nc, s.st);
    }
}

/// y [T][n_out] = W x: the decode path (q8_1 + MMVQ) for T <= kDecMax, the FP16 GEMM above.  `quantized_input`: s.q8 already holds
/// x's q8_1 image (several matrices read one input).
void EngineImpl::mm(Stage& s, const Mat& w, const float* x, float* y, int T, bool quantized_input) {
    if (T <= kDecMax) {
        if (!quantized_input) quant(s, x, w.n_in, T, s.q8);
        mmq_q(s, w, s.q8, y, T);
    } else {
        G::to_f16(x, s.x16, (int64_t) T * w.n_in, s.st);
        s.gemm->native((const uint16_t*) s.x16, w.type, w.d, y, T, w.n_out, w.n_in);
    }
}

// ================================================================================================================ forward
/// The main model over T rows (device tokens in stage 0's s.tok, positions p0 ..): hmain (last stage) gets every row's final normed
/// hidden; the last `heads` rows get logits (last().logits) and greedy picks (last().picks).  `verify`: the Mamba states are read only
/// (commit() writes the kept rows).  The MTP block's K / V are filled for the rows (mtp_fill).
void EngineImpl::forward(const int32_t* toks, int T, int64_t p0, bool verify, int heads) {
    const int N = (int) g.n_embd;
    // an image cell's id is its picture's identity (negative): the embedding lookup reads <image>, whose row put_rows replaces
    for (int t = 0; t < T; ++t) h_tok[t] = toks[t] >= 0 ? toks[t] : std::max(0, img_pad);
    // the cells p0 .. p0 + T - 1 get these tokens' K / V (a verify window's drafts too: the cells hold them until they are written over)
    if ((int64_t) kv_tokens.size() < p0) kv_tokens.resize((size_t) p0, kNoToken);
    kv_tokens.resize((size_t) p0);
    kv_tokens.insert(kv_tokens.end(), toks, toks + T);
    for (Stage& s : stages) {   // a chunk over the small set's rows takes the big set (and its loan); lent experts that are back count again
        if (T > s.t_small) lend(s, T);   // (feed lent for its longest chunk already; this is for a caller that did not)
        else if (!s.lent) land_restore(s, false);
    }
    for (size_t si = 0; si < stages.size(); ++si) {
        Stage& s = stages[si];
        use(s);
        if (si == 0) {
            prof(s, "start");
            CK(cudaMemcpyAsync(s.tok, h_tok, (size_t) T * 4, cudaMemcpyHostToDevice, s.st));
            K::iq_embed_rows(embd_type, s.embd, embd_row, s.tok, T, N, s.x, s.st);
            put_rows(s.x, T, p0, s.st);
            prof(s, "embed");
        } else {
            // the residual from the previous card
            Stage& p = stages[si - 1];
            CK(cudaStreamSynchronize(p.st));
            CK(cudaMemcpyPeerAsync(s.x, s.dev, p.x, p.dev, (size_t) T * N * 4, s.st));
            CK(cudaStreamSynchronize(s.st));   // the previous card's x is its next chunk's buffer
            prof(s, "waits for the previous GPU");
            if (si + 1 == stages.size() && has_mtp)   // the last card's MTP fill reads the tokens
                CK(cudaMemcpyAsync(s.tok, h_tok, (size_t) T * 4, cudaMemcpyHostToDevice, s.st));
        }
        run_blocks(s, T, p0, verify);
    }
    forward_tail(T, p0, heads);
}

/// Stage s's blocks over the T rows in s.x.
void EngineImpl::run_blocks(Stage& s, int T, int64_t p0, bool verify) {
    const int N = (int) g.n_embd;
    static const bool check_finite = std::getenv("STRATA_NEMO_CHECK") != nullptr;   // debugging: the first non-finite block
    for (int l = s.first; l < s.last; ++l) {
        if (layers[(size_t) l].kind == BlockKind::MtpAttentionMoe) continue;
        block(l, T, p0, verify);
        if (check_finite) {
            std::vector<float> hx((size_t) T * N);
            CK(cudaStreamSynchronize(s.st));
            CK(cudaMemcpy(hx.data(), s.x, hx.size() * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < hx.size(); ++i)
                if (!std::isfinite(hx[i])) {
                    std::fprintf(stderr, "strata-nemotron: CHECK block %d (%s) T %d p0 %lld: x[%zu] = %g\n", l,
                                 layers[(size_t) l].kind == BlockKind::Mamba ? "Mamba2" : layers[(size_t) l].kind == BlockKind::Moe ? "MoE" : "attention",
                                 T, (long long) p0, i, (double) hx[i]);
                    break;
                }
        }
    }
}

/// After the last stage's blocks: the normed hidden rows, the last `heads` rows' logits and picks, the MTP block's K / V.
void EngineImpl::forward_tail(int T, int64_t p0, int heads) {
    const int N = (int) g.n_embd;
    Stage& ls = last();
    use(ls);
    G::rms_rows(ls.x, out_norm, ls.hmain, N, T, g.rms_eps, ls.st);
    if (heads > 0) head_rows(ls.hmain + (size_t) (T - heads) * N, heads, ls.logits, ls.picks);
    prof(ls, "head");
    if (has_mtp) mtp_fill(ls.tok, T, p0);
    prof(ls, "mtp fill");
}

void EngineImpl::head_rows(const float* hn, int n, float* lg, int32_t* pk) {
    Stage& s = last();
    for (int r0 = 0; r0 < n; r0 += kDecMax) {
        const int nr = std::min(kDecMax, n - r0);
        quant(s, hn + (size_t) r0 * g.n_embd, (int) g.n_embd, nr, s.q8);
        mmq_q(s, head, s.q8, lg + (size_t) r0 * g.n_vocab, nr);
        G::argmax_rows(lg + (size_t) r0 * g.n_vocab, (int) g.n_vocab, nr, pk + r0, s.st);
    }
}

void EngineImpl::block(int l, int T, int64_t p0, bool verify) {
    Layer& L = layers[(size_t) l];
    Stage& s = stage_of(l);
    const int N = (int) g.n_embd;
    G::rms_rows(s.x, L.attn_norm, s.h, N, T, g.rms_eps, s.st);
    prof(s, "block norm");
    if (L.kind == BlockKind::Mamba) {
        const int DI = (int) g.ssm_d_inner, C = (int) g.ssm_conv_dim(), W_IN = (int) g.ssm_in_dim();
        float* proj = verify ? L.vproj : s.proj;
        mm(s, L.ssm_in, s.h, proj, T);
        prof(s, "mamba in-proj");
        G::mamba_conv(proj, W_IN, DI, L.conv_state, L.conv_w, L.conv_b, s.xbc, C, (int) g.ssm_d_conv, T, !verify, s.st);
        prof(s, "mamba conv");
        G::mamba_scan(s.xbc, C, proj, W_IN, 0, DI + C, L.dt_b, L.A, L.Dv, L.ssm_state, s.ym, (int) g.ssm_n_head, (int) g.ssm_head_dim(),
                      (int) g.ssm_d_state, (int) g.ssm_n_group, T, !verify, s.st);
        prof(s, "mamba scan");
        G::group_rms_rows(s.ym, L.ssm_norm, DI, DI / (int) g.ssm_n_group, T, g.rms_eps, s.st);
        prof(s, "mamba group norm");
        mm(s, L.ssm_out, s.ym, s.y, T);
        prof(s, "mamba out-proj");
    } else if (L.kind == BlockKind::Attention) {
        const int HQ = (int) (g.n_head * g.head_dim);
        if (T <= kDecMax) {
            quant(s, s.h, N, T, s.q8);
            mmq_q(s, L.wq, s.q8, s.q, T);
            mmq_q(s, L.wk, s.q8, s.k, T);
            mmq_q(s, L.wv, s.q8, s.v, T);
        } else {
            mm(s, L.wq, s.h, s.q, T);
            mm(s, L.wk, s.h, s.k, T);
            mm(s, L.wv, s.h, s.v, T);
        }
        prof(s, "attn qkv");
        G::kv_store(s.k, s.v, L.kc, L.vc, ctx, (int) p0, T, (int) g.n_head_kv, (int) g.head_dim, s.st);
        // decode / verify windows: split by the cells (each row by its own count, see attention()); +1 keeps the merge form
        // for every decode row, so a row's arithmetic does not depend on the window it is in
        static const bool tc_env = !(std::getenv("STRATA_NEMO_ATTN_TC") && std::atoi(std::getenv("STRATA_NEMO_ATTN_TC")) == 0);
        if (T > kDecMax && tc_env && G::attention_tc_supported((int) g.n_head, (int) g.n_head_kv, (int) g.head_dim)) {
            // a prompt chunk: tensor cores (decode / verify windows keep the exact row-split kernel below)
            G::attention_tc(s.q, L.kc, L.vc, ctx, (int) p0, T, (int) g.n_head, (int) g.n_head_kv, (int) g.head_dim,
                            1.0f / std::sqrt((float) g.head_dim), s.att, s.st);
        } else {
            const int splits = T <= kDecMax ? G::attn_splits((int) p0 + T) + 1 : 1;
            G::attention(s.q, L.kc, L.vc, ctx, (int) p0, T, (int) g.n_head, (int) g.n_head_kv, (int) g.head_dim,
                         1.0f / std::sqrt((float) g.head_dim), splits, s.attn_scr, s.att, s.st);
        }
        prof(s, "attn attention");
        mm(s, L.wo, s.att, s.y, T);
        prof(s, "attn out-proj");
        (void) HQ;
    } else {
        moe(l, T, s.h, s.y);
    }
    G::add(s.x, s.y, (int64_t) T * N, s.st);
    prof(s, "block add");
}

// ================================================================================================================ MoE
/// out [T][N] = latent_up(sum_k w_k expert_k(latent_down h)) + shared(h), the experts on the GPU slots / the CPU (decode) or streamed
/// through MMQ (a prompt chunk).
void EngineImpl::moe(int l, int T, const float* h, float* out) {
    Layer& L = layers[(size_t) l];
    Stage& s = stage_of(l);
    const int N = (int) g.n_embd, E = (int) g.n_expert, Kp = (int) g.n_expert_used;
    G::gemv_f32(L.router, h, s.logits_r, N, E, T, s.st);
    G::router_topk(s.logits_r, E, L.probs_b, E, Kp, T, g.expert_weights_norm, g.expert_weights_scale, s.ids, s.wts, s.st);
    prof(s, "moe router");
    // the shared expert and the latent input (one q8_1 image of h for both on the decode path)
    if (T <= kDecMax) {
        quant(s, h, N, T, s.q8);
        mmq_q(s, L.lat_down, s.q8, s.lat, T);
        mmq_q(s, L.up_sh, s.q8, s.shu, T);
    } else {
        mm(s, L.lat_down, h, s.lat, T);
        mm(s, L.up_sh, h, s.shu, T);
    }
    prof(s, "moe latent-down + shared up");
    G::relu2(s.shu, (int64_t) T * g.n_ff_shexp, s.st);
    mm(s, L.down_sh, s.shu, s.shy, T);
    prof(s, "moe shared down");
    if (T <= kDecMax || !L.mmq) {
        for (int r0 = 0; r0 < T; r0 += kDecMax) {   // (a prompt chunk of a layer MMQ does not take: decode windows, slow but right)
            const int nr = std::min(kDecMax, T - r0);
            if (r0 > 0) {   // the window's rows at the front of the buffers the decode path reads
                CK(cudaMemcpyAsync(s.lat, s.lat + (size_t) r0 * g.moe_latent, (size_t) nr * g.moe_latent * 4, cudaMemcpyDeviceToDevice, s.st));
                CK(cudaMemcpyAsync(s.ids, s.ids + (size_t) r0 * Kp, (size_t) nr * Kp * 4, cudaMemcpyDeviceToDevice, s.st));
                CK(cudaMemcpyAsync(s.wts, s.wts + (size_t) r0 * Kp, (size_t) nr * Kp * 4, cudaMemcpyDeviceToDevice, s.st));
            }
            moe_decode(l, nr, h);
            G::moe_combine(s.parts, s.wts, Kp, (int) g.moe_latent, nr, s.acc + (size_t) r0 * g.moe_latent, s.st);
            prof(s, "moe combine");
        }
        if (T > kDecMax) {
            // (the copies above moved rows forward; the combine wrote each window's rows in place)
        }
    } else {
        moe_prompt(l, T, h);
        G::moe_combine(s.parts, s.wts, Kp, (int) g.moe_latent, T, s.acc, s.st, s.rinv);
    }
    static const bool check_finite = std::getenv("STRATA_NEMO_CHECK") != nullptr;
    if (check_finite) {
        auto bad = [&](const char* what, const float* d, size_t n) {
            std::vector<float> hv(n);
            CK(cudaStreamSynchronize(s.st));
            CK(cudaMemcpy(hv.data(), d, n * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < n; ++i)
                if (!std::isfinite(hv[i])) {
                    std::fprintf(stderr, "strata-nemotron: CHECK MoE %d T %d (%s): %s[%zu] = %g\n", l, T, T <= kDecMax ? "decode" : L.mmq ? "MMQ" : "windows",
                                 what, i, (double) hv[i]);
                    return;
                }
        };
        bad("lat", s.lat, (size_t) T * g.moe_latent);
        bad("wts", s.wts, (size_t) T * Kp);
        bad("shy", s.shy, (size_t) T * N);
        bad("acc", s.acc, (size_t) T * g.moe_latent);
    }
    mm(s, L.lat_up, s.acc, out, T);
    G::add(out, s.shy, (int64_t) T * N, s.st);
    prof(s, "moe latent-up");
}

/// The routed experts of T <= kDecMax rows (s.lat, s.ids): the slots' on the GPU, the others on the CPU at the same time; the
/// unweighted expert outputs in s.parts rows t * K + k.
void EngineImpl::moe_decode(int l, int T, const float* /*h*/) {
    Layer& L = layers[(size_t) l];
    Stage& s = stage_of(l);
    const int Kp = (int) g.n_expert_used, Lt = (int) g.moe_latent, F = (int) g.n_ff_exp;
    const int n_ent = T * Kp;
    // this stage's host staging and counters (the names shadow the engine's: a stage's thread of the prefill pipeline has its own)
    HostSet& hb = hs(L.stage);
    int32_t* const h_ids = hb.ids;
    int32_t* const h_mask = hb.mask;
    float* const h_lat = hb.lat;
    float* const h_cpu = hb.cpu;
    StatAdd sa{this};
    double& route_ms = sa.route;
    double& cpu_ms = sa.cpu;
    double& tail_ms = sa.tail;
    long long& gpu_ent = sa.gpu;
    long long& cpu_ent = sa.cpu_n;
    long long& pcie_ent = sa.pcie;
    CK(cudaMemcpyAsync(h_ids, s.ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, s.st));
    CK(cudaMemcpyAsync(h_lat, s.lat, (size_t) T * Lt * 4, cudaMemcpyDeviceToHost, s.st));
    quant(s, s.lat, Lt, T, s.q8);   // the experts' input, q8_1
    prof(s, "moe ids to host");
    DecStat* ds = dstat.empty() ? nullptr : &dstat[L.kind == BlockKind::MtpAttentionMoe ? stages.size() : (size_t) L.stage];
    const auto tw0 = Clock::now();
    CK(cudaStreamSynchronize(s.st));   // (the GPU's work up to this layer's routing: dense blocks, router)
    const double w_route = ms_since(tw0);
    route_ms += w_route;
    if (ds) {
        ds->route += w_route;
        ++ds->calls;
    }
    const long long pcie0 = pcie_ent;
    const size_t qbl = K::native_q8_1_bytes(Lt, 1), qbf = K::native_q8_1_bytes(F, 1);
    // the entries: in a slot (the GPU), copied over the link into the staging blobs (the GPU, the PCIe share), or on the CPU
    std::vector<CpuJob> jobs;
    std::vector<std::pair<int, const uint8_t*>> gpu;   // (entry, its blob on the device)
    std::vector<int> missed;                           // the distinct missed experts, in routing order
    for (int e = 0; e < n_ent; ++e) {
        const int x = h_ids[e];
        L.score[(size_t) x] += 1.0;
        L.total[(size_t) x] += 1.0;
        L.dec[(size_t) x] += 1.0;
        const int slot = L.slot_of.empty() ? -1 : L.slot_of[(size_t) x];
        if (slot >= 0) gpu.emplace_back(e, slot_ptr(L, slot));
        else if (std::find(missed.begin(), missed.end(), x) == missed.end()) missed.push_back(x);
    }
    int n_fetch = 0;
    const size_t n_res = gpu.size();   // (gpu[0, n_res): the experts in a slot; the copied ones come after them)
    static const bool grouped_env = !(std::getenv("STRATA_NEMO_GROUPED") && std::atoi(std::getenv("STRATA_NEMO_GROUPED")) == 0);
    static const bool overlap_env = !(std::getenv("STRATA_NEMO_COPY_OVERLAP") && std::atoi(std::getenv("STRATA_NEMO_COPY_OVERLAP")) == 0);
    const bool grouped = grouped_env && L.d_ent && G::grouped_mmvq_supported(L.fmt.up_type) && G::grouped_mmvq_supported(L.fmt.down_type);
    const bool overlap = grouped && overlap_env;
    // (a layer left in the mapped file: a pageable copy blocks the host)
    const bool adapt_share = share_adapting(s);
    const double share = L.pin_kind ? std::min(1.0, pcie_share(s, L.stride) * (adapt_share ? s.share_k : 1.0)) : 0.0;
    if (share > 0 && s.staging != nullptr && !missed.empty()) {
        // which misses cross the link is a fixed property of (layer, expert) - a hash below the share - so an expert is computed on
        // the same side whatever window it is in (an MTP window then computes what single steps compute); the staging blobs cap it
        const int cap = std::min<int>(kGroupMax, (int) (s.staging_bytes / std::max<size_t>(1, L.stride)));
        const uint32_t lim = (uint32_t) (share * 4294967295.0);
        std::stable_partition(missed.begin(), missed.end(), [&](int x) {
            uint32_t hsh = (uint32_t) x * 2654435761u ^ (uint32_t) l * 40503u;
            hsh ^= hsh >> 15;
            hsh *= 2246822519u;
            hsh ^= hsh >> 13;
            return hsh <= lim;
        });
        while (n_fetch < (int) missed.size() && n_fetch < cap) {
            const int x = missed[(size_t) n_fetch];
            uint32_t hsh = (uint32_t) x * 2654435761u ^ (uint32_t) l * 40503u;
            hsh ^= hsh >> 15;
            hsh *= 2246822519u;
            hsh ^= hsh >> 13;
            if (hsh > lim) break;
            ++n_fetch;
        }
        // the copies on the GPU's second stream, so the experts already in VRAM are computed while they cross the link (the grouped
        // path below waits for them only before the copied ones); s.st is idle here (the routing sync), so the staging blobs are free.
        // STRATA_NEMO_COPY_OVERLAP=0: on the compute stream, one after the other, as before.
        cudaStream_t cs = overlap ? s.st2 : s.st;
        for (int q = 0; q < n_fetch; ++q) {
            const int x = missed[(size_t) q];
            uint8_t* dst = s.staging + (size_t) q * L.stride;
            CK(cudaMemcpyAsync(dst, L.host_up + (size_t) x * L.fmt.up_bytes, L.fmt.up_bytes, cudaMemcpyHostToDevice, cs));
            CK(cudaMemcpyAsync(dst + L.down_off, L.host_down + (size_t) x * L.fmt.down_bytes, L.fmt.down_bytes, cudaMemcpyHostToDevice, cs));
        }
        if (overlap && n_fetch > 0) CK(cudaEventRecord(s.ev_copied, s.st2));
    }
    prof(s, "moe pcie copies");
    for (int e = 0; e < n_ent; ++e) {
        const int x = h_ids[e];
        const int slot = L.slot_of.empty() ? -1 : L.slot_of[(size_t) x];
        h_mask[e] = 0;
        if (slot >= 0) continue;
        const int q = (int) (std::find(missed.begin(), missed.begin() + n_fetch, x) - missed.begin());
        if (q < n_fetch) {
            gpu.emplace_back(e, s.staging + (size_t) q * L.stride);
            ++pcie_ent;
            continue;
        }
        h_mask[e] = 1;
        CpuJob j;
        j.up = L.host_up + (size_t) x * L.fmt.up_bytes;
        j.down = L.host_down + (size_t) x * L.fmt.down_bytes;
        j.tok = e / Kp;
        j.out = h_cpu + (size_t) e * Lt;
        jobs.push_back(j);
    }
    gpu_ent += (long long) gpu.size();
    cpu_ent += (long long) jobs.size();
    if (ds) {
        ds->gpu += (long long) gpu.size();
        ds->res += (long long) n_res;
        ds->link += pcie_ent - pcie0;
        ds->cpu_n += (long long) jobs.size();
    }
    // the GPU's: up -> relu^2 -> q8_1 -> down (queued; the CPU works meanwhile): one launch for every entry of the layer, or one an entry.
    // The grouped path keeps its hidden rows compact (gpu[i]'s in row i of s.eu / s.q8b: the up table writes row i, the down table reads
    // row i and writes the entry's s.parts row), so the experts in a slot (rows [0, n_res)) are computed whole before the wait for the
    // copied ones (rows [n_res, n)); each entry's arithmetic is the one it had.
    if (!gpu.empty() && grouped) {
        const int n = (int) gpu.size(), nr = (int) n_res;
        G::GroupedEntry* hu = L.h_ent;
        G::GroupedEntry* hd = L.h_ent + (size_t) kDecMax * Kp;
        for (int i = 0; i < n; ++i) {
            hu[i] = G::GroupedEntry{gpu[(size_t) i].second, gpu[(size_t) i].first / Kp, i};
            hd[i] = G::GroupedEntry{gpu[(size_t) i].second, i, gpu[(size_t) i].first};
        }
        CK(cudaMemcpyAsync(L.d_ent, L.h_ent, (size_t) 2 * kDecMax * Kp * sizeof(G::GroupedEntry), cudaMemcpyHostToDevice, s.st));
        const G::GroupedEntry* du = L.d_ent;
        const G::GroupedEntry* dd = L.d_ent + (size_t) kDecMax * Kp;
        auto group = [&](int a, int b) {   // entries [a, b): up, relu^2 and q8_1 of their rows, down
            if (b <= a) return;
            G::grouped_mmvq(L.fmt.up_type, du + a, b - a, 0, s.q8, qbl, false, s.eu, (size_t) F, Lt, F, s.st);
            G::relu2(s.eu + (size_t) a * F, (int64_t) (b - a) * F, s.st);
            quant(s, s.eu + (size_t) a * F, F, b - a, (char*) s.q8b + (size_t) a * qbf);
            G::grouped_mmvq(L.fmt.down_type, dd + a, b - a, L.down_off, s.q8b, qbf, false, s.parts, (size_t) Lt, F, Lt, s.st);
        };
        group(0, nr);
        if (n > nr) {
            if (overlap) CK(cudaStreamWaitEvent(s.st, s.ev_copied, 0));
            group(nr, n);
        }
        gpu.clear();   // (done: the per-entry path below has nothing left)
    }
    for (const auto& [e, blob] : gpu)
        K::native_mmvq(L.fmt.up_type, blob, (const char*) s.q8 + (size_t) (e / Kp) * qbl, s.eu + (size_t) e * F, Lt, F, 1, s.st);
    if (!gpu.empty()) {
        G::relu2(s.eu, (int64_t) n_ent * F, s.st);
        quant(s, s.eu, F, n_ent, s.q8b);
        for (const auto& [e, blob] : gpu)
            K::native_mmvq(L.fmt.down_type, blob + L.down_off, (const char*) s.q8b + (size_t) e * qbf, s.parts + (size_t) e * Lt, F, Lt, 1,
                           s.st);
    }
    prof(s, "moe gpu experts");
    if (!jobs.empty()) {
        const auto t0 = Clock::now();
        {
            std::lock_guard<std::mutex> lk(pool_mu);
            pool->run(L.fmt, h_lat, T, jobs.data(), (int) jobs.size());
        }
        const double w_cpu = ms_since(t0);
        cpu_ms += w_cpu;
        CK(cudaMemcpyAsync(s.ecpu, h_cpu, (size_t) n_ent * Lt * 4, cudaMemcpyHostToDevice, s.st));
        CK(cudaMemcpyAsync(s.mask, h_mask, (size_t) n_ent * 4, cudaMemcpyHostToDevice, s.st));
        G::merge_rows(s.parts, s.ecpu, s.mask, Lt, n_ent, s.st);
        prof(s, "moe idle for the cpu experts + merge");
        const auto tt0 = Clock::now();
        CK(cudaStreamSynchronize(s.st));   // h_cpu / h_mask are reused by the next layer
        const double w_tail = ms_since(tt0);
        tail_ms += w_tail;
        if (ds) {
            ds->cpu += w_cpu;
            ds->tail += w_tail;
        }
        // the share follows the waits: the GPU still busy when the CPU's experts were done means it took too many (less next
        // time), else it had room (more).  Only the main layers move it (an MTP draft's one row behaves otherwise); it settles where
        // both sides end together.  The factor's top keeps every layer's share at most 1.
        if (adapt_share && L.kind == BlockKind::Moe && L.pin_kind) {
            const double base = std::max(pcie_share(s, L.stride), 1e-3);
            if (w_tail > kShareWaitMs) {
                s.share_k = std::max(0.0, s.share_k * 0.97);
                ++s.share_down;
            } else {
                s.share_k = std::min(1.0 / base, std::max(s.share_k, 0.05) * 1.03);
                ++s.share_up;
            }
        }
    }
}

/// The routed experts of a prompt chunk through MMQ: the entries sorted by expert, the used experts streamed to the GPU in groups
/// (from their slot, or from the mapped file), up -> relu^2 -> down per group; s.parts rows are the sorted rows (s.rinv maps).
void EngineImpl::moe_prompt(int l, int T, const float* /*h*/) {
    Layer& L = layers[(size_t) l];
    Stage& s = stage_of(l);
    const int E = (int) g.n_expert, Kp = (int) g.n_expert_used, Lt = (int) g.moe_latent, F = (int) g.n_ff_exp;
    const int n_ent = T * Kp;
    // this stage's host staging and counters (as in moe_decode)
    HostSet& hb = hs(L.stage);
    int32_t* const h_cnt = hb.cnt;
    int32_t* const h_rsrc = hb.rsrc;
    int32_t* const h_mask = hb.mask;
    int32_t* const h_gb = hb.gb;
    float* const h_lat = hb.lat;
    float* const h_cpu = hb.cpu;
    StatAdd sa{this};
    double& cpu_ms = sa.cpu;
    long long& gpu_ent = sa.gpu;
    long long& cpu_ent = sa.cpu_n;
    G::route_sort(s.ids, n_ent, Kp, E, s.rcnt, s.rcur, s.rbounds, s.rdst, s.rsrc, s.rinv, s.rmax, s.st);
    CK(cudaMemcpyAsync(h_cnt, s.rcnt, (size_t) E * 4, cudaMemcpyDeviceToHost, s.st));
    CK(cudaStreamSynchronize(s.st));
    std::vector<int> start((size_t) E + 1, 0);
    for (int x = 0; x < E; ++x) {
        start[(size_t) x + 1] = start[(size_t) x] + h_cnt[x];
        L.score[(size_t) x] += h_cnt[x];
        L.total[(size_t) x] += h_cnt[x];
    }
    // An expert that is not in a slot and has only a few rows is computed on the CPU (from the mapped file) while the GPU
    // streams the others - copying its 2-3 MB over PCIe costs more than its rows do on the CPU.  Its rows stay in the sorted order and
    // are written over the GPU's (merge_rows); the MMQ group that spans them computes them with a neighbour's weights, unused.
    std::vector<char> on_cpu((size_t) E, 0);
    std::vector<CpuJob> jobs;
    if (s.cpu_rows > 0 && !L.slot_of.empty()) {
        // the split that ends both sides together: the non-resident experts, fewest rows first, go to the CPU while its time (about
        // cpu_a ms an expert - its weights read once - plus cpu_b ms a row) stays under what the link still has to stream (a blob
        // each); at most cpu_rows rows an expert
        static const double cpu_a = std::getenv("STRATA_NEMO_CPU_A") ? std::atof(std::getenv("STRATA_NEMO_CPU_A")) : 0.125;
        static const double cpu_b = std::getenv("STRATA_NEMO_CPU_B") ? std::atof(std::getenv("STRATA_NEMO_CPU_B")) : 0.03;
        const double link = (double) L.stride / (std::max(s.h2d_gbps, 0.1) * 1e9) * 1000.0;
        std::vector<int> cand;
        int n_out = 0;
        for (int e = 0; e < E; ++e)
            if (h_cnt[e] > 0 && L.slot_of[(size_t) e] < 0) {
                ++n_out;
                if (h_cnt[e] <= s.cpu_rows) cand.push_back(e);
            }
        std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) { return h_cnt[a] < h_cnt[b]; });
        double gpu_ms = n_out * link, cpu_ms_est = 0;
        for (int e : cand) {
            const double c = cpu_a + cpu_b * h_cnt[e];
            if (cpu_ms_est + c > gpu_ms - link) break;
            on_cpu[(size_t) e] = 1;
            cpu_ms_est += c;
            gpu_ms -= link;
        }
        bool any = false;
        for (int e = 0; e < E && !any; ++e) any = on_cpu[(size_t) e] != 0;
        if (any) {
            CK(cudaMemcpyAsync(h_rsrc, s.rsrc, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, s.st));
            CK(cudaMemcpyAsync(h_lat, s.lat, (size_t) T * Lt * 4, cudaMemcpyDeviceToHost, s.st));
            CK(cudaStreamSynchronize(s.st));
            for (int r = 0; r < n_ent; ++r) h_mask[r] = 0;
            for (int e = 0; e < E; ++e) {
                if (!on_cpu[(size_t) e]) continue;
                for (int r = start[(size_t) e]; r < start[(size_t) e + 1]; ++r) {
                    CpuJob j;
                    j.up = L.host_up + (size_t) e * L.fmt.up_bytes;
                    j.down = L.host_down + (size_t) e * L.fmt.down_bytes;
                    j.tok = h_rsrc[r];
                    j.out = h_cpu + (size_t) r * Lt;
                    jobs.push_back(j);
                    h_mask[r] = 1;
                }
            }
            cpu_ent += (long long) jobs.size();
        }
    }
    std::future<double> cpu_done;
    if (!jobs.empty())
        cpu_done = std::async(std::launch::async, [&]() {
            std::lock_guard<std::mutex> lk(pool_mu);   // (the other stage's chunk may have the pool: its time is not this layer's)
            const auto t0 = Clock::now();
            pool->run(L.fmt, h_lat, T, jobs.data(), (int) jobs.size());
            return ms_since(t0);
        });
    int x = 0;
    while (x < E) {
        // the next group: up to kGroupMax used experts on the GPU, at most kGroupRows rows (CPU experts' rows between them included)
        std::vector<int> grp;
        while (x < E && (int) grp.size() < kGroupMax) {
            if (h_cnt[x] == 0 || on_cpu[(size_t) x]) { ++x; continue; }
            if (!grp.empty() && start[(size_t) x + 1] - start[(size_t) grp.front()] > s.mq_rows) break;
            grp.push_back(x);
            ++x;
        }
        if (grp.empty()) break;
        const int row0 = start[(size_t) grp.front()];
        const int rows = start[(size_t) grp.back() + 1] - row0;
        int maxr = 0;
        for (size_t q = 0; q < grp.size(); ++q) {
            const int e = grp[q];
            h_gb[q] = start[(size_t) e] - row0;
            maxr = std::max(maxr, (q + 1 < grp.size() ? start[(size_t) grp[q + 1]] : start[(size_t) e + 1]) - start[(size_t) e]);
            uint8_t* dst = s.staging + q * L.stride;
            const int slot = L.slot_of[(size_t) e];
            if (slot >= 0) {
                CK(cudaMemcpyAsync(dst, slot_ptr(L, slot), L.stride, cudaMemcpyDeviceToDevice, s.st));
                ++gpu_ent;
            } else {
                CK(cudaMemcpyAsync(dst, L.host_up + (size_t) e * L.fmt.up_bytes, L.fmt.up_bytes, cudaMemcpyHostToDevice, s.st));
                CK(cudaMemcpyAsync(dst + L.down_off, L.host_down + (size_t) e * L.fmt.down_bytes, L.fmt.down_bytes, cudaMemcpyHostToDevice,
                                   s.st));
                // the guards zeroed: the staging buffer is shared by the layers, whose strides differ (another layer's rows lie here)
                CK(cudaMemsetAsync(dst + L.fmt.up_bytes, 0, L.down_off - L.fmt.up_bytes, s.st));
                CK(cudaMemsetAsync(dst + L.down_off + L.fmt.down_bytes, 0, L.stride - L.down_off - L.fmt.down_bytes, s.st));
                ++cpu_ent;
            }
        }
        h_gb[grp.size()] = rows;
        CK(cudaMemcpyAsync(s.gbounds, h_gb, (grp.size() + 1) * 4, cudaMemcpyHostToDevice, s.st));
        static const bool zero_q = std::getenv("STRATA_NEMO_ZERO_Q") != nullptr;   // debugging
        if (zero_q) {
            CK(cudaMemsetAsync(s.mq_x, 0, MMQ::q8_bytes(s.mq_rows, Lt), s.st));
            CK(cudaMemsetAsync(s.mq_h, 0, MMQ::q8_bytes(s.mq_rows, F), s.st));
            CK(cudaMemsetAsync(s.mq_up, 0, (size_t) s.mq_rows * F * 4, s.st));
        }
        MMQ::quantize(s.lat, s.rsrc + row0, s.mq_x, L.fmt.up_type, Lt, Lt, rows, s.st);
        MMQ::Product up;
        up.w = s.staging;
        up.type = L.fmt.up_type;
        up.w_rows = F;
        up.w_cols = Lt;
        up.expert_bytes = L.stride;
        up.n = (int) grp.size();
        up.xq = s.mq_x;
        up.bounds = s.gbounds;
        up.ids = s.ident;
        up.total_rows = rows;
        up.max_rows = maxr;
        up.dst = s.mq_up;
        up.ld_dst = F;
        s.mctx->run(up, s.st);
        G::relu2(s.mq_up, (int64_t) rows * F, s.st);
        MMQ::quantize(s.mq_up, nullptr, s.mq_h, L.fmt.down_type, F, F, rows, s.st);
        MMQ::Product dn = up;
        dn.w = s.staging + L.down_off;
        dn.type = L.fmt.down_type;
        dn.w_rows = Lt;
        dn.w_cols = F;
        dn.xq = s.mq_h;
        dn.dst = s.parts + (size_t) row0 * Lt;
        dn.ld_dst = Lt;
        s.mctx->run(dn, s.st);
        CK(cudaStreamSynchronize(s.st));   // the staging buffer and h_gb are the next group's
        static const bool check_finite = std::getenv("STRATA_NEMO_CHECK") != nullptr;
        if (check_finite) {
            std::vector<float> hu((size_t) rows * F), hp((size_t) rows * Lt);
            CK(cudaMemcpy(hu.data(), s.mq_up, hu.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hp.data(), s.parts + (size_t) row0 * Lt, hp.size() * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < hu.size(); ++i)
                if (!std::isfinite(hu[i])) {
                    std::fprintf(stderr, "strata-nemotron: CHECK MMQ layer %d group of %zu experts, %d rows: up[%zu] (row %zu) = %g\n", l, grp.size(),
                                 rows, i, i / F, (double) hu[i]);
                    break;
                }
            for (size_t i = 0; i < hp.size(); ++i)
                if (!std::isfinite(hp[i])) {
                    const int r = (int) (i / Lt);
                    size_t q = 0;
                    while (q + 1 < grp.size() && h_gb[q + 1] <= r) ++q;
                    const int e = grp[q];
                    std::vector<uint8_t> blob(L.stride);
                    CK(cudaMemcpy(blob.data(), s.staging + q * L.stride, L.stride, cudaMemcpyDeviceToHost));
                    const bool up_same = std::memcmp(blob.data(), L.host_up + (size_t) e * L.fmt.up_bytes, L.fmt.up_bytes) == 0;
                    const bool dn_same = std::memcmp(blob.data() + L.down_off, L.host_down + (size_t) e * L.fmt.down_bytes, L.fmt.down_bytes) == 0;
                    size_t nz = 0;
                    for (size_t b = L.fmt.up_bytes; b < L.down_off; ++b) nz += blob[b] != 0;
                    for (size_t b = L.down_off + L.fmt.down_bytes; b < L.stride; ++b) nz += blob[b] != 0;
                    std::fprintf(stderr, "strata-nemotron: CHECK MMQ layer %d group of %zu experts, %d rows: down[%zu] (row %d) = %g: expert %d "
                                 "(group slot %zu, rows %d..%d, VRAM slot %d): staged up %s, down %s, %zu non-zero guard bytes\n", l, grp.size(),
                                 rows, i, r, (double) hp[i], e, q, h_gb[q], h_gb[q + 1], L.slot_of[(size_t) e], up_same ? "same" : "DIFFERS",
                                 dn_same ? "same" : "DIFFERS", nz);
                    break;
                }
        }
    }
    if (cpu_done.valid()) {
        cpu_ms += cpu_done.get();
        // only the CPU's rows go up (merge_rows reads no other): each CPU expert's rows are one run of the sorted order, and runs that
        // touch are one copy.  Copying every row was n_ent x latent floats a layer (a 4,096-row chunk: 369 MB, 120 ms on a 3 GB/s
        // link).  STRATA_NEMO_CPU_ROWS_ALL=1: every row, as before.
        static const bool all_rows = std::getenv("STRATA_NEMO_CPU_ROWS_ALL") && std::atoi(std::getenv("STRATA_NEMO_CPU_ROWS_ALL")) != 0;
        if (all_rows) {
            CK(cudaMemcpyAsync(s.ecpu, h_cpu, (size_t) n_ent * Lt * 4, cudaMemcpyHostToDevice, s.st));
        } else {
            for (int e = 0; e < E;) {
                if (!on_cpu[(size_t) e] || start[(size_t) e + 1] == start[(size_t) e]) {
                    ++e;
                    continue;
                }
                const int r0 = start[(size_t) e];
                int r1 = start[(size_t) e + 1];
                int f = e + 1;
                // (the next CPU expert with rows starts where this one ends when the experts between have none)
                while (f < E && (on_cpu[(size_t) f] || start[(size_t) f + 1] == start[(size_t) f])) {
                    r1 = start[(size_t) f + 1];
                    ++f;
                }
                CK(cudaMemcpyAsync(s.ecpu + (size_t) r0 * Lt, h_cpu + (size_t) r0 * Lt, (size_t) (r1 - r0) * Lt * 4, cudaMemcpyHostToDevice, s.st));
                e = f;
            }
        }
        CK(cudaMemcpyAsync(s.mask, h_mask, (size_t) n_ent * 4, cudaMemcpyHostToDevice, s.st));
        G::merge_rows(s.parts, s.ecpu, s.mask, Lt, n_ent, s.st);
        CK(cudaStreamSynchronize(s.st));   // h_cpu / h_mask / h_lat are the next layer's
    }
}

/// GENI: the picture rows of positions p0 .. p0 + T - 1 (those that have one) over the token embeddings in `dst` (T x n_embd, on the
/// current device).  A run of cells whose rows are consecutive in host memory (one picture, row-major) goes up as one copy.
void EngineImpl::put_rows(float* dst, int T, int64_t p0, cudaStream_t st) {
    if (rows == nullptr) return;
    const size_t N = (size_t) g.n_embd;
    const int64_t n = (int64_t) rows->size();
    for (int t = 0; t < T;) {
        const int64_t p = p0 + t;
        const float* r = p < n ? (*rows)[(size_t) p] : nullptr;
        if (r == nullptr) {
            ++t;
            continue;
        }
        int t1 = t + 1;
        while (t1 < T && p0 + t1 < n && (*rows)[(size_t) (p0 + t1)] == r + (size_t) (t1 - t) * N) ++t1;
        CK(cudaMemcpyAsync(dst + (size_t) t * N, r, (size_t) (t1 - t) * N * 4, cudaMemcpyHostToDevice, st));
        t = t1;
    }
    CK(cudaStreamSynchronize(st));   // the rows are the request's pageable memory
}

// ================================================================================================================ MTP
/// The MTP block's K / V for the T rows just read at p0 ..: cell p0 - 1 + i takes (h of position p0 - 1 + i, token of row i)
/// (hlast for i = 0; cell -1 does not exist).  Then hlast = the last row's hidden.
void EngineImpl::mtp_fill(const int32_t* d_toks, int T, int64_t p0) {
    Stage& s = last();
    const Layer& M = layers[(size_t) g.n_layers];
    const int N = (int) g.n_embd;
    CK(cudaMemcpyAsync(s.hprev, s.hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, s.st));
    if (T > 1) CK(cudaMemcpyAsync(s.hprev + N, s.hmain, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, s.st));
    K::iq_embed_rows(embd_type, s.embd, embd_row, d_toks, T, N, s.emb, s.st);
    put_rows(s.emb, T, p0, s.st);   // a picture's cells: the block's "token embedding" is the picture's row, as the main model's input
    const int skip = p0 == 0 ? 1 : 0;   // (row 0 at position 0 has no cell before it)
    const int R = T - skip;
    if (R > 0) {
        G::rms_rows_ld(s.emb + (size_t) skip * N, M.enorm, s.cat, 2 * N, N, R, g.rms_eps, s.st);
        G::rms_rows_ld(s.hprev + (size_t) skip * N, M.hnorm, s.cat + N, 2 * N, N, R, g.rms_eps, s.st);
        mm(s, M.eh_proj, s.cat, s.xm, R);
        G::rms_rows(s.xm, M.attn_norm, s.h, N, R, g.rms_eps, s.st);
        if (R <= kDecMax) {
            quant(s, s.h, N, R, s.q8);
            mmq_q(s, M.wk, s.q8, s.k, R);
            mmq_q(s, M.wv, s.q8, s.v, R);
        } else {
            mm(s, M.wk, s.h, s.k, R);
            mm(s, M.wv, s.h, s.v, R);
        }
        G::kv_store(s.k, s.v, s.mtp_kc, s.mtp_vc, ctx, (int) (p0 - 1 + skip), R, (int) g.n_head_kv, (int) g.head_dim, s.st);
    }
    CK(cudaMemcpyAsync(s.hlast, s.hmain + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, s.st));
}

/// One MTP row: cell `cell` takes (the seed hidden `hseed` [N, device], token `tok`); returns the greedy draft, `hout` its seed.
int32_t EngineImpl::mtp_draft(int32_t tok, int64_t cell, const float* hseed, float* hout, std::vector<float>* lg, int topk) {
    Stage& s = last();
    use(s);
    prof_quiet = true;   // (one draft is one item of the profile)
    struct Unquiet { EngineImpl* e; Stage* s; ~Unquiet() { e->prof_quiet = false; e->prof(*s, "mtp draft"); } } unquiet{this, &s};
    const int l = (int) g.n_layers;
    const Layer& M = layers[(size_t) l];
    const int N = (int) g.n_embd;
    h_tok[0] = tok;
    CK(cudaMemcpyAsync(s.tok1, h_tok, 4, cudaMemcpyHostToDevice, s.st));
    K::iq_embed_rows(embd_type, s.embd, embd_row, s.tok1, 1, N, s.emb, s.st);
    G::rms_rows_ld(s.emb, M.enorm, s.cat, 2 * N, N, 1, g.rms_eps, s.st);
    G::rms_rows_ld(hseed, M.hnorm, s.cat + N, 2 * N, N, 1, g.rms_eps, s.st);
    mm(s, M.eh_proj, s.cat, s.x, 1);   // s.x: the MTP block's residual (the main model's is not needed now)
    G::rms_rows(s.x, M.attn_norm, s.h, N, 1, g.rms_eps, s.st);
    quant(s, s.h, N, 1, s.q8);
    mmq_q(s, M.wq, s.q8, s.q, 1);
    mmq_q(s, M.wk, s.q8, s.k, 1);
    mmq_q(s, M.wv, s.q8, s.v, 1);
    G::kv_store(s.k, s.v, s.mtp_kc, s.mtp_vc, ctx, (int) cell, 1, (int) g.n_head_kv, (int) g.head_dim, s.st);
    G::attention(s.q, s.mtp_kc, s.mtp_vc, ctx, (int) cell, 1, (int) g.n_head, (int) g.n_head_kv, (int) g.head_dim,
                 1.0f / std::sqrt((float) g.head_dim), G::attn_splits((int) cell + 1), s.attn_scr, s.att, s.st);
    mm(s, M.wo, s.att, s.y, 1);
    G::add(s.x, s.y, N, s.st);
    G::rms_rows(s.x, M.post_norm, s.h, N, 1, g.rms_eps, s.st);
    moe(l, 1, s.h, s.y);
    G::add(s.x, s.y, N, s.st);
    G::layer_norm_rows(s.x, M.shnorm, hout, N, 1, g.rms_eps, s.st);
    head_rows(hout, 1, s.logits, s.picks);
    CK(cudaMemcpyAsync(h_picks, s.picks, 4, cudaMemcpyDeviceToHost, s.st));
    if (topk > 0) {   // the draft's distribution (sampled drafting): its top-k only
        G::topk_rows(s.logits, (int) g.n_vocab, 1, topk, d_topi, d_topv, s.st);
        CK(cudaMemcpyAsync(h_topi, d_topi, (size_t) topk * 4, cudaMemcpyDeviceToHost, s.st));
        CK(cudaMemcpyAsync(h_topv, d_topv, (size_t) topk * 4, cudaMemcpyDeviceToHost, s.st));
    } else if (lg) {   // the draft's distribution: every logit
        lg->resize((size_t) g.n_vocab);
        CK(cudaMemcpyAsync(lg->data(), s.logits, (size_t) g.n_vocab * 4, cudaMemcpyDeviceToHost, s.st));
    }
    CK(cudaStreamSynchronize(s.st));
    return h_picks[0];
}

/// After a verify window: the Mamba states advance by its first n_keep rows (from the ssm_in rows kept in vproj).
void EngineImpl::commit(int n_keep) {
    const int DI = (int) g.ssm_d_inner, C = (int) g.ssm_conv_dim(), W_IN = (int) g.ssm_in_dim();
    for (int l = 0; l < g.n_layers; ++l) {
        Layer& L = layers[(size_t) l];
        if (L.kind != BlockKind::Mamba) continue;
        Stage& s = stage_of(l);
        use(s);
        G::mamba_conv(L.vproj, W_IN, DI, L.conv_state, L.conv_w, L.conv_b, s.xbc, C, (int) g.ssm_d_conv, n_keep, true, s.st);
        G::mamba_scan(s.xbc, C, L.vproj, W_IN, 0, DI + C, L.dt_b, L.A, L.Dv, L.ssm_state, s.ym, (int) g.ssm_n_head, (int) g.ssm_head_dim(),
                      (int) g.ssm_d_state, (int) g.ssm_n_group, n_keep, true, s.st);
    }
}

void EngineImpl::rebalance(int max_swaps, double decay, int per_layer) {
    if (!o.adapt) return;
    int main_left = max_swaps, mtp_left = max_swaps;   // (the MTP block has a budget of its own: it comes first and never uses the others')
    std::vector<int> order_l;
    const int ml = mtp_layer();
    if (ml >= 0 && layers[(size_t) ml].own_pool) order_l.push_back(ml);
    for (int l = 0; l < (int) layers.size(); ++l)
        if (layers[(size_t) l].kind == BlockKind::Moe) order_l.push_back(l);
    for (int li : order_l) {
        Layer& L = layers[(size_t) li];
        int& left = L.own_pool ? mtp_left : main_left;
        if (L.n_slots <= 0 || L.n_slots >= g.n_expert || left <= 0) continue;
        Stage& s = stages[(size_t) L.stage];
        use(s);
        land_restore(s, true);
        const int ne = (int) g.n_expert, S = L.n_slots;
        std::vector<int> order((size_t) ne);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return L.score[(size_t) a] > L.score[(size_t) b]; });
        std::vector<int> victims((size_t) S);
        std::iota(victims.begin(), victims.end(), 0);
        std::sort(victims.begin(), victims.end(), [&](int a, int b) {
            const int ea = L.exp_in_slot[(size_t) a], eb = L.exp_in_slot[(size_t) b];   // (an empty slot first)
            return (ea < 0 ? -1.0 : L.score[(size_t) ea]) < (eb < 0 ? -1.0 : L.score[(size_t) eb]);
        });
        size_t vi = 0;
        int here = 0;
        for (int i = 0; i < S && left > 0 && here < per_layer && vi < victims.size(); ++i) {
            const int x = order[(size_t) i];
            if (L.slot_of[(size_t) x] >= 0) continue;
            const int slot = victims[vi];
            const int y = L.exp_in_slot[(size_t) slot];
            if (y >= 0 && L.score[(size_t) y] >= L.score[(size_t) x]) break;
            ++vi;
            if (y >= 0) L.slot_of[(size_t) y] = -1;
            L.slot_of[(size_t) x] = slot;
            L.exp_in_slot[(size_t) slot] = x;
            uint8_t* dst = slot_ptr(L, slot);
            CK(cudaMemcpyAsync(dst, L.host_up + (size_t) x * L.fmt.up_bytes, L.fmt.up_bytes, cudaMemcpyHostToDevice, s.st));
            CK(cudaMemcpyAsync(dst + L.down_off, L.host_down + (size_t) x * L.fmt.down_bytes, L.fmt.down_bytes, cudaMemcpyHostToDevice, s.st));
            --left;
            ++here;
            ++swaps;
        }
        for (double& v : L.score) v *= decay;
        CK(cudaStreamSynchronize(s.st));
    }
}

// ================================================================================================================ profiles
// Strata's expert profile (tools/make_profile.py): "STRP", u32 version 1, n_layer, n_expert, n_ranked, n_ranked, then n_ranked
// (u16 layer, u16 expert) pairs hottest first, then n_layer x n_expert i32 (each pair's rank, -1 unranked).  The layer is the MoE
// layer's ordinal here (40 for Nemotron-3.5-Super), so serve/server.py's profile_shape reads it as it reads Qwen's.
bool EngineImpl::load_profile(const std::string& path) {
    // the MTP block's routing (not in the STRP file, whose layers are the MoE layers): "STRM", u32 n_expert, n_expert f64 counts
    if (std::FILE* fm = std::fopen((path + ".mtp").c_str(), "rb")) {
        char mg[4];
        uint32_t ne = 0;
        std::vector<double> cnt((size_t) g.n_expert);
        if (std::fread(mg, 1, 4, fm) == 4 && std::memcmp(mg, "STRM", 4) == 0 && std::fread(&ne, 4, 1, fm) == 1 && ne == (uint32_t) g.n_expert &&
            std::fread(cnt.data(), 8, cnt.size(), fm) == cnt.size()) {
            std::vector<int> r((size_t) g.n_expert);
            std::iota(r.begin(), r.end(), 0);
            std::stable_sort(r.begin(), r.end(), [&](int a, int b) { return cnt[(size_t) a] > cnt[(size_t) b]; });
            while (!r.empty() && cnt[(size_t) r.back()] <= 0) r.pop_back();
            mtp_rank = std::move(r);
        }
        std::fclose(fm);
    }
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    uint32_t h[5];
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "STRP", 4) == 0 && std::fread(h, 4, 5, f) == 5 && h[0] == 1 &&
              h[1] == (uint32_t) n_moe && h[2] == (uint32_t) g.n_expert;
    std::vector<std::vector<int>> rank((size_t) n_moe);
    for (uint32_t i = 0; ok && i < h[3]; ++i) {
        uint16_t pr[2];
        if (std::fread(pr, 2, 2, f) != 2) { ok = false; break; }
        if (pr[0] < n_moe && pr[1] < g.n_expert) rank[pr[0]].push_back(pr[1]);
    }
    std::fclose(f);
    if (!ok) return false;
    prof_rank = std::move(rank);
    if (o.verbose) std::fprintf(stderr, "strata-nemotron: the slots start from the expert profile %s\n", path.c_str());
    return true;
}

/// What was routed since the start (prompts and decoding), ranked over all layers; written to a temporary file and moved over `path`.
bool EngineImpl::save_profile(const std::string& path) {
    std::vector<std::pair<double, uint32_t>> pairs;
    for (const Layer& L : layers)
        if (L.moe_idx >= 0)
            for (int x = 0; x < g.n_expert; ++x)
                if (L.total[(size_t) x] > 0) pairs.emplace_back(L.total[(size_t) x], (uint32_t) L.moe_idx << 16 | (uint32_t) x);
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    std::vector<int32_t> table((size_t) n_moe * g.n_expert, -1);
    for (size_t r = 0; r < pairs.size(); ++r) table[(size_t) (pairs[r].second >> 16) * g.n_expert + (pairs[r].second & 0xffff)] = (int32_t) r;
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const uint32_t h[5] = {1, (uint32_t) n_moe, (uint32_t) g.n_expert, (uint32_t) pairs.size(), (uint32_t) pairs.size()};
    bool ok = std::fwrite("STRP", 1, 4, f) == 4 && std::fwrite(h, 4, 5, f) == 5;
    for (const auto& p : pairs) {
        const uint16_t pr[2] = {(uint16_t) (p.second >> 16), (uint16_t) (p.second & 0xffff)};
        ok = ok && std::fwrite(pr, 2, 2, f) == 2;
    }
    ok = ok && std::fwrite(table.data(), 4, table.size(), f) == table.size();
    ok = std::fclose(f) == 0 && ok;
    std::error_code ec;
    if (ok) std::filesystem::rename(tmp, path, ec);
    if (!ok || ec) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    const int ml = mtp_layer();   // the MTP block's counts beside it (load_profile's side file)
    if (ml >= 0 && std::any_of(layers[(size_t) ml].total.begin(), layers[(size_t) ml].total.end(), [](double v) { return v > 0; })) {
        const std::string mp = path + ".mtp", mt = mp + ".tmp";
        if (std::FILE* fm = std::fopen(mt.c_str(), "wb")) {
            const uint32_t ne = (uint32_t) g.n_expert;
            const std::vector<double>& cnt = layers[(size_t) ml].total;
            bool okm = std::fwrite("STRM", 1, 4, fm) == 4 && std::fwrite(&ne, 4, 1, fm) == 1 && std::fwrite(cnt.data(), 8, cnt.size(), fm) == cnt.size();
            okm = std::fclose(fm) == 0 && okm;
            if (okm) std::filesystem::rename(mt, mp, ec);
            if (!okm || ec) std::filesystem::remove(mt, ec);
        }
    }
    return true;
}

// ================================================================================================================ sessions
void EngineImpl::reset_state() {
    for (int l = 0; l < g.n_layers; ++l) {
        Layer& L = layers[(size_t) l];
        if (L.kind != BlockKind::Mamba) continue;
        // on the stage's own stream: the legacy stream's memset is not ordered with it (non-blocking), so the next prompt's first
        // Mamba scan could run on the old state before the zeroing landed
        Stage& s = stage_of(l);
        use(s);
        CK(cudaMemsetAsync(L.conv_state, 0, (size_t) (g.ssm_d_conv - 1) * g.ssm_conv_dim() * 4, s.st));
        CK(cudaMemsetAsync(L.ssm_state, 0, (size_t) g.ssm_d_inner * g.ssm_d_state * 4, s.st));
    }
    use(last());
    CK(cudaMemsetAsync(last().hlast, 0, (size_t) g.n_embd * 4, last().st));
    pos = 0;
    live.clear();
}

void EngineImpl::capture_state(Checkpoint& c) {
    c.tokens = live;
    c.conv.assign((size_t) g.n_layers, {});
    c.ssm.assign((size_t) g.n_layers, {});
    for (int l = 0; l < g.n_layers; ++l) {
        Layer& L = layers[(size_t) l];
        if (L.kind != BlockKind::Mamba) continue;
        Stage& s = stage_of(l);
        use(s);
        CK(cudaStreamSynchronize(s.st));
        c.conv[(size_t) l].resize((size_t) (g.ssm_d_conv - 1) * g.ssm_conv_dim());
        c.ssm[(size_t) l].resize((size_t) g.ssm_d_inner * g.ssm_d_state);
        CK(cudaMemcpy(c.conv[(size_t) l].data(), L.conv_state, c.conv[(size_t) l].size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(c.ssm[(size_t) l].data(), L.ssm_state, c.ssm[(size_t) l].size() * 4, cudaMemcpyDeviceToHost));
    }
    use(last());
    c.hlast.resize((size_t) g.n_embd);
    CK(cudaMemcpy(c.hlast.data(), last().hlast, (size_t) g.n_embd * 4, cudaMemcpyDeviceToHost));
}

/// The K / V of cells [0, c.tokens.size()) of every attention block and of the MTP block, into c (2-D copies: [Hk][ctx][D] -> [Hk][L][D]).
void EngineImpl::capture_kv(Checkpoint& c) {
    const size_t L = c.tokens.size(), D = (size_t) g.head_dim, Hk = (size_t) g.n_head_kv;
    c.k.assign((size_t) g.n_layers + 1, {});
    c.v.assign((size_t) g.n_layers + 1, {});
    if (L == 0) return;
    auto take = [&](int idx, Stage& s, const __half* kc, const __half* vc) {
        use(s);
        c.k[(size_t) idx].resize(Hk * L * D);
        c.v[(size_t) idx].resize(Hk * L * D);
        CK(cudaStreamSynchronize(s.st));
        CK(cudaMemcpy2DAsync(c.k[(size_t) idx].data(), L * D * 2, kc, (size_t) ctx * D * 2, L * D * 2, Hk, cudaMemcpyDeviceToHost, s.st));
        CK(cudaMemcpy2DAsync(c.v[(size_t) idx].data(), L * D * 2, vc, (size_t) ctx * D * 2, L * D * 2, Hk, cudaMemcpyDeviceToHost, s.st));
        CK(cudaStreamSynchronize(s.st));
    };
    for (int l = 0; l < g.n_layers; ++l)
        if (layers[(size_t) l].kind == BlockKind::Attention) take(l, stage_of(l), layers[(size_t) l].kc, layers[(size_t) l].vc);
    if (has_mtp && last().mtp_kc) take((int) g.n_layers, last(), last().mtp_kc, last().mtp_vc);
}

void EngineImpl::restore_state(const Checkpoint& c) {
    for (int l = 0; l < g.n_layers; ++l) {
        Layer& L = layers[(size_t) l];
        if (L.kind != BlockKind::Mamba || c.conv[(size_t) l].empty()) continue;
        // on the stage's stream (a pageable cudaMemcpy may return before its DMA lands, and the streams are non-blocking)
        Stage& s = stage_of(l);
        use(s);
        CK(cudaMemcpyAsync(L.conv_state, c.conv[(size_t) l].data(), c.conv[(size_t) l].size() * 4, cudaMemcpyHostToDevice, s.st));
        CK(cudaMemcpyAsync(L.ssm_state, c.ssm[(size_t) l].data(), c.ssm[(size_t) l].size() * 4, cudaMemcpyHostToDevice, s.st));
    }
    use(last());
    CK(cudaMemcpyAsync(last().hlast, c.hlast.data(), (size_t) g.n_embd * 4, cudaMemcpyHostToDevice, last().st));
    for (Stage& s : stages) {   // (the host vectors are the checkpoint's: the copies land before it can change)
        use(s);
        CK(cudaStreamSynchronize(s.st));
    }
    live = c.tokens;
    pos = (int64_t) live.size();
}

void EngineImpl::restore_kv(const Checkpoint& c) {
    const size_t L = c.tokens.size(), D = (size_t) g.head_dim, Hk = (size_t) g.n_head_kv;
    if (L == 0) return;
    auto put = [&](int idx, Stage& s, __half* kc, __half* vc) {
        if (c.k.size() <= (size_t) idx || c.k[(size_t) idx].size() != Hk * L * D) return;
        use(s);
        CK(cudaStreamSynchronize(s.st));
        // on the stage's stream and waited for: a pageable copy may return before its DMA lands (the streams are non-blocking)
        CK(cudaMemcpy2DAsync(kc, (size_t) ctx * D * 2, c.k[(size_t) idx].data(), L * D * 2, L * D * 2, Hk, cudaMemcpyHostToDevice, s.st));
        CK(cudaMemcpy2DAsync(vc, (size_t) ctx * D * 2, c.v[(size_t) idx].data(), L * D * 2, L * D * 2, Hk, cudaMemcpyHostToDevice, s.st));
        CK(cudaStreamSynchronize(s.st));
    };
    for (int l = 0; l < g.n_layers; ++l)
        if (layers[(size_t) l].kind == BlockKind::Attention) put(l, stage_of(l), layers[(size_t) l].kc, layers[(size_t) l].vc);
    if (has_mtp && last().mtp_kc) put((int) g.n_layers, last(), last().mtp_kc, last().mtp_vc);
    kv_tokens = c.tokens;
}

/// The held conversation (its state and its cells' K / V) into RAM before another one writes over it, so a later request that
/// continues it (a parent agent after its sub-agent) starts where it was.  Conversations shorter than park_min tokens are read again.
void EngineImpl::park_live() {
    if ((int64_t) live.size() < o.park_min || o.park_mib <= 0 || !kv_holds(live)) return;
    for (const Checkpoint& c : ckpts)
        if (c.parked && c.tokens == live) return;
    const auto t0 = Clock::now();
    Checkpoint c;
    capture_state(c);
    capture_kv(c);
    c.parked = true;
    c.used = ++use_clock;
    const size_t b = c.bytes();
    ckpts.push_back(std::move(c));
    trim_checkpoints();
    if (o.verbose)
        std::fprintf(stderr, "strata-nemotron: conversation cache: parked %zu tokens (%.0f MiB) in %.0f ms\n", live.size(), b / 1048576.0,
                     ms_since(t0));
}

/// At most o.checkpoints plain checkpoints (the least recently used go), one root, and parked ones within o.park_mib.
void EngineImpl::trim_checkpoints() {
    auto lru = [&](auto pred) {
        auto best = ckpts.end();
        for (auto it = ckpts.begin(); it != ckpts.end(); ++it)
            if (pred(*it) && (best == ckpts.end() || it->used < best->used)) best = it;
        return best;
    };
    for (;;) {
        int plain = 0, roots = 0;
        size_t parked = 0;
        for (const Checkpoint& c : ckpts) {
            if (c.parked) parked += c.bytes();
            else if (c.root) ++roots;
            else ++plain;
        }
        if (plain > o.checkpoints) ckpts.erase(lru([](const Checkpoint& c) { return !c.parked && !c.root; }));
        else if (roots > 1) ckpts.erase(lru([](const Checkpoint& c) { return !c.parked && c.root; }));
        else if (parked > (size_t) o.park_mib << 20) ckpts.erase(lru([](const Checkpoint& c) { return c.parked; }));
        else break;
    }
}

void EngineImpl::save_checkpoint(bool root) {
    Checkpoint c;
    capture_state(c);
    c.root = root;
    c.used = ++use_clock;
    for (auto it = ckpts.begin(); it != ckpts.end(); ++it)
        if (!it->parked && it->tokens == c.tokens) {
            c.root = c.root || it->root;
            ckpts.erase(it);
            break;
        }
    ckpts.push_back(std::move(c));
    trim_checkpoints();
}

/// The longest checkpoint whose tokens begin `ids`, at most `limit` tokens long, whose cells still hold its tokens' K / V (or that
/// carries them: a parked conversation): restored (true), or false.
bool EngineImpl::restore_checkpoint(const std::vector<int32_t>& ids, size_t limit) {
    Checkpoint* best = nullptr;
    for (Checkpoint& c : ckpts)
        if (c.tokens.size() <= limit && c.tokens.size() <= ids.size() && std::equal(c.tokens.begin(), c.tokens.end(), ids.begin()) &&
            (c.parked || kv_holds(c.tokens)) && (!best || c.tokens.size() > best->tokens.size()))
            best = &c;
    if (!best || best->tokens.empty()) return false;
    const auto t0 = Clock::now();
    restore_state(*best);
    if (best->parked) {
        restore_kv(*best);
        if (std::getenv("STRATA_NEMO_DEBUG_PARK")) {   // debugging: the device state read back against the parked copy
            Checkpoint back;
            capture_state(back);
            capture_kv(back);
            size_t bad_s = 0, bad_k = 0;
            for (size_t l = 0; l < back.ssm.size(); ++l) bad_s += back.ssm[l] != best->ssm[l] || back.conv[l] != best->conv[l];
            for (size_t l = 0; l < back.k.size(); ++l)
                bad_k += std::memcmp(back.k[l].data(), best->k[l].data(), back.k[l].size() * 2) != 0 || back.k[l].size() != best->k[l].size();
            std::fprintf(stderr, "strata-nemotron: DEBUG park restore of %zu tokens: %zu Mamba blocks and %zu K/V blocks differ; hlast %s\n",
                         best->tokens.size(), bad_s, bad_k, back.hlast == best->hlast ? "same" : "DIFFERS");
        }
        if (o.verbose)
            std::fprintf(stderr, "strata-nemotron: conversation cache: restored %zu tokens in %.0f ms\n", best->tokens.size(), ms_since(t0));
    }
    best->used = ++use_clock;
    return true;
}

// ---- session files: "SNSS", u32 version 1, the geometry (n_layers, n_embd, head_dim, n_head_kv, ctx), the first shard's size, then
// u64 L, L tokens, the Mamba states of each Mamba block, hlast, the K / V of each attention block and of the MTP block ([Hk][L][D])
namespace {
struct SessionHead {
    char magic[4];
    uint32_t version, n_layers, n_embd, head_dim, n_head_kv, ctx;
    uint64_t model_bytes, L;
};
}  // namespace

std::string EngineImpl::save_session(const std::string& path, size_t& bytes) {
    if (live.empty()) return "invalid 0 there is no conversation to save";
    if (!kv_holds(live)) return "invalid 0 the held conversation's cells were written over";
    Checkpoint c;
    capture_state(c);
    capture_kv(c);
    SessionHead h{{'S', 'N', 'S', 'S'}, 1, (uint32_t) g.n_layers, (uint32_t) g.n_embd, (uint32_t) g.head_dim, (uint32_t) g.n_head_kv,
                  (uint32_t) ctx, (uint64_t) model_bytes, (uint64_t) c.tokens.size()};
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return "io 0 cannot write " + tmp;
    bool ok = std::fwrite(&h, sizeof h, 1, f) == 1 && std::fwrite(c.tokens.data(), 4, c.tokens.size(), f) == c.tokens.size();
    for (int l = 0; ok && l < g.n_layers; ++l)
        if (!c.conv[(size_t) l].empty())
            ok = std::fwrite(c.conv[(size_t) l].data(), 4, c.conv[(size_t) l].size(), f) == c.conv[(size_t) l].size() &&
                 std::fwrite(c.ssm[(size_t) l].data(), 4, c.ssm[(size_t) l].size(), f) == c.ssm[(size_t) l].size();
    ok = ok && std::fwrite(c.hlast.data(), 4, c.hlast.size(), f) == c.hlast.size();
    for (size_t i = 0; ok && i < c.k.size(); ++i)
        if (!c.k[i].empty())
            ok = std::fwrite(c.k[i].data(), 2, c.k[i].size(), f) == c.k[i].size() && std::fwrite(c.v[i].data(), 2, c.v[i].size(), f) == c.v[i].size();
    ok = std::fclose(f) == 0 && ok;
    std::error_code ec;
    if (!ok) {
        std::filesystem::remove(tmp, ec);
        return "storage 0 the session file could not be written (disk full?)";
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return "io 0 cannot replace " + path;
    }
    bytes = sizeof h + c.bytes();
    return "";
}

std::string EngineImpl::restore_session(const std::string& path, size_t& bytes) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "invalid 0 cannot open " + path;
    SessionHead h{};
    Checkpoint c;
    std::string why;
    bool ok = std::fread(&h, sizeof h, 1, f) == 1;
    if (!ok || std::memcmp(h.magic, "SNSS", 4) != 0 || h.version != 1) why = "invalid 0 not a session file of this engine";
    else if (h.n_layers != (uint32_t) g.n_layers || h.n_embd != (uint32_t) g.n_embd || h.head_dim != (uint32_t) g.head_dim ||
             h.n_head_kv != (uint32_t) g.n_head_kv || h.model_bytes != (uint64_t) model_bytes)
        why = "invalid 0 the session file belongs to another model";
    else if (h.ctx != (uint32_t) ctx) why = "invalid 0 the session file was saved with another --max-context";
    else if (h.L == 0 || h.L >= (uint64_t) ctx) why = "invalid 0 the session file's length is not one this context holds";
    if (why.empty()) {
        const size_t L = (size_t) h.L, D = (size_t) g.head_dim, Hk = (size_t) g.n_head_kv;
        c.tokens.resize(L);
        ok = std::fread(c.tokens.data(), 4, L, f) == L;
        c.conv.assign((size_t) g.n_layers, {});
        c.ssm.assign((size_t) g.n_layers, {});
        for (int l = 0; ok && l < g.n_layers; ++l)
            if (layers[(size_t) l].kind == BlockKind::Mamba) {
                c.conv[(size_t) l].resize((size_t) (g.ssm_d_conv - 1) * g.ssm_conv_dim());
                c.ssm[(size_t) l].resize((size_t) g.ssm_d_inner * g.ssm_d_state);
                ok = std::fread(c.conv[(size_t) l].data(), 4, c.conv[(size_t) l].size(), f) == c.conv[(size_t) l].size() &&
                     std::fread(c.ssm[(size_t) l].data(), 4, c.ssm[(size_t) l].size(), f) == c.ssm[(size_t) l].size();
            }
        c.hlast.resize((size_t) g.n_embd);
        ok = ok && std::fread(c.hlast.data(), 4, c.hlast.size(), f) == c.hlast.size();
        c.k.assign((size_t) g.n_layers + 1, {});
        c.v.assign((size_t) g.n_layers + 1, {});
        for (int l = 0; ok && l <= g.n_layers; ++l) {
            const bool attn = l < g.n_layers ? layers[(size_t) l].kind == BlockKind::Attention : (has_mtp && last().mtp_kc != nullptr);
            if (!attn) continue;
            c.k[(size_t) l].resize(Hk * L * D);
            c.v[(size_t) l].resize(Hk * L * D);
            ok = std::fread(c.k[(size_t) l].data(), 2, c.k[(size_t) l].size(), f) == c.k[(size_t) l].size() &&
                 std::fread(c.v[(size_t) l].data(), 2, c.v[(size_t) l].size(), f) == c.v[(size_t) l].size();
        }
        if (!ok) why = "invalid 0 the session file is cut short";
    }
    std::fclose(f);
    if (!why.empty()) return why;
    restore_state(c);
    restore_kv(c);
    c.parked = true;   // (kept as a parked conversation: the next request continues it, and another one can come back to it)
    c.used = ++use_clock;
    bytes = sizeof h + c.bytes();
    ckpts.push_back(std::move(c));
    trim_checkpoints();
    return "";
}

int64_t EngineImpl::feed(const std::vector<int32_t>& ids, std::vector<float>* last_logits,
                         const std::function<bool(int64_t, int64_t)>& progress) {
    // the held state continues when it is a strict prefix of the prompt (the last prompt token must be read for its logits)
    size_t m = 0;
    while (m < live.size() && m < ids.size() && live[m] == ids[m]) ++m;
    if (!(m == live.size() && m < ids.size())) {
        park_live();   // (another conversation: the held one goes to RAM first, when it is long enough to be worth it)
        if (!restore_checkpoint(ids, ids.size() - 1)) reset_state();
    }
    const int64_t from = pos, total = (int64_t) ids.size() - from;
    // checkpoints: in front of the last <|im_start|> (the turn the next request re-renders differently: its thinking is dropped), at
    // the end of a long system prompt (the second <|im_start|>: kept for good, a new chat with the same start begins there), and every
    // o.ckpt_every tokens of a long prompt
    std::vector<int64_t> marks;
    int64_t root = -1;
    if (im_start >= 0) {
        for (int64_t i = (int64_t) ids.size() - 1; i > from; --i)
            if (ids[(size_t) i] == im_start) {
                marks.push_back(i);
                break;
            }
        int seen = 0;
        for (int64_t i = 0; i < (int64_t) ids.size() - 1 && seen < 2; ++i)
            if (ids[(size_t) i] == im_start && ++seen == 2 && i >= o.root_min && i > from) {
                root = i;
                marks.push_back(i);
            }
    }
    if (o.ckpt_every > 0)
        for (int64_t i = (from / o.ckpt_every + 1) * o.ckpt_every; i < (int64_t) ids.size() - 1; i += o.ckpt_every) marks.push_back(i);
    std::sort(marks.begin(), marks.end());
    // the loan for the longest chunk this prompt reads (all of it up to a full chunk): a short prompt lends no rows, a 300-token one few
    for (Stage& s : stages) lend(s, (int) std::min<int64_t>(tmax, (int64_t) ids.size() - pos));
    int64_t done = 0;
    // the chunks: at most tmax rows, cut at the marks
    std::vector<std::array<int64_t, 2>> plan;
    for (int64_t p = pos; p < (int64_t) ids.size();) {
        int64_t end = std::min<int64_t>((int64_t) ids.size(), p + tmax);
        for (int64_t mk : marks)
            if (mk > p && mk < end) {
                end = mk;
                break;
            }
        plan.push_back({p, end});
        p = end;
    }
    auto is_mark = [&](int64_t p) { return std::find(marks.begin(), marks.end(), p) != marks.end(); };
    const bool piped = pipe_on();
    size_t c = 0;
    while (c < plan.size()) {
        // the prefill pipeline: a run of consecutive chunks over kDecMax rows (the prompt path), up to the next checkpoint
        size_t e = c;
        if (piped)
            while (e < plan.size() && plan[e][1] - plan[e][0] > kDecMax) {
                ++e;
                if (is_mark(plan[e - 1][1])) break;
            }
        if (e - c >= 2) {
            // the run's chunks but its last are taken care of in it (the held state, progress); a cancel ends the read there
            if (!feed_piped(ids, plan, c, e, progress, done, total)) {
                for (Stage& s : stages) give_back(s);
                return done;
            }
            c = e - 1;   // (its last chunk is read; the bookkeeping below)
        } else {
            forward(ids.data() + plan[c][0], (int) (plan[c][1] - plan[c][0]), plan[c][0], false, plan[c][1] == (int64_t) ids.size() ? 1 : 0);
        }
        const int64_t end = plan[c][1];
        const int T = (int) (end - pos);
        live.insert(live.end(), ids.begin() + pos, ids.begin() + end);
        pos = end;
        done += T;
        ++c;
        if (is_mark(pos)) save_checkpoint(pos == root);
        if (progress && !progress(done, total)) {
            for (Stage& s : stages) give_back(s);
            return done;
        }
    }
    for (Stage& s : stages) give_back(s);
    Stage& ls = last();
    use(ls);
    if (last_logits) {
        last_logits->resize((size_t) g.n_vocab);
        CK(cudaMemcpyAsync(last_logits->data(), ls.logits, (size_t) g.n_vocab * 4, cudaMemcpyDeviceToHost, ls.st));
    }
    CK(cudaStreamSynchronize(ls.st));
    save_checkpoint();
    return done;
}

/// The prefill pipeline: the chunks plan[c0, c1) (each over kDecMax rows) through the stages, each stage on a thread of its own (the
/// last on this one) a chunk behind the one before it, so the GPUs read different chunks at the same time.  A stage hands a chunk's
/// rows to the next through host memory (hand_x: one chunk at a time).  Every stage computes its chunks in order and as forward()
/// does, so the result is forward()'s, bit for bit.  The held state (live, pos, done, progress) follows the chunks the last stage
/// finished, but the run's last chunk, which the caller books.  False: progress() cancelled; the state is then what the last stage
/// read (or its last checkpoint, when a stage before it had read further: that stage's Mamba states are past it).
bool EngineImpl::feed_piped(const std::vector<int32_t>& ids, const std::vector<std::array<int64_t, 2>>& plan, size_t c0, size_t c1,
                            const std::function<bool(int64_t, int64_t)>& progress, int64_t& done, int64_t total) {
    const int S = (int) stages.size(), n = (int) (c1 - c0), N = (int) g.n_embd;
    const int64_t a = plan[c0][0], b = plan[c1 - 1][1];
    // forward()'s set-up, once for the run: the cells' tokens, the loan for its longest chunk
    if ((int64_t) kv_tokens.size() < a) kv_tokens.resize((size_t) a, kNoToken);
    kv_tokens.resize((size_t) a);
    kv_tokens.insert(kv_tokens.end(), ids.begin() + a, ids.begin() + b);
    int t_max = 0;
    for (size_t i = c0; i < c1; ++i) t_max = std::max(t_max, (int) (plan[i][1] - plan[i][0]));
    for (Stage& s : stages) {
        if (t_max > s.t_small) lend(s, t_max);
        else if (!s.lent) land_restore(s, false);
    }
    if (prof_on && prof_st.size() < stages.size()) prof_st.resize(stages.size());   // (prof() would resize it from two threads)
    struct Hand {
        std::mutex mu;
        std::condition_variable cv;
        int full = -1;   // the chunk whose rows are in hand_x, -1: free
    };
    std::vector<std::unique_ptr<Hand>> hands;
    for (int i = 0; i + 1 < S; ++i) hands.push_back(std::make_unique<Hand>());
    std::atomic<bool> quit{false};
    bool cancelled = false;
    std::mutex err_mu;
    std::exception_ptr err;
    std::vector<int> reached((size_t) S, 0);   // the chunks each stage has read (its states are past them)
    auto stop_all = [&] {
        quit.store(true);
        for (auto& h : hands) {
            { std::lock_guard<std::mutex> lk(h->mu); }
            h->cv.notify_all();
        }
    };
    auto tokens_into = [&](int32_t* dst, const int32_t* toks, int T) {
        for (int t = 0; t < T; ++t) dst[t] = toks[t] >= 0 ? toks[t] : std::max(0, img_pad);
    };
    auto work = [&](int si) {
        try {
            Stage& s = stages[(size_t) si];
            use(s);
            HostSet& hb = hs(si);
            const size_t row_bytes_x = (size_t) N * 4;
            for (int i = 0; i < n; ++i) {
                if (quit.load()) return;
                const int64_t p0 = plan[c0 + (size_t) i][0];
                const int T = (int) (plan[c0 + (size_t) i][1] - p0);
                const bool last_chunk = plan[c0 + (size_t) i][1] == (int64_t) ids.size();
                if (si == 0) {
                    tokens_into(hb.tok, ids.data() + p0, T);
                    CK(cudaMemcpyAsync(s.tok, hb.tok, (size_t) T * 4, cudaMemcpyHostToDevice, s.st));
                    K::iq_embed_rows(embd_type, s.embd, embd_row, s.tok, T, N, s.x, s.st);
                    put_rows(s.x, T, p0, s.st);
                } else {
                    Hand& h = *hands[(size_t) si - 1];
                    {
                        std::unique_lock<std::mutex> lk(h.mu);
                        h.cv.wait(lk, [&] { return h.full == i || quit.load(); });
                        if (quit.load()) return;
                    }
                    CK(cudaMemcpyAsync(s.x, hand_x[(size_t) si - 1], (size_t) T * row_bytes_x, cudaMemcpyHostToDevice, s.st));
                    CK(cudaStreamSynchronize(s.st));
                    {
                        std::lock_guard<std::mutex> lk(h.mu);
                        h.full = -1;
                    }
                    h.cv.notify_all();
                    if (si + 1 == S && has_mtp) {   // the last card's MTP fill reads the tokens
                        tokens_into(hb.tok, ids.data() + p0, T);
                        CK(cudaMemcpyAsync(s.tok, hb.tok, (size_t) T * 4, cudaMemcpyHostToDevice, s.st));
                    }
                }
                run_blocks(s, T, p0, false);
                if (si + 1 < S) {
                    CK(cudaStreamSynchronize(s.st));
                    reached[(size_t) si] = i + 1;
                    Hand& h = *hands[(size_t) si];
                    {
                        std::unique_lock<std::mutex> lk(h.mu);
                        h.cv.wait(lk, [&] { return h.full == -1 || quit.load(); });
                        if (quit.load()) return;
                    }
                    CK(cudaMemcpyAsync(hand_x[(size_t) si], s.x, (size_t) T * row_bytes_x, cudaMemcpyDeviceToHost, s.st));
                    CK(cudaStreamSynchronize(s.st));
                    {
                        std::lock_guard<std::mutex> lk(h.mu);
                        h.full = i;
                    }
                    h.cv.notify_all();
                } else {
                    forward_tail(T, p0, last_chunk ? 1 : 0);
                    CK(cudaStreamSynchronize(s.st));
                    reached[(size_t) si] = i + 1;
                    if (i + 1 < n) {   // (this is the caller's thread: the held state and progress are its)
                        live.insert(live.end(), ids.begin() + pos, ids.begin() + (p0 + T));
                        pos = p0 + T;
                        done += T;
                        if (progress && !progress(done, total)) {
                            cancelled = true;
                            stop_all();
                            return;
                        }
                    }
                }
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lk(err_mu);
                if (!err) err = std::current_exception();
            }
            stop_all();
        }
    };
    std::vector<std::thread> threads;
    for (int si = 0; si + 1 < S; ++si) threads.emplace_back(work, si);
    work(S - 1);
    for (std::thread& t : threads) t.join();
    if (err || cancelled) {
        // the cells past what the last stage read do not hold their tokens on every layer
        if ((int64_t) kv_tokens.size() > pos) kv_tokens.resize((size_t) pos);
        bool ahead = false;
        for (int si = 0; si + 1 < S; ++si) ahead = ahead || reached[(size_t) si] > reached[(size_t) S - 1];
        if (ahead) {   // back to the last checkpoint of what was read (else from the start)
            try {
                const std::vector<int32_t> held = live;
                if (!restore_checkpoint(held, held.size())) reset_state();
            } catch (...) {
                if (!err) throw;
            }
        }
        if (err) std::rethrow_exception(err);
        return false;
    }
    use(last());
    return true;
}

// ================================================================================================================ decoding
/// Read `tok` at pos (one row); returns the greedy pick of its logits (`lg`: the logits row).
int32_t EngineImpl::decode_step(int32_t tok, std::vector<float>* lg, int topk) {
    struct Flush { EngineImpl* e; ~Flush() { e->prof_flush(); } } flush_at_end{this};
    forward(&tok, 1, pos, false, 1);
    live.push_back(tok);
    ++pos;
    Stage& ls = last();
    use(ls);
    CK(cudaMemcpyAsync(h_picks, ls.picks, 4, cudaMemcpyDeviceToHost, ls.st));
    if (topk > 0) {   // a sampled step: its top-k only (h_topi / h_topv)
        G::topk_rows(ls.logits, (int) g.n_vocab, 1, topk, d_topi, d_topv, ls.st);
        CK(cudaMemcpyAsync(h_topi, d_topi, (size_t) topk * 4, cudaMemcpyDeviceToHost, ls.st));
        CK(cudaMemcpyAsync(h_topv, d_topv, (size_t) topk * 4, cudaMemcpyDeviceToHost, ls.st));
    } else if (lg) {
        lg->resize((size_t) g.n_vocab);
        CK(cudaMemcpyAsync(lg->data(), ls.logits, (size_t) g.n_vocab * 4, cudaMemcpyDeviceToHost, ls.st));
    }
    CK(cudaStreamSynchronize(ls.st));
    return h_picks[0];
}

/// One speculative step: x (decided, at pos) and D MTP drafts through one verify window.  `accepted` gets x and the drafts the main
/// model agreed with (they are read: pos advances by their count); `next` the token after them.  Returns the accepted drafts.
int EngineImpl::spec_step(int32_t x, int D, std::vector<int32_t>& accepted, int32_t& next) {
    Stage& ls = last();
    use(ls);
    prof(ls, "start");
    struct Flush { EngineImpl* e; ~Flush() { e->prof_flush(); } } flush_at_end{this};
    const int N = (int) g.n_embd;
    std::vector<int32_t> win = {x};
    // drafts: cell pos - 1 = (hlast, x); then each draft's own seed (the block's output) with the draft before it
    float* seed = ls.hlast;
    float* bufs[2] = {ls.hprev, ls.hprev + N};   // (hprev is free between forwards)
    int64_t cell = pos - 1;
    if (cell < 0) D = 0;
    const std::vector<int32_t> lk = lookup_window(x, D);
    if (!lk.empty()) {
        win.insert(win.end(), lk.begin(), lk.end());
        D = 0;   // (no MTP drafts)
    }
    const auto td0 = Clock::now();
    for (int d = 0; d < D; ++d) {
        float* out = bufs[d % 2];
        const int32_t dr = mtp_draft(win.back(), cell + d, seed, out);
        win.push_back(dr);
        seed = out;
    }
    draft_ms += ms_since(td0);
    draft_n += D;
    const auto tv0 = Clock::now();
    forward(win.data(), (int) win.size(), pos, true, (int) win.size());
    use(ls);
    CK(cudaMemcpyAsync(h_picks, ls.picks, win.size() * 4, cudaMemcpyDeviceToHost, ls.st));
    CK(cudaStreamSynchronize(ls.st));
    verify_ms += ms_since(tv0);
    ++verify_n;
    int a = 0;
    while (a < (int) win.size() - 1 && h_picks[a] == win[(size_t) a + 1]) ++a;
    last_offered = (int) win.size() - 1;
    if (!lk.empty()) lookup_kept((int) lk.size(), a);
    commit(a + 1);
    use(ls);
    CK(cudaMemcpyAsync(ls.hlast, ls.hmain + (size_t) a * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, ls.st));
    accepted.assign(win.begin(), win.begin() + a + 1);
    live.insert(live.end(), accepted.begin(), accepted.end());
    pos += a + 1;
    next = h_picks[a];
    return a;
}

// ================================================================================================================ sampling
namespace {
void apply_penalties(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist) {
    if (s.repeat == 1.f && s.freq == 0.f && s.present == 0.f) return;
    std::map<int32_t, int> cnt;
    const size_t from = s.last_n > 0 && hist.size() > (size_t) s.last_n ? hist.size() - (size_t) s.last_n : 0;
    for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
    for (const auto& kv : cnt) {
        if (kv.first < 0 || (size_t) kv.first >= lg.size()) continue;
        float& v = lg[(size_t) kv.first];
        if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
        v -= s.freq * (float) kv.second + s.present;
    }
}

/// The distribution a sampled token is drawn from: penalties, then the top_k (64 at most) by logit, min_p and top_p, at the temperature;
/// (token, probability) pairs that sum to 1, the most likely first.  Greedy (temperature 0): the argmax with probability 1.
using Dist = std::vector<std::pair<int, double>>;
Dist dist_of(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist) {
    apply_penalties(lg, s, hist);
    if (s.temperature <= 0.f) return {{(int) (std::max_element(lg.begin(), lg.end()) - lg.begin()), 1.0}};
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; });
    idx.resize((size_t) kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) {
            keep = i;
            break;
        }
        cum += p[i] / sum;
        if (cum >= s.top_p) {
            keep = i + 1;
            break;
        }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    Dist d(keep);
    for (size_t i = 0; i < keep; ++i) d[i] = {idx[i], p[i] / tot};
    return d;
}

/// The k the GPU top-k path picks for this sampling, or 0 when it does not apply: greedy, any penalty (it changes logits after the
/// pick, so a token from outside the top k could come in), or a top_k above G::kTopKMax.  The distribution is then dist_of's.
int gpu_topk_k(const Sampling& s, int vocab) {
    if (s.temperature <= 0.f || s.repeat != 1.f || s.freq != 0.f || s.present != 0.f) return 0;
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, vocab));
    return kk <= G::kTopKMax ? kk : 0;
}

/// dist_of from the top kk logits already sorted (largest first): v the logits, id their tokens.
Dist dist_top(const float* v, const int32_t* id, int kk, const Sampling& s) {
    std::vector<double> p((size_t) kk);
    const double mx = v[0];
    double sum = 0;
    for (int i = 0; i < kk; ++i) sum += (p[(size_t) i] = std::exp((v[i] - mx) / s.temperature));
    size_t keep = (size_t) kk;
    double cum = 0;
    for (size_t i = 0; i < (size_t) kk; ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) {
            keep = i;
            break;
        }
        cum += p[i] / sum;
        if (cum >= s.top_p) {
            keep = i + 1;
            break;
        }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    Dist d(keep);
    for (size_t i = 0; i < keep; ++i) d[i] = {id[i], p[i] / tot};
    return d;
}

int draw(const Dist& d, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng);
    for (const auto& [t, p] : d)
        if ((r -= p) <= 0) return t;
    return d.back().first;
}

double prob_of(const Dist& d, int t) {
    for (const auto& [x, p] : d)
        if (x == t) return p;
    return 0.0;
}

int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    return draw(dist_of(lg, s, hist), rng);
}
std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s == ' ') ++s;
    while (*s) {
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
        if (*s == ',') ++s;
        else break;
    }
    return v;
}
}  // namespace

/// Prompt lookup: the tokens that followed the latest earlier occurrence of the context's last `min_match`..8 tokens (the longest match
/// wins), at most `max_n`.  `ctx` is the held tokens plus the decided one.  Empty when the context does not repeat itself.
std::vector<int32_t> lookup_drafts(const std::vector<int32_t>& ctx, int max_n, int min_match) {
    const int64_t n = (int64_t) ctx.size();
    if (n < min_match + 1 || max_n <= 0) return {};
    int best_len = 0;
    int64_t best_at = -1;
    const int64_t from = std::max<int64_t>(0, n - 131072);   // (the last 128K tokens are enough)
    for (int64_t i = n - 2; i >= from + min_match - 1; --i) {   // ctx[i] matches the last token, and the match ends at i
        if (ctx[(size_t) i] != ctx[(size_t) (n - 1)]) continue;
        int len = 1;
        while (len < 8 && i - len >= 0 && ctx[(size_t) (i - len)] == ctx[(size_t) (n - 1 - len)]) ++len;
        if (len >= min_match && len > best_len) {
            best_len = len;
            best_at = i;
            if (len == 8) break;
        }
    }
    if (best_at < 0) return {};
    std::vector<int32_t> out;
    for (int64_t j = best_at + 1; j < n - 1 && (int) out.size() < max_n; ++j)
        if (ctx[(size_t) j] >= 0) out.push_back(ctx[(size_t) j]);   // (a picture's cells are not tokens)
        else break;
    return out;
}

/// spec_step for a sampled request (speculative sampling, Leviathan et al. / Chen et al.): each draft is drawn from the MTP block's
/// distribution q under the request's sampling, the window gives the main model's p for every row, draft d is kept with probability
/// min(1, p(d) / q(d)), and at the first rejection the next token is drawn from max(0, p - q) normalized (after every draft kept: from
/// the last row's p).  The tokens are distributed exactly as sampling the main model one token at a time.  `hist`: the tokens so far
/// (the penalties' window; x included).
int EngineImpl::spec_step_sampled(int32_t x, int D, const Sampling& smp, const std::vector<int32_t>& hist, std::mt19937_64& rng,
                                  std::vector<int32_t>& accepted, int32_t& next) {
    Stage& ls = last();
    use(ls);
    prof(ls, "start");
    struct Flush { EngineImpl* e; ~Flush() { e->prof_flush(); } } flush_at_end{this};
    const int N = (int) g.n_embd;
    const size_t V = (size_t) g.n_vocab;
    std::vector<int32_t> win = {x};
    float* seed = ls.hlast;
    float* bufs[2] = {ls.hprev, ls.hprev + N};
    int64_t cell = pos - 1;
    if (cell < 0) D = 0;
    std::vector<Dist> q;
    std::vector<int32_t> h = hist;
    std::vector<float> lg;
    const std::vector<int32_t> lk = lookup_window(x, D);
    if (!lk.empty()) {   // prompt-lookup drafts: q puts all its mass on the draft (kept with probability p(d))
        for (int32_t t : lk) {
            win.push_back(t);
            q.push_back(Dist{{t, 1.0}});
        }
        D = (int) lk.size();
    }
    static const bool gpu_sample_env = !(std::getenv("STRATA_NEMO_GPU_SAMPLE") && std::atoi(std::getenv("STRATA_NEMO_GPU_SAMPLE")) == 0);
    const int kk = o.gpu_sample && gpu_sample_env ? gpu_topk_k(smp, (int) V) : 0;   // > 0: the top kk of each row from the GPU
    const auto td0 = Clock::now();
    const int d_first = (int) q.size();
    for (int d = d_first; d < D; ++d) {
        float* out = bufs[d % 2];
        mtp_draft(win.back(), cell + d, seed, out, kk > 0 ? nullptr : &lg, kk);
        const auto ts0 = Clock::now();
        q.push_back(kk > 0 ? dist_top(h_topv, h_topi, kk, smp) : dist_of(lg, smp, h));
        const int32_t dr = draw(q.back(), rng);
        samp_ms += ms_since(ts0);
        win.push_back(dr);
        h.push_back(dr);
        seed = out;
    }
    draft_ms += ms_since(td0);
    draft_n += D - d_first;
    const int T = (int) win.size();
    const auto tv0 = Clock::now();
    forward(win.data(), T, pos, true, T);
    use(ls);
    std::vector<float> rows;
    if (kk > 0) {
        G::topk_rows(ls.logits, (int) V, T, kk, d_topi, d_topv, ls.st);
        CK(cudaMemcpyAsync(h_topi, d_topi, (size_t) T * kk * 4, cudaMemcpyDeviceToHost, ls.st));
        CK(cudaMemcpyAsync(h_topv, d_topv, (size_t) T * kk * 4, cudaMemcpyDeviceToHost, ls.st));
    } else {
        rows.resize(V * (size_t) T);
        CK(cudaMemcpyAsync(rows.data(), ls.logits, rows.size() * 4, cudaMemcpyDeviceToHost, ls.st));
    }
    CK(cudaStreamSynchronize(ls.st));
    verify_ms += ms_since(tv0);
    ++verify_n;
    const auto ts1 = Clock::now();
    // row r's distribution under the request's sampling (the penalties' window `hh`: the tokens before it)
    auto row_dist = [&](int r, std::vector<int32_t>& hh) -> Dist {
        if (kk > 0) return dist_top(h_topv + (size_t) r * kk, h_topi + (size_t) r * kk, kk, smp);
        std::vector<float> row(rows.begin() + (std::ptrdiff_t) ((size_t) r * V), rows.begin() + (std::ptrdiff_t) ((size_t) (r + 1) * V));
        return dist_of(row, smp, hh);
    };
    std::uniform_real_distribution<double> u(0.0, 1.0);
    h = hist;
    int a = 0;
    next = -1;
    for (; a < D; ++a) {
        const Dist p = row_dist(a, h);
        const int32_t d = win[(size_t) a + 1];
        const double pd = prob_of(p, d), qd = prob_of(q[(size_t) a], d);
        if (qd > 0 && u(rng) < std::min(1.0, pd / qd)) {
            h.push_back(d);
            continue;
        }
        // rejected: the residual max(0, p - q), normalized
        Dist r;
        double tot = 0;
        for (const auto& [t, pt] : p) {
            const double v = pt - prob_of(q[(size_t) a], t);
            if (v > 0) {
                r.emplace_back(t, v);
                tot += v;
            }
        }
        if (tot <= 0) next = draw(p, rng);
        else {
            for (auto& e : r) e.second /= tot;
            next = draw(r, rng);
        }
        break;
    }
    if (next < 0) next = draw(row_dist(D, h), rng);   // every draft kept: the token after them from the last row
    samp_ms += ms_since(ts1);
    last_offered = D;
    if (!lk.empty()) lookup_kept(D, a);
    commit(a + 1);
    use(ls);
    CK(cudaMemcpyAsync(ls.hlast, ls.hmain + (size_t) a * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, ls.st));
    accepted.assign(win.begin(), win.begin() + a + 1);
    live.insert(live.end(), accepted.begin(), accepted.end());
    pos += a + 1;
    return a;
}

/// --bench-cpu: how fast the CPU reads the experts in RAM, against how fast the expert kernel goes through them (decides whether the
/// CPU experts are bound by the RAM or by the kernel).  Random experts of every MoE layer, 23 a layer (a 7-row decode window's CPU
/// experts, measured 2026-11-13), 3 passes; the same pool threads for all.  Prints the numbers, changes nothing.
int EngineImpl::bench_cpu() {
    std::vector<Layer*> moe;
    for (Layer& y : layers)
        if (y.kind == BlockKind::Moe && y.host_up) moe.push_back(&y);
    if (moe.empty()) return 1;
    constexpr int kPerLayer = 23, kPasses = 3;
    std::mt19937 rng(7);
    struct Pick { Layer* L; int x; };
    std::vector<std::vector<Pick>> sets((size_t) kPasses);
    for (auto& set : sets)
        for (Layer* L : moe) {
            std::vector<int> ids((size_t) g.n_expert);
            std::iota(ids.begin(), ids.end(), 0);
            std::shuffle(ids.begin(), ids.end(), rng);
            for (int i = 0; i < kPerLayer; ++i) set.push_back({L, ids[(size_t) i]});
        }
    const size_t per_expert = moe[0]->fmt.up_bytes + moe[0]->fmt.down_bytes;
    std::fprintf(stderr, "strata-nemotron: bench: %d CPU threads; %s experts in RAM; %zu MoE layers x %d random experts x %d passes (%.2f MiB an expert)\n",
                 pool->threads(), pinned ? "page-locked" : "mapped-file", moe.size(), kPerLayer, kPasses, per_expert / 1048576.0);
    // 1. the same bytes read by the same threads (no arithmetic): the RAM's rate for this access pattern
    double read_ms = 0, read_b = 0;
    uint64_t chk = 0;
    for (int p = 0; p < kPasses; ++p) {
        std::vector<const uint8_t*> ptr;
        std::vector<size_t> nb;
        for (const Pick& k : sets[(size_t) (p + 1) % kPasses]) {   // (a pass the kernel below does not read right after)
            ptr.push_back(k.L->host_up + (size_t) k.x * k.L->fmt.up_bytes);
            nb.push_back(k.L->fmt.up_bytes);
            ptr.push_back(k.L->host_down + (size_t) k.x * k.L->fmt.down_bytes);
            nb.push_back(k.L->fmt.down_bytes);
            read_b += (double) (k.L->fmt.up_bytes + k.L->fmt.down_bytes);
        }
        const auto t0 = Clock::now();
        chk += pool->read_all(ptr.data(), nb.data(), (int) ptr.size());
        read_ms += ms_since(t0);
    }
    // 1b. the same, per kind of memory the layers are in
    for (int kind = 0; kind <= 2; ++kind) {
        std::vector<const uint8_t*> ptr;
        std::vector<size_t> nb;
        double b2 = 0;
        int nl = 0;
        for (const Pick& k : sets[2])
            if (k.L->pin_kind == kind) {
                ptr.push_back(k.L->host_up + (size_t) k.x * k.L->fmt.up_bytes);
                nb.push_back(k.L->fmt.up_bytes);
                ptr.push_back(k.L->host_down + (size_t) k.x * k.L->fmt.down_bytes);
                nb.push_back(k.L->fmt.down_bytes);
                b2 += (double) (k.L->fmt.up_bytes + k.L->fmt.down_bytes);
                ++nl;
            }
        if (ptr.empty()) continue;
        const auto t0 = Clock::now();
        chk += pool->read_all(ptr.data(), nb.data(), (int) ptr.size());
        std::fprintf(stderr, "strata-nemotron: bench: read only, random experts in %s (%d layers): %.1f GB/s\n",
                     kind == 0 ? "the mapped file" : kind == 1 ? "registered memory" : "cudaHostAlloc's memory", nl / kPerLayer, b2 / ms_since(t0) / 1e6);
    }
    // 2. the expert kernel as the decode calls it: one run() per layer with its 23 jobs (one row each)
    std::vector<float> lat((size_t) g.moe_latent * kDecMax);
    for (size_t i = 0; i < lat.size(); ++i) lat[i] = 0.01f * (float) ((int) (i % 97) - 48);
    std::vector<float> out((size_t) kPerLayer * g.moe_latent);
    double run_ms = 0, run_b = 0;
    long long run_n = 0;
    pool->time_phases(true);
    for (int p = 0; p < kPasses; ++p) {
        const auto& set = sets[(size_t) p];
        const auto t0 = Clock::now();
        for (size_t i = 0; i < set.size(); i += kPerLayer) {
            Layer* L = set[i].L;
            std::vector<CpuJob> jobs((size_t) kPerLayer);
            for (int j = 0; j < kPerLayer; ++j) {
                const int x = set[i + (size_t) j].x;
                jobs[(size_t) j] = CpuJob{L->host_up + (size_t) x * L->fmt.up_bytes, L->host_down + (size_t) x * L->fmt.down_bytes, 0,
                                          out.data() + (size_t) j * g.moe_latent};
                run_b += (double) (L->fmt.up_bytes + L->fmt.down_bytes);
            }
            pool->run(L->fmt, lat.data(), 1, jobs.data(), kPerLayer);
            run_n += kPerLayer;
        }
        run_ms += ms_since(t0);
    }
    std::fprintf(stderr, "strata-nemotron: bench: the expert kernel's phases, by weight type:%s\n", pool->phase_report().c_str());
    pool->time_phases(false);
    // 3. one expert per call: the fixed cost of a call (waking the workers, the three phases)
    double one_ms = 0;
    int one_n = 0;
    for (const Pick& k : sets[0]) {
        if (one_n >= 400) break;
        CpuJob j{k.L->host_up + (size_t) k.x * k.L->fmt.up_bytes, k.L->host_down + (size_t) k.x * k.L->fmt.down_bytes, 0, out.data()};
        const auto t0 = Clock::now();
        pool->run(k.L->fmt, lat.data(), 1, &j, 1);
        one_ms += ms_since(t0);
        ++one_n;
    }
    // 4. long sequential reads (whole layers of the arena): the RAM's best rate
    double seq_ms = 0, seq_b = 0;
    {
        std::vector<const uint8_t*> ptr;
        std::vector<size_t> nb;
        for (size_t i = 0; i < moe.size() && seq_b < 8e9; ++i) {
            ptr.push_back(moe[i]->host_up);
            nb.push_back(moe[i]->fmt.up_bytes * (size_t) g.n_expert);
            ptr.push_back(moe[i]->host_down);
            nb.push_back(moe[i]->fmt.down_bytes * (size_t) g.n_expert);
            seq_b += (double) (nb[nb.size() - 2] + nb.back());
        }
        const auto t0 = Clock::now();
        chk += pool->read_all(ptr.data(), nb.data(), (int) ptr.size());
        seq_ms = ms_since(t0);
    }
    std::fprintf(stderr, "strata-nemotron: bench: read only, the same random experts: %.1f GB/s (%.3f ms an expert)\n", read_b / read_ms / 1e6,
                 read_ms / (double) (kPasses * kPerLayer * moe.size()));
    std::fprintf(stderr, "strata-nemotron: bench: read only, %.1f GB in whole layers: %.1f GB/s\n", seq_b / 1e9, seq_b / seq_ms / 1e6);
    std::fprintf(stderr, "strata-nemotron: bench: the expert kernel, %d experts a call (as a decode layer): %.3f ms an expert = %.1f GB/s of weights\n",
                 kPerLayer, run_ms / (double) run_n, run_b / run_ms / 1e6);
    std::fprintf(stderr, "strata-nemotron: bench: the expert kernel, 1 expert a call: %.3f ms a call\n", one_ms / std::max(1, one_n));
    // 5. the same sequential read of ordinary (pageable) memory: whether the page-locked arena itself reads slower
    {
        const size_t nb = (size_t) 4 << 30;
        std::vector<uint8_t> plain(nb);
        std::memset(plain.data(), 1, nb);   // (every page committed before the timing)
        const uint8_t* p = plain.data();
        const auto t0 = Clock::now();
        chk += pool->read_all(&p, &nb, 1);
        const double ms = ms_since(t0);
        std::fprintf(stderr, "strata-nemotron: bench: read only, 4.3 GB of ordinary (pageable) memory: %.1f GB/s\n", (double) nb / ms / 1e6);
        // 6. the same memory page-locked afterwards (cudaHostRegister): the CPU's rate, and the copy rate to the last GPU
        use(last());
        if (cudaHostRegister(plain.data(), nb, cudaHostRegisterPortable) == cudaSuccess) {
            const auto t1 = Clock::now();
            chk += pool->read_all(&p, &nb, 1);
            const double ms1 = ms_since(t1);
            uint8_t* d = dalloc<uint8_t>((size_t) 64 << 20);
            double gbps = 0;
            for (int rep = 0; rep < 6; ++rep) {   // (not fatal: with the arena locked too, this can meet the locked-memory limit)
                const auto t2 = Clock::now();
                if (cudaMemcpyAsync(d, plain.data() + ((size_t) rep << 26), (size_t) 64 << 20, cudaMemcpyHostToDevice, last().st) != cudaSuccess ||
                    cudaStreamSynchronize(last().st) != cudaSuccess) {
                    cudaGetLastError();
                    break;
                }
                gbps = std::max(gbps, (double) (64 << 20) / (ms_since(t2) / 1000.0) / 1e9);
            }
            CK(cudaFree(d));
            CK(cudaHostUnregister(plain.data()));
            std::fprintf(stderr, "strata-nemotron: bench: ... the same memory page-locked by cudaHostRegister: the CPU reads %.1f GB/s; "
                         "to GPU %d %.1f GB/s (the arena: %.1f)\n", (double) nb / ms1 / 1e6, last().dev, gbps, last().h2d_gbps);
        } else {
            cudaGetLastError();
            std::fprintf(stderr, "strata-nemotron: bench: cudaHostRegister of 4.3 GB failed\n");
        }
    }
    std::fprintf(stderr, "strata-nemotron: bench: (checksum %llu)\n", (unsigned long long) (chk & 0xffff));
    return 0;
}

/// One request's decode in detail (stderr): where each GPU's MoE layers waited, the MTP drafts and their experts, the verify windows,
/// the host's sampling; and how concentrated the MTP block's routing is (what --mtp-slots N would hold).
void EngineImpl::print_detail(double decode_ms) {
    if (!o.verbose || dstat.empty()) return;
    std::string line = "strata-nemotron: decode detail:";
    char b[512];
    for (size_t i = 0; i < stages.size(); ++i) {
        const DecStat& d = dstat[i];
        const long long all_n = d.gpu + d.cpu_n;
        std::snprintf(b, sizeof b, " GPU %d's MoE layers: waits %.0f ms before + %.0f after, CPU experts %.0f ms; experts %lld on the GPU (%lld in a slot = %.1f%%, %lld over the link), %lld on the CPU; PCIe share x%.2f (%s, %lld down / %lld up);",
                      stages[i].dev, d.route, d.tail, d.cpu, d.gpu, d.res, all_n ? 100.0 * (double) d.res / (double) all_n : 0.0, d.link, d.cpu_n, stages[i].share_k,
                      share_adapting(stages[i]) ? "adapting" : "fixed", stages[i].share_down, stages[i].share_up);
        line += b;
    }
    std::fprintf(stderr, "%s\n", line.c_str());
    const DecStat& m = dstat.back();
    std::fprintf(stderr, "strata-nemotron: decode detail: MTP drafts %lld in %.0f ms (%.2f ms each; their experts: waits %.0f + %.0f ms, CPU %.0f ms, "
                 "%lld on the GPU (%lld over the link), %lld on the CPU); verify windows %lld in %.0f ms (%.1f ms each); host sampling %.0f ms; "
                 "the rest %.0f ms\n",
                 draft_n, draft_ms, draft_n ? draft_ms / (double) draft_n : 0.0, m.route, m.tail, m.cpu, m.gpu, m.link, m.cpu_n, verify_n, verify_ms,
                 verify_n ? verify_ms / (double) verify_n : 0.0, samp_ms, decode_ms - draft_ms - verify_ms - samp_ms);
    if (prof_on && !prof_ms.empty()) {   // STRATA_NEMO_PROFILE=1: the GPU's time per operation, per GPU, and per verify window
        std::vector<std::pair<std::string, std::vector<double>>> rows = prof_ms;
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
            double sa = 0, sb = 0;
            for (double v : a.second) sa += v;
            for (double v : b.second) sb += v;
            return sa > sb;
        });
        std::string pl = "strata-nemotron: decode profile (GPU time, ms in all / per verify window, GPU 0 | GPU 1):";
        for (const auto& [name, v] : rows) {
            std::snprintf(b, sizeof b, "\n    %-40s", name.c_str());
            pl += b;
            for (double ms : v) {
                std::snprintf(b, sizeof b, " %9.0f / %6.2f |", ms, verify_n ? ms / (double) verify_n : 0.0);
                pl += b;
            }
        }
        std::fprintf(stderr, "%s\n", pl.c_str());
    }
    const int ml = mtp_layer();
    if (ml < 0) return;
    std::vector<double> t = layers[(size_t) ml].total;
    std::sort(t.begin(), t.end(), std::greater<double>());
    double all = 0;
    for (double v : t) all += v;
    if (all <= 0) return;
    auto share = [&](int n) {
        double s = 0;
        for (int i = 0; i < n && i < (int) t.size(); ++i) s += t[(size_t) i];
        return 100.0 * s / all;
    };
    std::fprintf(stderr, "strata-nemotron: the MTP block's routing since the start (%.0f picks): its 32 / 64 / 128 / 256 most-used experts take "
                 "%.0f / %.0f / %.0f / %.0f%%; %d of its experts in VRAM\n",
                 all, share(32), share(64), share(128), share(256), layers[(size_t) ml].n_slots);
}

// ================================================================================================================ serve
int EngineImpl::serve() {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::printf("INFO engine=nemotron-0.1.0 family=nemotron_h_moe gpus=%zu mtp=%d\n", stages.size(), o.mtp);
    auto last_save = Clock::now();
    std::printf("READY %d stop\n", ctx);
    std::fflush(stdout);
    std::thread reader([&] {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                stop.store(true);
                continue;
            }
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(l);
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    });
    reader.detach();
    for (;;) {
        std::string line;
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return eof || !lines.empty(); });
            if (lines.empty()) {
                if (!o.profile_save.empty()) save_profile(o.profile_save);   // (the end of the input: the server is stopping)
                return 0;
            }
            line = lines.front();
            lines.pop_front();
        }
        if (line.empty()) continue;
        if (line.rfind("SAVE ", 0) == 0 || line.rfind("RESTORE ", 0) == 0) {   // session files (serve/server.py _session_file)
            const bool save = line[0] == 'S';
            const std::string path = line.substr(save ? 5 : 8);
            const auto t0 = Clock::now();
            size_t bytes = 0;
            const std::string why = save ? save_session(path, bytes) : restore_session(path, bytes);
            if (!why.empty()) std::printf("SERR %s\n", why.c_str());
            else std::printf("%s %zu %zu %.1f\n", save ? "SAVED" : "RESTORED", live.size(), bytes, ms_since(t0));
            std::fflush(stdout);
            continue;
        }
        if (line == "QUIT") {   // the server's stop: what was routed is the next start's profile
            if (!o.profile_save.empty()) save_profile(o.profile_save);
            return 0;
        }
        const bool geni = line.rfind("GENI ", 0) == 0;
        if (!geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or GENI <max_new> [key=value ...] <file> <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        const char* p = line.c_str() + (geni ? 5 : 4);
        char* pe = nullptr;
        std::vector<uint64_t> img_keys;   // img_keys=<hex>,...: each picture's key from the server (its normalized bytes' hash)
        const long long max_new = std::strtoll(p, &pe, 10);
        p = pe;
        Sampling smp;
        bool unsupported = false;
        std::string why;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) {
                p = const_cast<char*>(st0);
                break;
            }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
            else if (key == "img_keys") {
                img_keys.clear();
                const char* q = v;
                while (*q != '\0') {
                    char* qe = nullptr;
                    const unsigned long long k = std::strtoull(q, &qe, 16);
                    if (qe == q || (*qe != ',' && *qe != '\0')) {   // malformed: the rows' hash instead
                        img_keys.clear();
                        break;
                    }
                    img_keys.push_back((uint64_t) k);
                    q = *qe == ',' ? qe + 1 : qe;
                }
            } else if (key == "logprobs" || key == "bias" || key == "logit_bias") {
                unsupported = true;
                why = key + " is not available on this engine path yet";
            }
            // other keys (ckpt, pin, cvec, ...) mean nothing here
        }
        std::string emb_path;
        if (geni) {   // the embeddings file: the first token without '=' (the server writes paths without spaces)
            while (*p == ' ') ++p;
            const char* gap = std::strchr(p, ' ');
            if (gap != nullptr) {
                emb_path.assign(p, (size_t) (gap - p));
                p = gap;
            }
        }
        std::vector<int32_t> ids = parse_ids(p);
        if (unsupported) {
            std::printf("ERR %s\n", why.c_str());
            std::fflush(stdout);
            continue;
        }
        if (max_new < 1 || ids.empty() || (geni && emb_path.empty())) {
            std::printf("ERR bad request: max_new, the embeddings file or the ids\n");
            std::fflush(stdout);
            continue;
        }
        if ((int64_t) ids.size() + max_new > ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, ctx);
            std::fflush(stdout);
            continue;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t < 0 || t >= (int32_t) g.n_vocab;
        if (bad) {
            std::printf("ERR a token id is outside the vocabulary\n");
            std::fflush(stdout);
            continue;
        }
        // ---- GENI: the pictures' rows (strata-vision records: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats, in prompt
        // order) take the place of their runs of <image> tokens; each cell's id in the held state becomes its picture's identity
        std::vector<float> img_rows;
        std::vector<const float*> row_ptr;
        if (geni) {
            std::string ve;
            struct Img { int64_t n, nx, ny; size_t off; };
            std::vector<Img> imgs;
            if (img_pad < 0) ve = "this model's vocabulary has no <image> token";
            std::FILE* f = ve.empty() ? std::fopen(emb_path.c_str(), "rb") : nullptr;
            if (ve.empty() && !f) ve = "cannot open " + emb_path;
            while (f && ve.empty()) {
                int32_t hdr[5];
                const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
                if (got == 0) break;
                if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 || (int64_t) hdr[2] * hdr[3] != hdr[1] ||
                    hdr[4] != (int32_t) g.n_embd) {
                    ve = "bad embeddings file (expected strata-vision records of width " + std::to_string((long long) g.n_embd) + ")";
                    break;
                }
                const size_t off = img_rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
                img_rows.resize(off + cnt);
                if (std::fread(img_rows.data() + off, sizeof(float), cnt, f) != cnt) {
                    ve = "short embeddings file";
                    break;
                }
                imgs.push_back({hdr[1], hdr[2], hdr[3], off});
            }
            if (f) std::fclose(f);
            row_ptr.assign(ids.size(), nullptr);
            const int64_t n = (int64_t) ids.size();
            size_t k = 0;
            for (int64_t i = 0; ve.empty() && i < n;) {
                if (ids[(size_t) i] != img_pad) {
                    ++i;
                    continue;
                }
                if (k >= imgs.size()) {
                    ve = "the prompt has more images than the embeddings file";
                    break;
                }
                const Img& im = imgs[k++];
                for (int64_t j = 0; j < im.n && ve.empty(); ++j)
                    if (i + j >= n || ids[(size_t) (i + j)] != img_pad)
                        ve = "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) + " rows but fewer <image> tokens";
                if (!ve.empty()) break;
                // what a held prefix is compared by: the picture's grid and the server's key (or its rows)
                auto fnv = [](const void* d, size_t b, uint64_t h) {
                    const auto* c = (const unsigned char*) d;
                    for (size_t q = 0; q < b; ++q) h = (h ^ c[q]) * 1099511628211ull;
                    return h;
                };
                const int64_t grid[3] = {im.n, im.nx, im.ny};
                uint64_t h = fnv(grid, sizeof grid, 1469598103934665603ull);
                h = img_keys.size() == imgs.size() ? fnv(&img_keys[k - 1], sizeof(uint64_t), h)
                                                   : fnv(img_rows.data() + im.off, (size_t) im.n * (size_t) g.n_embd * 4, h);
                for (int64_t j = 0; j < im.n; ++j) {
                    ids[(size_t) (i + j)] = image_cell_id(h, j);
                    row_ptr[(size_t) (i + j)] = img_rows.data() + im.off + (size_t) j * (size_t) g.n_embd;
                }
                i += im.n;
            }
            if (ve.empty() && k != imgs.size()) ve = "the embeddings file has more images than the prompt";
            if (ve.empty() && ids.back() < 0) ve = "the prompt cannot end in an image";
            if (!ve.empty()) {
                std::printf("ERR %s\n", ve.c_str());
                std::fflush(stdout);
                continue;
            }
        }
        rows = geni ? &row_ptr : nullptr;
        stop.store(false);
        const auto t0 = Clock::now();
        std::vector<float> lg;
        bool cancelled = false;
        const int64_t read = feed(ids, &lg, [&](int64_t done, int64_t total) {
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) total, ms_since(t0),
                        (double) done / std::max(1e-9, ms_since(t0) / 1000.0));
            std::fflush(stdout);
            if (stop.load() && done < total) {
                cancelled = true;
                return false;
            }
            return true;
        });
        rows = nullptr;   // (the pictures' rows are this request's; decoding reads tokens only)
        const size_t reused = ids.size() - (size_t) read;
        const double prompt_ms = ms_since(t0);
        if (cancelled) {
            // the state holds the tokens read so far (live): a request that continues them reuses it
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, reused, (long long) read);
            std::fflush(stdout);
            continue;
        }
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist;
        reset_detail();
        const auto d0 = Clock::now();
        const double cpu0 = cpu_ms, route0 = route_ms, tail0 = tail_ms;
        long long produced = 0, offered = 0, accepted_n = 0;
        const char* finish = "length";
        int32_t x = smp.needs_logits() ? sample(lg, smp, hist, rng) : (int32_t) (std::max_element(lg.begin(), lg.end()) - lg.begin());
        // drafts for greedy and for sampled requests (speculative sampling; STRATA_NEMO_SPEC_SAMPLED=0: sampled requests draft nothing)
        static const bool spec_sampled_on = !(std::getenv("STRATA_NEMO_SPEC_SAMPLED") && std::atoi(std::getenv("STRATA_NEMO_SPEC_SAMPLED")) == 0);
        const bool spec = o.mtp > 0 && has_mtp && (!smp.needs_logits() || spec_sampled_on);
        auto emit = [&](int32_t t) {   // false: the answer ends here
            std::printf("T %d\n", t);
            std::fflush(stdout);
            ++produced;
            hist.push_back(t);
            if (t == eos) {
                finish = "stop";
                return false;
            }
            if (o.adapt && o.adapt_every > 0 && produced % o.adapt_every == 0) rebalance(o.adapt_swaps, 0.9, 1);   // follow the answer
            if (produced >= max_new) return false;
            if (stop.load()) {
                finish = "cancel";
                return false;
            }
            return true;
        };
        for (;;) {
            if (!emit(x)) break;
            if (spec && pos + kDecMax + 1 <= ctx) {
                std::vector<int32_t> acc;
                int32_t nx = 0;
                const int a = smp.needs_logits() ? spec_step_sampled(x, o.mtp, smp, hist, rng, acc, nx) : spec_step(x, o.mtp, acc, nx);
                offered += last_offered;
                accepted_n += a;
                bool ended = false;
                for (int i = 1; i <= a && !ended; ++i)
                    if (!emit(acc[(size_t) i])) ended = true;
                if (ended) break;
                x = nx;
                continue;
            }
            if (pos + 1 > ctx) break;
            if (smp.needs_logits()) {
                static const bool gpu_sample_env = !(std::getenv("STRATA_NEMO_GPU_SAMPLE") && std::atoi(std::getenv("STRATA_NEMO_GPU_SAMPLE")) == 0);
                const int kk = o.gpu_sample && gpu_sample_env ? gpu_topk_k(smp, (int) g.n_vocab) : 0;
                decode_step(x, kk > 0 ? nullptr : &lg, kk);
                const auto ts0 = Clock::now();
                x = kk > 0 ? draw(dist_top(h_topv, h_topi, kk, smp), rng) : sample(lg, smp, hist, rng);
                samp_ms += ms_since(ts0);
            } else {
                x = decode_step(x, nullptr);
            }
        }
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms, decode_ms, finish,
                    accepted_n, offered, reused, (long long) read);
        std::fflush(stdout);
        std::fprintf(stderr, "strata-nemotron: prompt %zu tokens = %zu reused + %lld read in %.0f ms (%.1f tok/s), %lld generated in %.0f ms "
                     "(%.2f tok/s), drafts %lld of %lld; experts so far: %lld on the GPU (%lld of them copied over the link), %lld on the CPU (%.0f ms), %lld swaps\n",
                     ids.size(), reused, (long long) read, prompt_ms, prompt_ms > 0 ? 1000.0 * (double) read / prompt_ms : 0.0, produced,
                     decode_ms, decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, accepted_n, offered, gpu_ent, pcie_ent, cpu_ent, cpu_ms,
                     swaps);
        if (lk_offered) std::fprintf(stderr, "strata-nemotron: prompt lookup so far: %lld of %lld drafts kept\n", lk_accepted, lk_offered);
        std::fprintf(stderr, "strata-nemotron: decode %.0f ms = CPU experts %.0f + waiting for the GPU %.0f before them and %.0f after + the rest %.0f\n",
                     decode_ms, cpu_ms - cpu0, route_ms - route0, tail_ms - tail0, decode_ms - (cpu_ms - cpu0) - (route_ms - route0) - (tail_ms - tail0));
        print_detail(decode_ms);
        // STRATA_NEMO_ROUTE_DUMP=<file>: the decode's routing per (layer, expert) since the start, and whether it is in a slot now (a
        // CSV for studying the slots' placement; written after every request, before the swaps that follow it)
        if (const char* rd = std::getenv("STRATA_NEMO_ROUTE_DUMP"); rd && *rd) {
            if (std::FILE* f = std::fopen(rd, "w")) {
                std::fprintf(f, "layer,mtp,stage,n_slots,stride,expert,count,resident\n");
                for (int l = 0; l < (int) layers.size(); ++l) {
                    const Layer& L = layers[(size_t) l];
                    if (L.dec.empty()) continue;
                    for (int x = 0; x < g.n_expert; ++x)
                        std::fprintf(f, "%d,%d,%d,%d,%zu,%d,%.0f,%d\n", l, L.kind == BlockKind::MtpAttentionMoe ? 1 : 0, L.stage, L.n_slots,
                                     L.stride, x, L.dec[(size_t) x], L.slot_of[(size_t) x] >= 0 ? 1 : 0);
                }
                std::fclose(f);
            }
        }
        rebalance(512);
        if (!o.profile_save.empty() && o.profile_save_min > 0 && ms_since(last_save) >= o.profile_save_min * 60000.0) {
            if (!save_profile(o.profile_save)) std::fprintf(stderr, "strata-nemotron: could not save the expert profile to %s\n", o.profile_save.c_str());
            last_save = Clock::now();
        }
    }
}

// ================================================================================================================ the API
Engine::~Engine() = default;

std::unique_ptr<Engine> Engine::load(const std::string& gguf, const EngineOptions& o, std::string& err) {
    std::unique_ptr<Engine> e(new Engine());
    e->p_ = std::make_unique<EngineImpl>();
    e->p_->o = o;
    try {
        if (!e->p_->load(gguf, err)) return nullptr;
    } catch (const std::exception& ex) {
        err = ex.what();
        return nullptr;
    }
    return e;
}
void Engine::reset() { p_->reset_state(); }
int64_t Engine::feed(const std::vector<int32_t>& ids, std::vector<float>* last_logits) {
    if (ids.empty()) return 0;
    return p_->feed(ids, last_logits, nullptr);
}
int64_t Engine::feed(const std::vector<int32_t>& ids, const std::vector<const float*>& rows, std::vector<float>* last_logits) {
    if (ids.empty()) return 0;
    p_->rows = &rows;
    int64_t r = 0;
    try {
        r = p_->feed(ids, last_logits, nullptr);
    } catch (...) {
        p_->rows = nullptr;
        throw;
    }
    p_->rows = nullptr;
    return r;
}
std::vector<float> Engine::token_rows(const std::vector<int32_t>& ids) {
    EngineImpl& e = *p_;
    const int N = (int) e.g.n_embd, T = (int) ids.size();
    std::vector<float> out((size_t) T * N);
    if (T == 0) return out;
    Stage& s = e.stages[0];
    e.use(s);
    int32_t* d_t = nullptr;
    float* d_x = nullptr;
    CK(cudaMalloc(&d_t, (size_t) T * 4));
    CK(cudaMalloc(&d_x, (size_t) T * N * 4));
    CK(cudaMemcpyAsync(d_t, ids.data(), (size_t) T * 4, cudaMemcpyHostToDevice, s.st));
    K::iq_embed_rows(e.embd_type, s.embd, e.embd_row, d_t, T, N, d_x, s.st);
    CK(cudaStreamSynchronize(s.st));
    CK(cudaMemcpy(out.data(), d_x, out.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaFree(d_t));
    CK(cudaFree(d_x));
    return out;
}
int64_t Engine::n_embd() const { return p_->g.n_embd; }
std::vector<int32_t> Engine::greedy(int32_t first, int n, std::vector<float>* logits) {
    std::vector<int32_t> out;
    int32_t x = first;
    std::vector<float> lg;
    if (logits) logits->clear();
    while ((int) out.size() < n) {
        if (p_->o.mtp > 0 && p_->has_mtp && !logits && p_->pos + kDecMax + 1 <= p_->ctx) {
            std::vector<int32_t> acc;
            int32_t nx = 0;
            const int a = p_->spec_step(x, p_->o.mtp, acc, nx);
            for (int i = 1; i <= a && (int) out.size() < n; ++i) out.push_back(acc[(size_t) i]);
            if ((int) out.size() < n) out.push_back(nx);
            x = nx;
            continue;
        }
        x = p_->decode_step(x, logits ? &lg : nullptr);
        if (logits) logits->insert(logits->end(), lg.begin(), lg.end());
        out.push_back(x);
    }
    return out;
}
int64_t Engine::pos() const { return p_->pos; }
int64_t Engine::vocab() const { return p_->g.n_vocab; }
int Engine::serve() { return p_->serve(); }
int Engine::bench_cpu() { return p_->bench_cpu(); }

int run(int argc, char** argv) {
    EngineOptions o;
    std::string model, tokens;
    int n_gen = 32;
    bool serve_mode = false, bench = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--serve") serve_mode = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-nemotron: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--families") {
            std::printf("nemotron_h_moe\n");
            return 0;
        } else if (a == "--model" || a == "--native") model = next();
        else if (a == "--serve") {}
        else if (a == "--vision") {}   // GENI is always taken (the server adds the flag when the config has a vision section)
        else if (a == "--gpu") (void) next();   // applied by strata before the switch (CUDA_VISIBLE_DEVICES)
        else if (a == "--max-context" || a == "--ctx") o.ctx = std::atoi(next().c_str());
        else if (a == "--chunk" || a == "--prefill") {
            const std::string v = next();
            if (v != "auto") o.chunk = std::atoi(v.c_str());
        } else if (a == "--mtp") {
            const std::string v = next();
            if (!v.empty() && std::isdigit((unsigned char) v[0])) o.mtp = std::atoi(v.c_str());
        } else if (a == "--threads" || a == "--pool-workers" || a == "--cpu-threads") o.threads = std::atoi(next().c_str());
        else if (a == "--expert-vram") o.expert_vram_gib = std::atof(next().c_str());
        else if (a == "--expert-slots") o.expert_slots = std::atoi(next().c_str());
        else if (a == "--vram-reserve-mib") o.reserve_mib = std::atoi(next().c_str());
        else if (a == "--gpus") o.max_gpus = std::atoi(next().c_str());
        else if (a == "--layer-split") {
            const std::string v = next();
            if (v != "auto") o.layer_split = std::atoi(v.c_str());
        } else if (a == "--no-warm" || a == "--no-warmup") o.warm = false;
        else if (a == "--no-adapt") o.adapt = false;
        else if (a == "--no-prefill-pipe") o.prefill_pipe = false;
        else if (a == "--adapt-every") o.adapt_every = std::max(0, std::atoi(next().c_str()));
        else if (a == "--adapt-swaps") o.adapt_swaps = std::max(0, std::atoi(next().c_str()));
        else if (a == "--expert-profile") o.profile = next();
        else if (a == "--expert-profile-save") o.profile_save = next();
        else if (a == "--expert-profile-save-every") o.profile_save_min = std::max(0.0, std::atof(next().c_str()));
        else if (a == "--pcie-frac") o.pcie_frac = std::atof(next().c_str());
        else if (a == "--no-pin") o.pin_experts = false;
        else if (a == "--no-lookup") o.lookup = false;
        else if (a == "--pool-cores") o.pool_cores = next();
        else if (a == "--prompt-cache") o.checkpoints = std::max(0, std::atoi(next().c_str()));
        else if (a == "--prompt-cache-every") o.ckpt_every = std::max(0LL, std::atoll(next().c_str()));
        else if (a == "--prompt-cache-root") o.root_min = std::max(0LL, std::atoll(next().c_str()));
        else if (a == "--conversation-cache-mib") o.park_mib = std::max(0LL, std::atoll(next().c_str()));
        else if (a == "--conversation-cache-min-tokens") o.park_min = std::max(0LL, std::atoll(next().c_str()));
        else if (a == "--prompt-cpu-rows") o.prompt_cpu_rows = std::max(0, std::atoi(next().c_str()));
        else if (a == "--mtp-slots") o.mtp_slots = std::max(0, std::atoi(next().c_str()));
        else if (a == "--pin-gib") o.pin_gib = std::max(0.0, std::atof(next().c_str()));
        else if (a == "--no-gpu-sample") o.gpu_sample = false;
        else if (a == "--no-share-adapt") o.share_adapt = false;
        else if (a == "--bench-cpu") bench = true;
        else if (a == "--quiet") o.verbose = false;
        else if (a == "--tokens") tokens = next();
        else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (serve_mode && a.rfind("--", 0) == 0) {
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-nemotron: ignoring %s%s%s (not an option of this engine)\n", a.c_str(), val ? " " : "",
                         val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-nemotron --model GGUF [--max-context 131072] [--chunk 1024] [--mtp N] [--threads N]\n"
                         "       [--expert-vram GiB] [--vram-reserve-mib M] [--gpus N] [--layer-split K] [--no-warm]\n"
                         "       (--serve | --tokens IDS [--n 32] | --families)\n");
            return 2;
        }
    }
    if (model.empty()) {
        std::fprintf(stderr, "strata-nemotron: --model is required\n");
        return 2;
    }
    if (o.ctx < 256 || o.chunk < kDecMax + 1 || o.mtp < 0 || o.mtp > 7) {
        std::fprintf(stderr, "strata-nemotron: bad --max-context / --chunk / --mtp\n");
        return 2;
    }
    std::string err;
    std::unique_ptr<Engine> e = Engine::load(model, o, err);
    if (!e) {
        std::fprintf(stderr, "strata-nemotron: %s\n", err.c_str());
        return 1;
    }
    if (bench) return e->bench_cpu();
    if (serve_mode) return e->serve();
    if (tokens.empty()) {
        std::fprintf(stderr, "strata-nemotron: loaded; nothing to do (--serve or --tokens)\n");
        return 0;
    }
    const std::vector<int32_t> ids = parse_ids(tokens.c_str());
    std::vector<float> lg;
    e->feed(ids, &lg);
    const int32_t first = (int32_t) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const std::vector<int32_t> out = e->greedy(first, n_gen - 1);
    std::printf("%d", first);
    for (int32_t t : out) std::printf(",%d", t);
    std::printf("\n");
    return 0;
}

}  // namespace strata::nemotron
