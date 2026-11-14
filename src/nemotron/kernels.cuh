// src/nemotron/kernels.cuh - the CUDA kernels of the nemotron_h_moe engine path that Strata has no kernel for: Mamba2 (conv,
// selective scan, grouped norm), NoPE GQA attention over an fp16 K / V cache, the sigmoid + bias router, relu^2, the MoE combine,
// LayerNorm.  Every kernel computes what include/strata/nemotron/reference.hpp computes (the fp32 CPU reference), and
// tests/nemotron/kernels_parity.cu checks each against it.  Quantized matrix products are not here: they are Strata's native
// MMVQ / MMQ (include/strata/kernels/native_mmvq.hpp, include/strata/prefill/moe_mmq.hpp).
//
// All pointers are device pointers; every call enqueues on `stream` and never synchronises.  Row-major [rows][n] everywhere.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strata::nemotron::gpu {

/// y[r] = rms(x[r]) * w (w may be null: no weight), `rows` rows of n.
void rms_rows(const float* x, const float* w, float* y, int n, int rows, float eps, cudaStream_t stream);
/// x[r] += y[r]; then, with `w`, h[r] = rms(x[r]) * w.
void add_rms_rows(float* x, const float* y, const float* w, float* h, int n, int rows, float eps, cudaStream_t stream);
/// y[r] = layernorm(x[r]) * w (no bias).
void layer_norm_rows(const float* x, const float* w, float* y, int n, int rows, float eps, cudaStream_t stream);
/// In place: every `group` consecutive values of each row of n get their own RMS norm, times w (n values, w[i] for value i).
void group_rms_rows(float* y, const float* w, int n, int group, int rows, float eps, cudaStream_t stream);

/// Mamba2's causal depthwise conv over the xBC channels of T rows: row t's channels are proj + t * ld + off (C of them);
/// state [K - 1][C] holds the previous tokens' channels, oldest first; w [C][K]; out [T][C] = silu(conv + b).  `write`: the
/// state advances by the T tokens (false: a verify window reads it only).
void mamba_conv(const float* proj, int ld, int off, float* state, const float* w, const float* b, float* out, int C, int K, int T,
                bool write, cudaStream_t stream);
/// Mamba2's selective scan of T rows (one block per head): x = xbc[t][0 .. H*P), B / C the next G*N each; dt = proj[t][dt_off + h]
/// + dt_b[h], z = proj[t][z_off + i]; state [H][P][N].  y[t][h*P + p] = (s . C + D x) * silu(z) (before the grouped norm).
/// Needs N % 32 == 0 or N <= 32, and P * max(1, N / 32) <= 1024.  `write` as for mamba_conv.
void mamba_scan(const float* xbc, int C, const float* proj, int ld, int z_off, int dt_off, const float* dt_b, const float* A,
                const float* D, float* state, float* y, int H, int P, int N, int G, int T, bool write, cudaStream_t stream);
/// Whether mamba_scan takes these sizes.
bool mamba_scan_supported(int H, int P, int N, int G);

/// K / V rows of T tokens (k, v: [T][Hk * D]) into the fp16 caches [Hk][ctx][D] at positions pos0 .. pos0 + T - 1.
void kv_store(const float* k, const float* v, __half* kc, __half* vc, int ctx, int pos0, int T, int Hk, int D, cudaStream_t stream);
/// Causal NoPE attention of T query rows q [T][Hq * D] at positions pos0 + t (each sees cells 0 .. pos0 + t) over the caches,
/// out [T][Hq * D].  Hq / Hk <= 16, D <= 128.  `splits` > 1 cuts every row's cells into that many ranges computed side by side
/// and merged (decode: T is small); it needs `scratch` of attn_scratch_floats(T, Hq, D, splits) floats.  splits == 1: no scratch.
/// With splits > 1 and D % 8 == 0 the window is computed in one pass over the cells instead, in fixed ranges of kAttnSplitCells
/// (more when ctx has over kAttnMaxSplits of them) that need at most `splits` of the scratch (STRATA_NEMO_ATTN_WIN=0: per row).
void attention(const float* q, const __half* kc, const __half* vc, int ctx, int pos0, int T, int Hq, int Hk, int D, float scale,
               int splits, float* scratch, float* out, cudaStream_t stream);
size_t attn_scratch_floats(int T, int Hq, int D, int splits);
/// The same on tensor cores for a prompt chunk (attn_tc.cu): fp16 Q K^T and P V, fp32 softmax and accumulation.  D == 128.
bool attention_tc_supported(int Hq, int Hk, int D);
void attention_tc(const float* q, const __half* kc, const __half* vc, int ctx, int pos0, int T, int Hq, int Hk, int D, float scale,
                  float* out, cudaStream_t stream);
/// The split count attention() should use for rows reaching `cells` cells (one range per kAttnSplitCells, at most kAttnMaxSplits).
/// With splits > 1 every row uses min(splits, attn_splits(its own cells)), so a row computes the same in a window as alone.
constexpr int kAttnSplitCells = 1024;
constexpr int kAttnMaxSplits = 256;
__host__ __device__ inline int attn_splits(int cells) {
    const int s = (cells + kAttnSplitCells - 1) / kAttnSplitCells;
    return s < 1 ? 1 : s > kAttnMaxSplits ? kAttnMaxSplits : s;
}

/// y[t][o] = W[o] . x[t] for an F32 matrix W [n_out][n_in] and T rows of x [T][n_in] (a warp per output row; any T).
void gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out, int T, cudaStream_t stream);
/// The router: per token t, p = sigmoid(logits[t]); the K experts with the largest p + bias (lowest id on ties) into ids[t][K]; their
/// weights p (normalised to sum 1 when `norm`, the sum clamped at 2^-14) times `scale` (when not 0 or 1) into w[t][K].
/// E <= 1024, K <= 32.
void router_topk(const float* logits, int ld, const float* bias, int E, int K, int T, bool norm, float scale, int32_t* ids, float* w,
                 cudaStream_t stream);
/// x = relu(x)^2, n values.
void relu2(float* x, int64_t n, cudaStream_t stream);
/// out[t] = sum_k w[t][k] * parts[row(t * K + k)] (rows of L), in pick order; row(e) = rows ? rows[e] : e.
void moe_combine(const float* parts, const float* w, int K, int L, int T, float* out, cudaStream_t stream,
                 const int32_t* rows = nullptr);
/// The prompt's routed entries (n_ent = tokens x K, entry e = t * K + k) sorted by expert: bounds[x] .. bounds[x + 1] are expert x's
/// sorted rows; sorted row r is entry dst[r] of token src[r]; inv[e] = entry e's sorted row.  cnt / cur: n_expert ints of scratch
/// (cnt is zeroed here); max_rows: the most rows one expert has.  The order inside an expert is the atomics' (no output depends on it).
void route_sort(const int32_t* ids, int n_ent, int K, int n_expert, int32_t* cnt, int32_t* cur, int32_t* bounds, int32_t* dst,
                int32_t* src, int32_t* inv, int32_t* max_rows, cudaStream_t stream);
/// dst row r = src row r where mask[r] != 0 (rows of n), `rows` rows: the CPU's expert rows over the GPU's.
void merge_rows(float* dst, const float* src, const int32_t* mask, int n, int rows, cudaStream_t stream);
/// y[r * ld_y + i] = rms(x[r]) * w[i], i < n (a norm into a slice of a wider row).
void rms_rows_ld(const float* x, const float* w, float* y, int ld_y, int n, int rows, float eps, cudaStream_t stream);
/// x += y, n values.
void add(float* x, const float* y, int64_t n, cudaStream_t stream);
/// dst[i] = src[idx[i]] rows of n floats, `rows` of them.
void gather_rows(const float* src, const int32_t* idx, int n, int rows, float* dst, cudaStream_t stream);
/// fp32 -> fp16 bits, n values.
void to_f16(const float* x, __half* y, int64_t n, cudaStream_t stream);
/// out[r] = argmax of row r of x [rows][n] (lowest id on ties).
void argmax_rows(const float* x, int n, int rows, int32_t* out, cudaStream_t stream);
/// The K largest values of each row of x [rows][n], largest first: ids [rows][K] and vals [rows][K] (topk.cu).  K <= kTopKMax.
/// Equal values: which ids are taken among them is the atomics' order (only the count is fixed).
constexpr int kTopKMax = 256;
void topk_rows(const float* x, int n, int rows, int K, int32_t* ids, float* vals, cudaStream_t stream);

}  // namespace strata::nemotron::gpu
