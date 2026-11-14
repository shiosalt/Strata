// src/nemotron/attn_tc.cu - the prompt path's attention on tensor cores (FlashAttention-2 style): fp16 Q K^T and P V with fp32
// accumulation (WMMA 16x16x16), the online softmax in fp32.  Causal NoPE GQA over the fp16 K / V caches [Hk][ctx][D], D = 128.
// A block: 64 query rows of one query head, 4 warps of 16 rows; the cells in tiles of 32 through shared memory.  The decode path keeps
// attention() (its row-split arithmetic is what makes an MTP window compute what single steps compute).
#include "kernels.cuh"

#include <mma.h>

#include <atomic>

#include <stdexcept>
#include <string>

namespace strata::nemotron::gpu {

namespace {
using namespace nvcuda;
constexpr int kD = 128, kBq = 64, kBk = 32, kWarps = 4, kRows = 16;   // rows per warp

struct Smem {
    __half q[kBq * kD];
    __half k[kBk * kD];
    __half v[kBk * kD];
    float s[kWarps][kRows * kBk];
    __half p[kWarps][kRows * kBk];
    float o[kWarps][kRows * kD];
    float m[kBq], l[kBq];
};

__global__ void __launch_bounds__(kWarps * 32) k_attn_tc(const float* __restrict__ q, const __half* __restrict__ kc,
                                                         const __half* __restrict__ vc, int ctx, int pos0, int T, int Hq, int Hk,
                                                         float scale, float* __restrict__ out) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int h = blockIdx.y, kvh = h / (Hq / Hk);
    const int q0 = blockIdx.x * kBq, nq = min(kBq, T - q0);
    const size_t qld = (size_t) Hq * kD;
    // Q (scaled) to fp16; O, m, l zero
    for (int i = tid; i < kBq * kD; i += kWarps * 32) {
        const int r = i / kD, d = i % kD;
        sm.q[i] = __float2half(r < nq ? q[(size_t) (q0 + r) * qld + (size_t) h * kD + d] * scale : 0.0f);
    }
    for (int i = tid; i < kWarps * kRows * kD; i += kWarps * 32) (&sm.o[0][0])[i] = 0.0f;
    for (int i = tid; i < kBq; i += kWarps * 32) {
        sm.m[i] = -INFINITY;
        sm.l[i] = 0.0f;
    }
    const int cells = pos0 + q0 + nq;   // the last row sees cells 0 .. pos0 + q0 + nq - 1
    const __half* kbase = kc + (size_t) kvh * ctx * kD;
    const __half* vbase = vc + (size_t) kvh * ctx * kD;
    const int r = lane / 2, half_c = (lane % 2) * (kBk / 2);   // the softmax: two lanes a row, 16 columns each
    const int row = warp * kRows + r, qpos = pos0 + q0 + row;
    const bool row_ok = row < nq;
    for (int kb = 0; kb < cells; kb += kBk) {
        __syncthreads();   // (the last tile's K / V are read)
        // K and V tiles: 8 halves a load
        for (int i = tid; i < kBk * kD / 8; i += kWarps * 32) {
            const int c = (i * 8) / kD, d = (i * 8) % kD;
            int4 kv = make_int4(0, 0, 0, 0), vv = make_int4(0, 0, 0, 0);
            if (kb + c < cells) {
                kv = *reinterpret_cast<const int4*>(kbase + (size_t) (kb + c) * kD + d);
                vv = *reinterpret_cast<const int4*>(vbase + (size_t) (kb + c) * kD + d);
            }
            *reinterpret_cast<int4*>(sm.k + c * kD + d) = kv;
            *reinterpret_cast<int4*>(sm.v + c * kD + d) = vv;
        }
        __syncthreads();
        if (warp * kRows >= nq) continue;   // (a warp with no rows: only the barriers)
        // S = Q K^T (16 x 32)
        for (int n = 0; n < kBk / 16; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int k = 0; k < kD / 16; ++k) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(a, sm.q + (warp * kRows) * kD + k * 16, kD);
                wmma::load_matrix_sync(b, sm.k + (n * 16) * kD + k * 16, kD);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sm.s[warp] + n * 16, acc, kBk, wmma::mem_row_major);
        }
        __syncwarp();
        // the online softmax of this tile (causal), P to fp16, O's rows rescaled
        float* srow = sm.s[warp] + r * kBk;
        float mx = -INFINITY;
        for (int c = half_c; c < half_c + kBk / 2; ++c) {
            const int cell = kb + c;
            const float v = row_ok && cell <= qpos ? srow[c] : -INFINITY;
            srow[c] = v;
            mx = fmaxf(mx, v);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        const float mold = sm.m[row], mnew = fmaxf(mold, mx);
        float sum = 0.0f;
        for (int c = half_c; c < half_c + kBk / 2; ++c) {
            const float e = mnew == -INFINITY ? 0.0f : __expf(srow[c] - mnew);
            sm.p[warp][r * kBk + c] = __float2half(e);
            sum += e;
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        const float alpha = mold == -INFINITY ? 0.0f : __expf(mold - mnew);
        float* orow = sm.o[warp] + r * kD;
        for (int d = (lane % 2) * (kD / 2); d < (lane % 2 + 1) * (kD / 2); ++d) orow[d] *= alpha;
        __syncwarp();
        if (lane % 2 == 0) {
            sm.l[row] = sm.l[row] * alpha + sum;
            sm.m[row] = mnew;
        }
        __syncwarp();
        // O += P V (16 x 128)
        for (int n = 0; n < kD / 16; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::load_matrix_sync(acc, sm.o[warp] + n * 16, kD, wmma::mem_row_major);
            for (int k = 0; k < kBk / 16; ++k) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                wmma::load_matrix_sync(a, sm.p[warp] + k * 16, kBk);
                wmma::load_matrix_sync(b, sm.v + (k * 16) * kD + n * 16, kD);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(sm.o[warp] + n * 16, acc, kD, wmma::mem_row_major);
        }
        __syncwarp();
    }
    __syncthreads();
    for (int i = tid; i < nq * kD; i += kWarps * 32) {
        const int rr = i / kD, d = i % kD;
        const float l = sm.l[rr];
        out[(size_t) (q0 + rr) * qld + (size_t) h * kD + d] = l > 0.0f ? sm.o[rr / kRows][(rr % kRows) * kD + d] / l : 0.0f;
    }
}
}  // namespace

bool attention_tc_supported(int Hq, int Hk, int D) { return D == kD && Hk > 0 && Hq % Hk == 0; }

void attention_tc(const float* q, const __half* kc, const __half* vc, int ctx, int pos0, int T, int Hq, int Hk, int D, float scale,
                  float* out, cudaStream_t st) {
    if (T <= 0) return;
    if (!attention_tc_supported(Hq, Hk, D)) throw std::invalid_argument("attention_tc: unsupported head layout");
    if (pos0 + T > ctx) throw std::invalid_argument("attention_tc: past the cache");
    // the shared memory opt-in, once per device (the attribute is the device's; the prefill pipeline calls from one thread per GPU,
    // each on its own device, so a device's flag is only ever written by the thread on it)
    static std::atomic<bool> attr_done[64] = {};
    const int bytes = (int) sizeof(Smem);
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64 || !attr_done[dev].load()) {
        if (cudaFuncSetAttribute(k_attn_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) != cudaSuccess)
            throw std::runtime_error("nemotron kernel attention_tc: the shared memory could not be had");
        if (dev >= 0 && dev < 64) attr_done[dev].store(true);
    }
    k_attn_tc<<<dim3((unsigned) ((T + kBq - 1) / kBq), (unsigned) Hq), kWarps * 32, bytes, st>>>(q, kc, vc, ctx, pos0, T, Hq, Hk, scale, out);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("nemotron kernel attention_tc: ") + cudaGetErrorString(e));
}

}  // namespace strata::nemotron::gpu
