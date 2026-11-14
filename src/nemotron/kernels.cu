// src/nemotron/kernels.cu - see kernels.cuh.
#include "kernels.cuh"

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::nemotron::gpu {

namespace {
constexpr int kT = 256;   // threads of the row kernels

__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float softplus(float v) { return v > 20.f ? v : logf(1.0f + expf(v)); }

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

unsigned grid1(int64_t n, int b = kT) { return (unsigned) ((n + b - 1) / b); }

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("nemotron kernel ") + what + ": " + cudaGetErrorString(e));
}

// ---------------------------------------------------------------------------------------------------------------- norms
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kT];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * (w ? w[i] : 1.0f);
}

__global__ void k_add_rms(float* x, const float* y, const float* w, float* h, int n, float eps) {
    __shared__ float sh[kT];
    float* xr = x + (size_t) blockIdx.x * n;
    const float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    if (!w) return;
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    float* hr = h + (size_t) blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) hr[i] = xr[i] * r * w[i];
}

__global__ void k_layer_norm(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kT];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i];
    const float mean = block_sum(s, sh) / (float) n;
    float v = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) v += (xr[i] - mean) * (xr[i] - mean);
    const float r = rsqrtf(block_sum(v, sh) / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = (xr[i] - mean) * r * (w ? w[i] : 1.0f);
}

/// grid (rows * n / group): block b normalises group b % (n / group) of row b / (n / group).
__global__ void k_group_rms(float* y, const float* w, int n, int group, float eps) {
    __shared__ float sh[kT];
    const int per_row = n / group;
    const int row = blockIdx.x / per_row, gi = blockIdx.x % per_row;
    float* yr = y + (size_t) row * n + (size_t) gi * group;
    const float* wr = w + (size_t) gi * group;
    float s = 0.f;
    for (int i = threadIdx.x; i < group; i += blockDim.x) s += yr[i] * yr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) group + eps);
    for (int i = threadIdx.x; i < group; i += blockDim.x) yr[i] = yr[i] * r * wr[i];
}

// ---------------------------------------------------------------------------------------------------------------- Mamba2
constexpr int kMaxConv = 8;

__global__ void k_mamba_conv(const float* proj, int ld, int off, float* state, const float* w, const float* b, float* out, int C,
                             int K, int T, bool write) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float hist[kMaxConv], wk[kMaxConv];
    for (int k = 0; k < K; ++k) wk[k] = w[(size_t) c * K + k];
    for (int k = 0; k < K - 1; ++k) hist[k] = state[(size_t) k * C + c];
    const float bias = b ? b[c] : 0.0f;
    for (int t = 0; t < T; ++t) {
        const float in = proj[(size_t) t * ld + off + c];
        float acc = 0.0f;
        for (int k = 0; k < K - 1; ++k) acc += hist[k] * wk[k];
        acc += in * wk[K - 1];
        out[(size_t) t * C + c] = silu(acc + bias);
        for (int k = 0; k + 1 < K - 1; ++k) hist[k] = hist[k + 1];
        if (K > 1) hist[K - 2] = in;
    }
    if (write)
        for (int k = 0; k < K - 1; ++k) state[(size_t) k * C + c] = hist[k];
}

constexpr int kMaxNs = 32;   // states a thread keeps in registers

/// One block per head: thread (p, sub) keeps states n = sub * NS .. + NS of head dim p.
__global__ void k_mamba_scan(const float* xbc, int C, const float* proj, int ld, int z_off, int dt_off, const float* dt_b,
                             const float* A, const float* Dv, float* state, float* y, int H, int P, int N, int G, int T, int tpp,
                             bool write) {
    extern __shared__ float bc[];   // B [N] then C [N] of this head's group for the current token
    const int h = blockIdx.x;
    const int p = threadIdx.x / tpp, sub = threadIdx.x % tpp;
    const int ns = N / tpp;
    const int DI = H * P;
    const int g = h / (H / G);
    float s[kMaxNs];
    float* sp = state + ((size_t) h * P + p) * N + (size_t) sub * ns;
    for (int k = 0; k < ns; ++k) s[k] = sp[k];
    const unsigned mask = blockDim.x >= 32 ? 0xffffffffu : ((1u << blockDim.x) - 1u);
    const float a = A[h], dpar = Dv[h], dtb = dt_b[h];
    for (int t = 0; t < T; ++t) {
        const float* row = xbc + (size_t) t * C;
        __syncthreads();
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            bc[i] = row[DI + g * N + i];
            bc[N + i] = row[DI + G * N + g * N + i];
        }
        __syncthreads();
        const float spv = softplus(proj[(size_t) t * ld + dt_off + h] + dtb);
        const float dA = expf(spv * a);
        const float x = row[h * P + p];
        const float xdt = x * spv;
        float acc = 0.0f;
        const float* Bs = bc + sub * ns;
        const float* Cs = bc + N + sub * ns;
        for (int k = 0; k < ns; ++k) {
            s[k] = s[k] * dA + Bs[k] * xdt;
            acc += s[k] * Cs[k];
        }
        for (int o = tpp / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(mask, acc, o, tpp);
        if (sub == 0) {
            const float z = proj[(size_t) t * ld + z_off + h * P + p];
            y[(size_t) t * DI + h * P + p] = (acc + x * dpar) * silu(z);
        }
    }
    if (write)
        for (int k = 0; k < ns; ++k) sp[k] = s[k];
}

int scan_tpp(int N) { return N <= kMaxNs ? 1 : N / kMaxNs; }

// ---------------------------------------------------------------------------------------------------------------- attention
constexpr int kMaxRep = 16, kMaxD = 128, kTile = 64, kAttnT = 256, kAccPer = kMaxRep * kMaxD / kAttnT;

__global__ void k_kv_store(const float* k, const float* v, __half* kc, __half* vc, int ctx, int pos0, int T, int Hk, int D) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * Hk * D) return;
    const int d = (int) (i % D), hh = (int) ((i / D) % Hk), t = (int) (i / ((int64_t) D * Hk));
    const size_t dst = ((size_t) hh * ctx + pos0 + t) * D + d;
    kc[dst] = __float2half(k[i]);
    vc[dst] = __float2half(v[i]);
}

/// grid (splits, T * Hk); block kAttnT.  The query heads of one KV head (R of them) over the cells of one split, online softmax.
__global__ void __launch_bounds__(kAttnT) k_attn(const float* q, const __half* kc, const __half* vc, int ctx, int pos0, int Hq,
                                                 int Hk, int D, float scale, int splits, float* pacc, float* pm, float* pl,
                                                 float* out) {
    __shared__ float qs[kMaxRep * kMaxD];
    __shared__ float sc[kMaxRep * kTile];
    __shared__ float m_s[kMaxRep], l_s[kMaxRep], c_s[kMaxRep];
    const int split = blockIdx.x;
    const int t = blockIdx.y / Hk, kvh = blockIdx.y % Hk;
    const int R = Hq / Hk;
    const int npos = pos0 + t + 1;
    // the row's own split count (attn_splits of its cells): a row gets the same arithmetic in a window of any length as alone
    const int rs = splits == 1 ? 1 : min(splits, attn_splits(npos));
    const int per = (npos + rs - 1) / rs;
    const int c0 = split * per, c1 = min(npos, c0 + per);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    for (int i = tid; i < R * D; i += kAttnT) qs[i] = q[(size_t) t * Hq * D + (size_t) kvh * R * D + i];
    if (tid < R) {
        m_s[tid] = -INFINITY;
        l_s[tid] = 0.0f;
    }
    float acc[kAccPer];
#pragma unroll
    for (int j = 0; j < kAccPer; ++j) acc[j] = 0.0f;
    __syncthreads();
    const __half* kbase = kc + (size_t) kvh * ctx * D;
    const __half* vbase = vc + (size_t) kvh * ctx * D;
    for (int cb = c0; cb < c1; cb += kTile) {
        const int nc = min(kTile, c1 - cb);
        for (int i = tid; i < R * kTile; i += kAttnT) {
            const int r = i / kTile, c = i % kTile;
            float s = -INFINITY;
            if (c < nc) {
                const __half* kr = kbase + (size_t) (cb + c) * D;
                const float* qr = qs + r * D;
                float d = 0.0f;
                for (int e = 0; e < D; ++e) d += qr[e] * __half2float(kr[e]);
                s = d * scale;
            }
            sc[i] = s;
        }
        __syncthreads();
        for (int r = warp; r < R; r += kAttnT / 32) {
            float mx = -INFINITY;
            for (int c = lane; c < nc; c += 32) mx = fmaxf(mx, sc[r * kTile + c]);
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float mold = m_s[r];
            const float mnew = fmaxf(mold, mx);
            float sum = 0.0f;
            for (int c = lane; c < kTile; c += 32) {
                const float e = c < nc ? expf(sc[r * kTile + c] - mnew) : 0.0f;
                sc[r * kTile + c] = e;
                sum += e;
            }
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            if (lane == 0) {
                const float corr = mold == -INFINITY ? 0.0f : expf(mold - mnew);
                l_s[r] = l_s[r] * corr + sum;
                m_s[r] = mnew;
                c_s[r] = corr;
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < kAccPer; ++j) {
            const int idx = tid + j * kAttnT;
            if (idx >= R * D) break;
            const int r = idx / D, d = idx % D;
            float a = acc[j] * c_s[r];
            const float* pr = sc + r * kTile;
            for (int c = 0; c < nc; ++c) a += pr[c] * __half2float(vbase[(size_t) (cb + c) * D + d]);
            acc[j] = a;
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < kAccPer; ++j) {
        const int idx = tid + j * kAttnT;
        if (idx >= R * D) break;
        const int r = idx / D, d = idx % D;
        const int qrow = t * Hq + kvh * R + r;
        if (splits == 1) {
            out[(size_t) qrow * D + d] = acc[j] / l_s[r];
        } else {
            pacc[((size_t) qrow * splits + split) * D + d] = acc[j];
            if (d == 0) {
                pm[(size_t) qrow * splits + split] = m_s[r];
                pl[(size_t) qrow * splits + split] = l_s[r];
            }
        }
    }
}

/// grid (T * Hq), block D.
__global__ void k_attn_merge(const float* pacc, const float* pm, const float* pl, int splits, int D, float* out) {
    const int row = blockIdx.x, d = threadIdx.x;
    float M = -INFINITY;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, pm[(size_t) row * splits + s]);
    float num = 0.0f, den = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float l = pl[(size_t) row * splits + s];
        if (l == 0.0f) continue;
        const float w = expf(pm[(size_t) row * splits + s] - M);
        num += w * pacc[((size_t) row * splits + s) * D + d];
        den += w * l;
    }
    out[(size_t) row * D + d] = num / den;
}

// The decode / verify window's attention: a block takes one KV head, one fixed range of `chunk` cells and up to kWinQ of the window's
// query rows (row-major (token, head)), so the K / V of a tile is read once into shared memory for every row of the window (the
// kernel above reads it again for each row, and one cell's K two bytes at a time: 22 GB/s on an RTX 5060 Ti).  The ranges and the
// tiles start at fixed cells (not at the row's own count), and each row stops at its own last cell, so a row's arithmetic is the
// same in a window of any length as alone.  Shared: Q [kWinQ][D] f32, K transposed [D][kWinC], V [kWinC][D], P [kWinQ][kWinC] f32.
constexpr int kWinQ = 64, kWinC = 64, kWinT = 256;
static size_t win_smem(int D) {
    return (size_t) kWinQ * D * 4 + (size_t) D * kWinC * 2 + (size_t) kWinC * D * 2 + (size_t) kWinQ * kWinC * 4 + (size_t) kWinQ * 16;
}

__global__ void __launch_bounds__(kWinT) k_attn_win(const float* q, const __half* kc, const __half* vc, int ctx, int pos0, int T, int Hq,
                                                    int Hk, int D, float scale, int chunk, int nch, float* pacc, float* pm, float* pl) {
    extern __shared__ __align__(16) unsigned char win_sm[];
    float* Qs = (float*) win_sm;
    __half* Kt = (__half*) (Qs + (size_t) kWinQ * D);
    __half* Vs = Kt + (size_t) D * kWinC;
    float* P = (float*) (Vs + (size_t) kWinC * D);
    float* m_s = P + kWinQ * kWinC;
    float* l_s = m_s + kWinQ;
    float* c_s = l_s + kWinQ;
    int* sk = (int*) (c_s + kWinQ);   // the tile has none of the row's cells: its state is left as it is
    const int ci = blockIdx.x, kvh = blockIdx.y, q0 = blockIdx.z * kWinQ;
    const int R = Hq / Hk, nq = T * R, nql = min(kWinQ, nq - q0);
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int npos_max = pos0 + (q0 + nql - 1) / R + 1;   // the last row of this block's queries
    const int c0 = ci * chunk, c1 = min(c0 + chunk, npos_max);
    for (int i = tid; i < kWinQ * D; i += kWinT) {
        const int ql = i / D, d = i % D, qg = q0 + ql;
        Qs[i] = ql < nql ? q[((size_t) (qg / R) * Hq + (size_t) kvh * R + qg % R) * D + d] : 0.0f;
    }
    if (tid < kWinQ) {
        m_s[tid] = -INFINITY;
        l_s[tid] = 0.0f;
    }
    // this thread: queries qa .. qa + 3; the scores of cells sub + 16 j, the outputs of dims 2 sub + 32 j (+1)
    const int qa = (tid / 16) * 4, sub = tid % 16;
    int npos_q[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) npos_q[i] = pos0 + (q0 + qa + i) / R + 1;
    float acc[4][8];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;
    const __half* kbase = kc + (size_t) kvh * ctx * D;
    const __half* vbase = vc + (size_t) kvh * ctx * D;
    const int D8 = D / 8;
    for (int cb = c0; cb < c1; cb += kWinC) {
        const int nc = min(kWinC, c1 - cb);
        __syncthreads();   // (the last tile's readers are done; the first time: Q and the state are written)
        for (int u = tid; u < kWinC * D8; u += kWinT) {
            const int c = u / D8, d0 = (u % D8) * 8;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (c < nc) {
                kv = *(const uint4*) (kbase + (size_t) (cb + c) * D + d0);
                vv = *(const uint4*) (vbase + (size_t) (cb + c) * D + d0);
            }
            *(uint4*) (Vs + (size_t) c * D + d0) = vv;
            const __half* kh = (const __half*) &kv;
#pragma unroll
            for (int i = 0; i < 8; ++i) Kt[(size_t) (d0 + i) * kWinC + c] = kh[i];
        }
        __syncthreads();
        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) s[i][j] = 0.0f;
        for (int d = 0; d < D; ++d) {
            float qv[4], kv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) qv[i] = Qs[(size_t) (qa + i) * D + d];
#pragma unroll
            for (int j = 0; j < 4; ++j) kv[j] = __half2float(Kt[(size_t) d * kWinC + sub + 16 * j]);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) s[i][j] += qv[i] * kv[j];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub + 16 * j;
                const bool ok = qa + i < nql && c < nc && cb + c < npos_q[i];
                P[(qa + i) * kWinC + c] = ok ? s[i][j] * scale : -INFINITY;
            }
        __syncthreads();
        for (int ql = warp; ql < kWinQ; ql += kWinT / 32) {
            const float a = P[ql * kWinC + lane], b = P[ql * kWinC + lane + 32];
            float mx = fmaxf(a, b);
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            if (mx == -INFINITY) {   // (the same for the whole warp)
                if (lane == 0) sk[ql] = 1;
                continue;
            }
            const float mold = m_s[ql], mnew = fmaxf(mold, mx);
            const float ea = expf(a - mnew), eb = expf(b - mnew);
            P[ql * kWinC + lane] = ea;
            P[ql * kWinC + lane + 32] = eb;
            float sum = ea + eb;
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            if (lane == 0) {
                const float corr = mold == -INFINITY ? 0.0f : expf(mold - mnew);
                l_s[ql] = l_s[ql] * corr + sum;
                m_s[ql] = mnew;
                c_s[ql] = corr;
                sk[ql] = 0;
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int ql = qa + i;
            if (ql >= nql || sk[ql]) continue;
            const float corr = c_s[ql];
            const int lim = min(nc, npos_q[i] - cb);   // (the row's own cells of the tile only)
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[i][j] *= corr;
            const float* pr = P + ql * kWinC;
            for (int c = 0; c < lim; ++c) {
                const float p = pr[c];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int d = 2 * sub + 32 * j;
                    if (d < D) {
                        const float2 f = __half22float2(*(const __half2*) (Vs + (size_t) c * D + d));
                        acc[i][2 * j] += p * f.x;
                        acc[i][2 * j + 1] += p * f.y;
                    }
                }
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int ql = qa + i;
        if (ql >= nql) continue;
        const int qg = q0 + ql;
        const size_t qrow = (size_t) (qg / R) * Hq + (size_t) kvh * R + qg % R;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = 2 * sub + 32 * j;
            if (d < D) {
                pacc[(qrow * nch + ci) * D + d] = acc[i][2 * j];
                pacc[(qrow * nch + ci) * D + d + 1] = acc[i][2 * j + 1];
            }
        }
        if (sub == 0) {
            pm[qrow * nch + ci] = m_s[ql];
            pl[qrow * nch + ci] = l_s[ql];
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------- MoE, misc
/// A warp per output row; T columns in groups of 8.
__global__ void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out, int T) {
    const int o = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (o >= n_out) return;
    const float* wr = W + (size_t) o * n_in;
    for (int t0 = 0; t0 < T; t0 += 8) {
        const int nt = min(8, T - t0);
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = lane; i < n_in; i += 32) {
            const float wv = wr[i];
            for (int c = 0; c < nt; ++c) acc[c] += wv * x[(size_t) (t0 + c) * n_in + i];
        }
        for (int c = 0; c < nt; ++c) {
            float v = acc[c];
            for (int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xffffffffu, v, s);
            if (lane == 0) y[(size_t) (t0 + c) * n_out + o] = v;
        }
    }
}

constexpr int kRouterPer = 4;   // E <= kT * kRouterPer

__global__ void k_router_topk(const float* logits, int ld, const float* bias, int E, int K, bool norm, float scale, int32_t* ids,
                              float* w) {
    __shared__ float bv[kT];
    __shared__ int bi[kT];
    __shared__ float pw[32];
    __shared__ int pid[32];
    const int t = blockIdx.x, tid = threadIdx.x;
    const float* lg = logits + (size_t) t * ld;
    float prob[kRouterPer], sel[kRouterPer];
    for (int j = 0; j < kRouterPer; ++j) {
        const int e = tid + j * kT;
        if (e < E) {
            prob[j] = sigm(lg[e]);
            sel[j] = prob[j] + (bias ? bias[e] : 0.0f);
        } else {
            prob[j] = 0.0f;
            sel[j] = -INFINITY;
        }
    }
    for (int k = 0; k < K; ++k) {
        float v = -INFINITY;
        int id = 0x7fffffff;
        for (int j = 0; j < kRouterPer; ++j) {
            const int e = tid + j * kT;
            if (sel[j] > v || (sel[j] == v && e < id)) {
                v = sel[j];
                id = e;
            }
        }
        bv[tid] = v;
        bi[tid] = id;
        __syncthreads();
        for (int s = kT / 2; s > 0; s >>= 1) {
            if (tid < s) {
                const float ov = bv[tid + s];
                const int oi = bi[tid + s];
                if (ov > bv[tid] || (ov == bv[tid] && oi < bi[tid])) {
                    bv[tid] = ov;
                    bi[tid] = oi;
                }
            }
            __syncthreads();
        }
        const int best = bi[0];
        __syncthreads();
        if (best % kT == tid) {
            const int j = best / kT;
            pw[k] = prob[j];
            pid[k] = best;
            sel[j] = -INFINITY;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += pw[k];
        const float inv = norm ? 1.0f / fmaxf(sum, 6.103515625e-5f) : 1.0f;
        const bool sc = scale != 0.0f && scale != 1.0f;
        for (int k = 0; k < K; ++k) {
            float v = pw[k] * inv;
            if (sc) v *= scale;
            ids[(size_t) t * K + k] = pid[k];
            w[(size_t) t * K + k] = v;
        }
    }
}

__global__ void k_relu2(float* x, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i];
        x[i] = v > 0.f ? v * v : 0.f;
    }
}

__global__ void k_combine(const float* parts, const float* w, int K, int L, int T, float* out, const int32_t* rows) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) T * L) return;
    const int t = (int) (i / L), c = (int) (i % L);
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int e = t * K + k;
        acc += w[e] * parts[(size_t) (rows ? rows[e] : e) * L + c];
    }
    out[i] = acc;
}

__global__ void k_route_count(const int32_t* ids, int n_ent, int32_t* cnt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n_ent) atomicAdd(&cnt[ids[e]], 1);
}
__global__ void k_route_scan(const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur, int32_t* max_rows) {
    if (threadIdx.x != 0) return;
    int pos = 0, mx = 0;
    for (int x = 0; x < n_expert; ++x) {
        bounds[x] = pos;
        cur[x] = pos;
        pos += cnt[x];
        mx = max(mx, cnt[x]);
    }
    bounds[n_expert] = pos;
    *max_rows = mx;
}
__global__ void k_route_scatter(const int32_t* ids, int n_ent, int K, int32_t* cur, int32_t* dst, int32_t* src, int32_t* inv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    const int r = atomicAdd(&cur[ids[e]], 1);
    dst[r] = e;
    src[r] = e / K;
    inv[e] = r;
}
__global__ void k_merge_rows(float* dst, const float* src, const int32_t* mask, int n, int rows) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n * rows) return;
    if (mask[i / n]) dst[i] = src[i];
}
__global__ void k_rms_ld(const float* x, const float* w, float* y, int ld_y, int n, float eps) {
    __shared__ float sh[kT];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * ld_y;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * (w ? w[i] : 1.0f);
}

__global__ void k_add(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void k_gather_rows(const float* src, const int32_t* idx, int n, int rows, float* dst) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n * rows) return;
    const int r = (int) (i / n), c = (int) (i % n);
    dst[i] = src[(size_t) idx[r] * n + c];
}

__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

__global__ void k_argmax(const float* x, int n, int32_t* out) {
    __shared__ float bv[kT];
    __shared__ int bi[kT];
    const float* xr = x + (size_t) blockIdx.x * n;
    float v = -INFINITY;
    int id = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += kT)
        if (xr[i] > v) {
            v = xr[i];
            id = i;
        }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = id;
    __syncthreads();
    for (int s = kT / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float ov = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = bi[0] == 0x7fffffff ? 0 : bi[0];
}
}  // namespace

// ---------------------------------------------------------------------------------------------------------------- host wrappers
void rms_rows(const float* x, const float* w, float* y, int n, int rows, float eps, cudaStream_t st) {
    if (rows <= 0) return;
    k_rms<<<rows, kT, 0, st>>>(x, w, y, n, eps);
    check_launch("rms_rows");
}
void add_rms_rows(float* x, const float* y, const float* w, float* h, int n, int rows, float eps, cudaStream_t st) {
    if (rows <= 0) return;
    k_add_rms<<<rows, kT, 0, st>>>(x, y, w, h, n, eps);
    check_launch("add_rms_rows");
}
void layer_norm_rows(const float* x, const float* w, float* y, int n, int rows, float eps, cudaStream_t st) {
    if (rows <= 0) return;
    k_layer_norm<<<rows, kT, 0, st>>>(x, w, y, n, eps);
    check_launch("layer_norm_rows");
}
void group_rms_rows(float* y, const float* w, int n, int group, int rows, float eps, cudaStream_t st) {
    if (rows <= 0) return;
    if (group <= 0 || n % group) throw std::invalid_argument("group_rms_rows: n is not a multiple of the group");
    k_group_rms<<<rows * (n / group), kT, 0, st>>>(y, w, n, group, eps);
    check_launch("group_rms_rows");
}
void mamba_conv(const float* proj, int ld, int off, float* state, const float* w, const float* b, float* out, int C, int K, int T,
                bool write, cudaStream_t st) {
    if (K < 1 || K > kMaxConv) throw std::invalid_argument("mamba_conv: conv kernel width out of range");
    if (T <= 0) return;
    k_mamba_conv<<<grid1(C, 128), 128, 0, st>>>(proj, ld, off, state, w, b, out, C, K, T, write);
    check_launch("mamba_conv");
}
bool mamba_scan_supported(int H, int P, int N, int G) {
    if (H <= 0 || P <= 0 || N <= 0 || G <= 0 || H % G) return false;
    const int tpp = scan_tpp(N);
    if (N > kMaxNs && N % kMaxNs) return false;
    if (tpp & (tpp - 1)) return false;   // a power of two (the shuffle width)
    return P * tpp <= 1024;
}
void mamba_scan(const float* xbc, int C, const float* proj, int ld, int z_off, int dt_off, const float* dt_b, const float* A,
                const float* D, float* state, float* y, int H, int P, int N, int G, int T, bool write, cudaStream_t st) {
    if (!mamba_scan_supported(H, P, N, G)) throw std::invalid_argument("mamba_scan: unsupported sizes");
    if (T <= 0) return;
    const int tpp = scan_tpp(N);
    k_mamba_scan<<<H, P * tpp, (size_t) 2 * N * sizeof(float), st>>>(xbc, C, proj, ld, z_off, dt_off, dt_b, A, D, state, y, H, P, N,
                                                                       G, T, tpp, write);
    check_launch("mamba_scan");
}
void kv_store(const float* k, const float* v, __half* kc, __half* vc, int ctx, int pos0, int T, int Hk, int D, cudaStream_t st) {
    if (T <= 0) return;
    if (pos0 < 0 || pos0 + T > ctx) throw std::invalid_argument("kv_store: past the cache");
    k_kv_store<<<grid1((int64_t) T * Hk * D), kT, 0, st>>>(k, v, kc, vc, ctx, pos0, T, Hk, D);
    check_launch("kv_store");
}
size_t attn_scratch_floats(int T, int Hq, int D, int splits) { return (size_t) T * Hq * splits * (D + 2); }
void attention(const float* q, const __half* kc, const __half* vc, int ctx, int pos0, int T, int Hq, int Hk, int D, float scale,
               int splits, float* scratch, float* out, cudaStream_t st) {
    if (T <= 0) return;
    if (Hk <= 0 || Hq % Hk || Hq / Hk > kMaxRep || D > kMaxD || D <= 0)
        throw std::invalid_argument("attention: unsupported head layout");
    if (pos0 + T > ctx) throw std::invalid_argument("attention: past the cache");
    if (splits < 1) splits = 1;
    float* pacc = nullptr;
    float* pm = nullptr;
    float* pl = nullptr;
    if (splits > 1) {
        if (!scratch) throw std::invalid_argument("attention: splits > 1 needs scratch");
        pacc = scratch;
        pm = pacc + (size_t) T * Hq * splits * D;
        pl = pm + (size_t) T * Hq * splits;
    }
    // decode / verify windows (splits > 1): every row of the window in one pass over the K / V (k_attn_win), in fixed ranges of
    // kAttnSplitCells cells (more when the cache has over kAttnMaxSplits of them); STRATA_NEMO_ATTN_WIN=0: the row-split kernel
    static const bool win_env = !(std::getenv("STRATA_NEMO_ATTN_WIN") && std::atoi(std::getenv("STRATA_NEMO_ATTN_WIN")) == 0);
    if (splits > 1 && win_env && D % 8 == 0) {
        int chunk = kAttnSplitCells;
        if ((int64_t) ctx > (int64_t) kAttnSplitCells * kAttnMaxSplits)
            chunk = (int) (((int64_t) ctx + kAttnMaxSplits - 1) / kAttnMaxSplits + kWinC - 1) / kWinC * kWinC;
        const int nch = (pos0 + T + chunk - 1) / chunk;
        if (nch <= splits) {
            const size_t smem = win_smem(D);
            static std::atomic<bool> attr_set[64] = {};   // (per device; the prefill pipeline has a thread per GPU)
            int dev = 0;
            cudaGetDevice(&dev);
            if (dev < 0 || dev >= 64 || !attr_set[dev].load()) {
                // (once a device, for the largest D: a later call with a larger D than the first must still launch)
                if (cudaFuncSetAttribute(k_attn_win, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) win_smem(kMaxD)) != cudaSuccess)
                    throw std::runtime_error("attention: the window kernel's shared memory could not be had");
                if (dev >= 0 && dev < 64) attr_set[dev].store(true);
            }
            const int groups = (T * (Hq / Hk) + kWinQ - 1) / kWinQ;
            k_attn_win<<<dim3(nch, Hk, groups), kWinT, smem, st>>>(q, kc, vc, ctx, pos0, T, Hq, Hk, D, scale, chunk, nch, pacc, pm, pl);
            check_launch("attention (window)");
            k_attn_merge<<<T * Hq, D, 0, st>>>(pacc, pm, pl, nch, D, out);
            check_launch("attention merge");
            return;
        }
    }
    k_attn<<<dim3(splits, T * Hk), kAttnT, 0, st>>>(q, kc, vc, ctx, pos0, Hq, Hk, D, scale, splits, pacc, pm, pl, out);
    check_launch("attention");
    if (splits > 1) {
        k_attn_merge<<<T * Hq, D, 0, st>>>(pacc, pm, pl, splits, D, out);
        check_launch("attention merge");
    }
}
void gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out, int T, cudaStream_t st) {
    if (T <= 0) return;
    k_gemv_f32<<<grid1(n_out, 8), 256, 0, st>>>(W, x, y, n_in, n_out, T);
    check_launch("gemv_f32");
}
void router_topk(const float* logits, int ld, const float* bias, int E, int K, int T, bool norm, float scale, int32_t* ids, float* w,
                 cudaStream_t st) {
    if (E > kT * kRouterPer || K > 32 || K > E || K < 1) throw std::invalid_argument("router_topk: E or K out of range");
    if (T <= 0) return;
    k_router_topk<<<T, kT, 0, st>>>(logits, ld, bias, E, K, norm, scale, ids, w);
    check_launch("router_topk");
}
void relu2(float* x, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    k_relu2<<<grid1(n), kT, 0, st>>>(x, n);
    check_launch("relu2");
}
void moe_combine(const float* parts, const float* w, int K, int L, int T, float* out, cudaStream_t st, const int32_t* rows) {
    if (T <= 0) return;
    k_combine<<<grid1((int64_t) T * L), kT, 0, st>>>(parts, w, K, L, T, out, rows);
    check_launch("moe_combine");
}
void route_sort(const int32_t* ids, int n_ent, int K, int n_expert, int32_t* cnt, int32_t* cur, int32_t* bounds, int32_t* dst,
                int32_t* src, int32_t* inv, int32_t* max_rows, cudaStream_t st) {
    cudaMemsetAsync(cnt, 0, (size_t) n_expert * sizeof(int32_t), st);
    if (n_ent > 0) k_route_count<<<grid1(n_ent), kT, 0, st>>>(ids, n_ent, cnt);
    k_route_scan<<<1, 32, 0, st>>>(cnt, n_expert, bounds, cur, max_rows);
    if (n_ent > 0) k_route_scatter<<<grid1(n_ent), kT, 0, st>>>(ids, n_ent, K, cur, dst, src, inv);
    check_launch("route_sort");
}
void merge_rows(float* dst, const float* src, const int32_t* mask, int n, int rows, cudaStream_t st) {
    if (rows <= 0) return;
    k_merge_rows<<<grid1((int64_t) n * rows), kT, 0, st>>>(dst, src, mask, n, rows);
    check_launch("merge_rows");
}
void rms_rows_ld(const float* x, const float* w, float* y, int ld_y, int n, int rows, float eps, cudaStream_t st) {
    if (rows <= 0) return;
    k_rms_ld<<<rows, kT, 0, st>>>(x, w, y, ld_y, n, eps);
    check_launch("rms_rows_ld");
}
void add(float* x, const float* y, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    k_add<<<grid1(n), kT, 0, st>>>(x, y, n);
    check_launch("add");
}
void gather_rows(const float* src, const int32_t* idx, int n, int rows, float* dst, cudaStream_t st) {
    if (rows <= 0) return;
    k_gather_rows<<<grid1((int64_t) n * rows), kT, 0, st>>>(src, idx, n, rows, dst);
    check_launch("gather_rows");
}
void to_f16(const float* x, __half* y, int64_t n, cudaStream_t st) {
    if (n <= 0) return;
    k_to_f16<<<grid1(n), kT, 0, st>>>(x, y, n);
    check_launch("to_f16");
}
void argmax_rows(const float* x, int n, int rows, int32_t* out, cudaStream_t st) {
    if (rows <= 0) return;
    k_argmax<<<rows, kT, 0, st>>>(x, n, out);
    check_launch("argmax_rows");
}

}  // namespace strata::nemotron::gpu
