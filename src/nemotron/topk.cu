// src/nemotron/topk.cu - the K largest logits of each row, picked on the GPU, so a sampled request copies K (id, logit) pairs a row
// instead of the whole vocabulary (131,072 floats, 512 KiB) and sorts nothing on the host.
//
// One block per row: a radix select on the floats' order-preserving 32-bit keys (4 passes of 8 bits) finds the K-th largest key,
// then the keys above it and enough of the equal ones are gathered and sorted (largest first, the lowest id first on ties).
#include "kernels.cuh"

namespace strata::nemotron::gpu {
namespace {

constexpr int kThreads = 1024;

__device__ __forceinline__ uint32_t key_of(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // larger float -> larger key
}
__device__ __forceinline__ float float_of(uint32_t k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

__global__ void topk_rows_kernel(const float* __restrict__ x, int n, int K, int P, int32_t* __restrict__ ids, float* __restrict__ vals) {
    extern __shared__ unsigned long long sbuf[];   // P sort entries: key << 32 | ~id (descending order = key desc, id asc)
    __shared__ unsigned int hist[256];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int s_left, s_gt, s_eq;
    const float* row = x + (size_t) blockIdx.x * n;
    const int tid = threadIdx.x;
    if (tid == 0) {
        s_prefix = 0;
        s_mask = 0;
        s_left = K;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += blockDim.x) hist[b] = 0;
        __syncthreads();
        const uint32_t prefix = s_prefix, mask = s_mask;
        for (int i = tid; i < n; i += blockDim.x) {
            const uint32_t k = key_of(row[i]);
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            int c = 0, left = s_left, sel = 0;
            for (int b = 255; b >= 0; --b) {
                if (c + (int) hist[b] >= left) {
                    sel = b;
                    break;
                }
                c += (int) hist[b];
            }
            s_prefix = prefix | ((uint32_t) sel << shift);
            s_mask = mask | (255u << shift);
            s_left = left - c;   // how many keys equal to the prefix so far are still needed
        }
        __syncthreads();
    }
    const uint32_t thr = s_prefix;   // the K-th largest key; K - s_left keys are above it
    const int need_eq = s_left, n_gt = K - s_left;
    if (tid == 0) {
        s_gt = 0;
        s_eq = 0;
    }
    for (int i = tid; i < P; i += blockDim.x) sbuf[i] = 0ull;   // (padding sorts last)
    __syncthreads();
    for (int i = tid; i < n; i += blockDim.x) {
        const uint32_t k = key_of(row[i]);
        if (k > thr) {
            const int p = atomicAdd(&s_gt, 1);
            if (p < n_gt) sbuf[p] = ((unsigned long long) k << 32) | (uint32_t) ~(uint32_t) i;
        } else if (k == thr) {
            const int p = atomicAdd(&s_eq, 1);
            if (p < need_eq) sbuf[n_gt + p] = ((unsigned long long) k << 32) | (uint32_t) ~(uint32_t) i;
        }
    }
    __syncthreads();
    // the equal keys came in the atomics' order: the lowest ids among them are not guaranteed, only how many (ties are rare in logits)
    for (int size = 2; size <= P; size <<= 1)   // bitonic sort, descending
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = tid; i < P; i += blockDim.x) {
                const int j = i ^ stride;
                if (j > i) {
                    const bool desc = (i & size) == 0;
                    const unsigned long long a = sbuf[i], b = sbuf[j];
                    if (desc ? a < b : a > b) {
                        sbuf[i] = b;
                        sbuf[j] = a;
                    }
                }
            }
            __syncthreads();
        }
    for (int i = tid; i < K; i += blockDim.x) {
        const unsigned long long e = sbuf[i];
        ids[(size_t) blockIdx.x * K + i] = (int32_t) ~(uint32_t) (e & 0xffffffffull);
        vals[(size_t) blockIdx.x * K + i] = float_of((uint32_t) (e >> 32));
    }
}

}  // namespace

void topk_rows(const float* x, int n, int rows, int K, int32_t* ids, float* vals, cudaStream_t stream) {
    if (rows <= 0 || K <= 0) return;
    int P = 1;
    while (P < K) P <<= 1;
    topk_rows_kernel<<<rows, kThreads, (size_t) P * sizeof(unsigned long long), stream>>>(x, n, K, P, ids, vals);
}

}  // namespace strata::nemotron::gpu
