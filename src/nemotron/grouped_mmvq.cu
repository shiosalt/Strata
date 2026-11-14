// src/nemotron/grouped_mmvq.cu - see grouped_mmvq.cuh.  ggml-cuda's mul_mat_vec_q for one column (its vec_dot_*_q8_1 functions and
// the same reduction: 4 warps a row, the warps' sums through shared memory, then a warp reduction), with the matrix, the input row and
// the output row of each block row taken from a per-entry table, so all of a layer's expert entries are one launch.
#include "grouped_mmvq.cuh"

#include "common.cuh"
#include "vecdotq.cuh"

#include <stdexcept>
#include <string>

namespace strata::nemotron::gpu {

namespace {
constexpr int kWarps = 4;
typedef float (*vec_dot_q_cuda_t)(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx, const int& iqs);

template <ggml_type type, int vdr, vec_dot_q_cuda_t vec_dot>
__launch_bounds__(kWarps * WARP_SIZE, 1)
__global__ void grouped_mmvq_kernel(const GroupedEntry* __restrict__ ent, size_t w_off, const char* __restrict__ x, size_t x_stride,
                                    int x_by_row, float* __restrict__ y, size_t y_stride, int n_in) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int bs = ggml_cuda_type_traits<type>::bs;
    constexpr int blocks_per_iter = vdr * kWarps * WARP_SIZE / qi;
    const GroupedEntry e = ent[blockIdx.y];
    const int row = blockIdx.x;
    const int blocks_per_row = n_in / qk;
    const char* vx = (const char*) e.w + w_off + (size_t) row * blocks_per_row * bs;
    const block_q8_1* yb = (const block_q8_1*) (x + (size_t) (x_by_row ? e.row : e.xrow) * x_stride);
    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    float tmp = 0.0f;
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (tid % (qi / vdr));
        tmp += vec_dot(vx, &yb[kby], kbx, kqs);
    }
    __shared__ float shared[kWarps - 1][WARP_SIZE];
    if (threadIdx.y > 0) shared[threadIdx.y - 1][threadIdx.x] = tmp;
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int w = 0; w < kWarps - 1; ++w) tmp += shared[w][threadIdx.x];
    tmp = warp_reduce_sum(tmp);
    if (threadIdx.x == 0) y[(size_t) e.row * y_stride + row] = tmp;
}

template <ggml_type type, int vdr, vec_dot_q_cuda_t vec_dot>
void launch(const GroupedEntry* ent, int n_ent, size_t w_off, const void* x, size_t x_stride, bool x_by_row, float* y, size_t y_stride,
            int n_in, int n_out, cudaStream_t st) {
    const dim3 grid((unsigned) n_out, (unsigned) n_ent), block(WARP_SIZE, kWarps);
    grouped_mmvq_kernel<type, vdr, vec_dot><<<grid, block, 0, st>>>(ent, w_off, (const char*) x, x_stride, x_by_row ? 1 : 0, y, y_stride, n_in);
}
}  // namespace

bool grouped_mmvq_supported(int t) {
    switch (t) {
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q2_0: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS: return true;
        default: return false;
    }
}

void grouped_mmvq(int type, const GroupedEntry* ent, int n_ent, size_t w_off, const void* x, size_t x_stride, bool x_by_row, float* y,
                  size_t y_stride, int n_in, int n_out, cudaStream_t st) {
    if (n_ent <= 0) return;
    switch (type) {
        case GGML_TYPE_Q8_0: launch<GGML_TYPE_Q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(ent, n_ent, w_off, x, x_stride, x_by_row, y, y_stride, n_in, n_out, st); break;
        case GGML_TYPE_Q2_0: launch<GGML_TYPE_Q2_0, VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(ent, n_ent, w_off, x, x_stride, x_by_row, y, y_stride, n_in, n_out, st); break;
        case GGML_TYPE_IQ2_XS: launch<GGML_TYPE_IQ2_XS, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1>(ent, n_ent, w_off, x, x_stride, x_by_row, y, y_stride, n_in, n_out, st); break;
        case GGML_TYPE_IQ2_S: launch<GGML_TYPE_IQ2_S, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1>(ent, n_ent, w_off, x, x_stride, x_by_row, y, y_stride, n_in, n_out, st); break;
        case GGML_TYPE_IQ3_XXS: launch<GGML_TYPE_IQ3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1>(ent, n_ent, w_off, x, x_stride, x_by_row, y, y_stride, n_in, n_out, st); break;
        case GGML_TYPE_IQ4_NL: launch<GGML_TYPE_IQ4_NL, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(ent, n_ent, w_off, x, x_stride, x_by_row, y, y_stride, n_in, n_out, st); break;
        case GGML_TYPE_IQ4_XS: launch<GGML_TYPE_IQ4_XS, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(ent, n_ent, w_off, x, x_stride, x_by_row, y, y_stride, n_in, n_out, st); break;
        default: throw std::invalid_argument("grouped_mmvq: type " + std::to_string(type) + " is not supported");
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("grouped_mmvq: ") + cudaGetErrorString(e));
}

}  // namespace strata::nemotron::gpu
