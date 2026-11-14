// src/nemotron/grouped_mmvq.cuh - one launch for many quantized matrix x vector products of the same type and shape (a MoE layer's
// expert entries in decode): entry i multiplies the matrix at ent[i].w + w_off by the q8_1 row x + (x_by_row ? ent[i].row : ent[i].xrow)
// * x_stride and writes y + ent[i].row * y_stride.  ggml-cuda's mul_mat_vec_q arithmetic for one column (vecdotq.cuh); the input is
// Strata's q8_1 (the same blocks as ggml's block_q8_1).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strata::nemotron::gpu {

struct GroupedEntry {
    const void* w;   ///< the matrix (n_out rows of n_in values, ggml rows), before w_off
    int xrow;        ///< the input row (the token) for x_by_row == false
    int row;         ///< the output row (and the input row for x_by_row == true)
};

bool grouped_mmvq_supported(int ggml_type);
void grouped_mmvq(int ggml_type, const GroupedEntry* ent, int n_ent, size_t w_off, const void* x, size_t x_stride, bool x_by_row, float* y,
                  size_t y_stride, int n_in, int n_out, cudaStream_t stream);

}  // namespace strata::nemotron::gpu
