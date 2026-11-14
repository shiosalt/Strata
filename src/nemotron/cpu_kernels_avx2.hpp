// src/nemotron/cpu_kernels_avx2.hpp - the CPU experts' own AVX2 dot products (cpu_kernels_avx2.cpp, compiled for AVX2 + FMA + F16C).
// Call them only when the CPU has those (cpu_experts.cpp checks).  ggml's vec_dot signature.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::nemotron {

#if defined(__x86_64__) || defined(_M_X64)
void vec_dot_q2_0_q8_0_avx2(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);
#endif

}  // namespace strata::nemotron
