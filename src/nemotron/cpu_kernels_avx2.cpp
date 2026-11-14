// src/nemotron/cpu_kernels_avx2.cpp - the CPU experts' own dot products for weight types whose ggml-cpu kernel is its plain C loop on
// x86.  This file alone is compiled for AVX2 + FMA + F16C (/arch:AVX2, -mavx2 -mfma -mf16c); cpu_experts.cpp calls it only after
// checking the CPU (cpuid), else it uses ggml's.  (Measured 2026-11-13: Q2_0 4.7 GB/s with ggml's C loop, 17 GB/s with this - and
// with no /O2, which the local build had for its C++ by mistake, this kernel ran at 0.5-4 GB/s.)
#include "cpu_kernels_avx2.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

#include <cstring>

namespace strata::nemotron {

namespace {
/// the fp16 at p (unaligned) as a float (F16C)
inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
}
}  // namespace

/// Q2_0 (64 weights: fp16 d, 16 bytes of 2-bit codes, code c -> c - 1) against Q8_0 activations (two blocks a Q2_0 block), as ggml's
/// ggml_vec_dot_q2_0_q8_0 computes it.  Each half block (32 weights): the four 2-bit fields of each code byte come out of one variable
/// shift of the bytes as 32-bit words, so byte 4p + i of a 128-bit lane is weight 4i + p; the activations are put in that order by a
/// shuffle.  The integer dot of each Q8_0 block is exact; only the float accumulation's order differs from the C loop.
void vec_dot_q2_0_q8_0_avx2(int n, float* s, size_t, const void* vx, size_t, const void* vy, size_t, int) {
    const uint8_t* x = (const uint8_t*) vx;   // blocks of 18 bytes: fp16 d, 16 code bytes
    const uint8_t* y = (const uint8_t*) vy;   // blocks of 34 bytes: fp16 d, 32 int8
    const int nb = n / 64;
    const __m256i shifts = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    const __m256i m3 = _mm256_set1_epi32(0x03030303);
    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i perm = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                          0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m256i w0 = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1), w1 = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();   // (two chains: the FMAs of one block do not wait on each other)
    for (int i = 0; i < nb; ++i) {
        const uint8_t* xb = x + (size_t) i * 18;
        const uint8_t* y0 = y + (size_t) (2 * i) * 34;
        const uint8_t* y1 = y0 + 34;
        const float d0 = h2f(xb);
        const __m256 s0 = _mm256_set1_ps(d0 * h2f(y0));
        const __m256 s1 = _mm256_set1_ps(d0 * h2f(y1));
        // the 16 code bytes as 4 words, each spread over the 4 dwords of its lane: words 0, 1 (first half), 2, 3 (second half)
        const __m256i codes = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*) (xb + 2)));
        const __m256i c0 = _mm256_permutevar8x32_epi32(codes, w0), c1 = _mm256_permutevar8x32_epi32(codes, w1);
        const __m256i x0 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srlv_epi32(c0, shifts), m3), one8);   // -1 .. 2
        const __m256i x1 = _mm256_sub_epi8(_mm256_and_si256(_mm256_srlv_epi32(c1, shifts), m3), one8);
        const __m256i q0 = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i*) (y0 + 2)), perm);
        const __m256i q1 = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i*) (y1 + 2)), perm);
        const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(x0, x0), _mm256_sign_epi8(q0, x0)), ones16);
        const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(x1, x1), _mm256_sign_epi8(q1, x1)), ones16);
        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), s0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), s1, acc1);
    }
    const __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_movehdup_ps(h));
    *s = _mm_cvtss_f32(h);
}

}  // namespace strata::nemotron
#endif
