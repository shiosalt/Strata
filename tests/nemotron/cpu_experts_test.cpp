// tests/nemotron/cpu_experts_test.cpp - nemotron's CPU experts (src/nemotron/cpu_experts.cpp) against the float computation on the
// same (dequantized) weights, for every up / down type pair the Unsloth UD-IQ3_XXS file uses, at its sizes (latent 1024, ff 2688).
// No GPU, no model: the weights are random floats quantized by ggml.  The difference is the activations' rounding (Q8_K / Q8_0).
#include "../../src/nemotron/cpu_experts.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace N = strata::nemotron;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

/// rows x cols random floats quantized to `type`; `deq` gets the dequantized floats.
std::vector<uint8_t> quantized(ggml_type type, int64_t rows, int64_t cols, uint32_t seed, std::vector<float>& deq) {
    std::mt19937 r(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f / std::sqrt((float) cols));
    std::vector<float> w((size_t) (rows * cols));
    for (float& x : w) x = nd(r);
    std::vector<float> imat((size_t) cols, 1.0f);
    const float* im = ggml_quantize_requires_imatrix(type) ? imat.data() : nullptr;
    std::vector<uint8_t> q(ggml_row_size(type, cols) * (size_t) rows);
    ggml_quantize_chunk(type, w.data(), q.data(), 0, rows, cols, im);
    deq.resize(w.size());
    ggml_get_type_traits(type)->to_float(q.data(), deq.data(), rows * cols);
    return q;
}

void pair_case(ggml_type ut, ggml_type dt, N::CpuExperts& pool) {
    const int64_t L = 1024, F = 2688;
    const int E = 3, T = 3;
    N::ExpertFmt f;
    std::string err;
    const std::string name = std::string(ggml_type_name(ut)) + " / " + ggml_type_name(dt);
    if (!N::expert_fmt((int) ut, (int) dt, L, F, f, err)) {
        check(false, name + ": " + err);
        return;
    }
    std::vector<std::vector<uint8_t>> up(E), dn(E);
    std::vector<std::vector<float>> upf(E), dnf(E);
    for (int e = 0; e < E; ++e) {
        up[(size_t) e] = quantized(ut, F, L, 10 + e, upf[(size_t) e]);
        dn[(size_t) e] = quantized(dt, L, F, 20 + e, dnf[(size_t) e]);
    }
    std::mt19937 r(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> lat((size_t) (T * L));
    for (float& x : lat) x = nd(r);
    // jobs: token t uses experts (t, t+1) mod E - 6 jobs, an expert used by two tokens
    std::vector<N::CpuJob> jobs;
    std::vector<std::vector<float>> out(6, std::vector<float>((size_t) L));
    std::vector<std::pair<int, int>> te;
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < 2; ++k) {
            const int e = (t + k) % E;
            N::CpuJob j;
            j.up = up[(size_t) e].data();
            j.down = dn[(size_t) e].data();
            j.tok = t;
            j.out = out[jobs.size()].data();
            jobs.push_back(j);
            te.emplace_back(t, e);
        }
    pool.run(f, lat.data(), T, jobs.data(), (int) jobs.size());
    float worst = 0.0f;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const int t = te[i].first, e = te[i].second;
        std::vector<float> h((size_t) F), want((size_t) L);
        for (int64_t o = 0; o < F; ++o) {
            double s = 0;
            for (int64_t c = 0; c < L; ++c) s += (double) upf[(size_t) e][(size_t) (o * L + c)] * lat[(size_t) (t * L + c)];
            h[(size_t) o] = s > 0 ? (float) (s * s) : 0.0f;
        }
        double num = 0, den = 0;
        for (int64_t o = 0; o < L; ++o) {
            double s = 0;
            for (int64_t c = 0; c < F; ++c) s += (double) dnf[(size_t) e][(size_t) (o * F + c)] * h[(size_t) c];
            want[(size_t) o] = (float) s;
            num += (out[i][(size_t) o] - s) * (out[i][(size_t) o] - s);
            den += s * s;
        }
        worst = std::max(worst, (float) std::sqrt(num / std::max(den, 1e-30)));
    }
    char buf[200];
    std::snprintf(buf, sizeof buf, "%s: 6 jobs (3 tokens, 3 experts) vs the float computation on the same weights (rel. L2 %.2e)",
                  name.c_str(), (double) worst);
    check(worst < 0.03f, buf);
}
/// this file's own kernel for weights of `wt` (AVX2) against ggml's, row by row at width n: the sums, and the speed on one thread
void own_kernel_case(ggml_type wt, int n) {
    const int rows = 300;
    std::vector<float> deq;
    const std::vector<uint8_t> w = quantized(wt, rows, n, 77, deq);
    std::mt19937 r(9);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> a((size_t) n);
    for (float& x : a) x = nd(r);
    const ggml_type_traits_cpu* tw = ggml_get_type_traits_cpu(wt);
    const ggml_type at = tw->vec_dot_type;
    std::vector<uint8_t> aq(ggml_row_size(at, n));
    ggml_get_type_traits_cpu(at)->from_float(a.data(), aq.data(), n);
    const size_t rb = ggml_row_size(wt, n);
    bool own = false;
    double worst = 0, scale = 0;
    for (int i = 0; i < rows; ++i) {
        float want = 0.0f;
        tw->vec_dot(n, &want, 0, w.data() + (size_t) i * rb, 0, aq.data(), 0, 1);
        const float got = N::row_dot((int) wt, n, w.data() + (size_t) i * rb, aq.data(), own);
        worst = std::max(worst, (double) std::fabs(got - want));
        scale = std::max(scale, (double) std::fabs(want));
    }
    char buf[200];
    std::snprintf(buf, sizeof buf, "%s x %s, %d rows of %d: %s kernel vs ggml's (max diff %.2e of %.2e)", ggml_type_name(wt), ggml_type_name(at),
                  rows, n, own ? "the AVX2" : "(no own kernel on this CPU: ggml's)", worst, scale);
    check(worst <= 1e-5 * std::max(1.0, scale), buf);
    auto time_ns = [&](bool mine) {   // one thread, the rows in the cache (printed, not checked)
        const auto t0 = std::chrono::steady_clock::now();
        float acc = 0.0f;
        bool o2 = false;
        for (int rep = 0; rep < 200; ++rep)
            for (int i = 0; i < rows; ++i) {
                float v = 0.0f;
                if (mine) v = N::row_dot((int) wt, n, w.data() + (size_t) i * rb, aq.data(), o2);
                else tw->vec_dot(n, &v, 0, w.data() + (size_t) i * rb, 0, aq.data(), 0, 1);
                acc += v;
            }
        const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / (200.0 * rows);
        return std::make_pair(ns, acc);
    };
    const auto g1 = time_ns(false), m1 = time_ns(true);
    std::printf("    one thread, %d values a row: ggml's %.1f ns a row, this file's %.1f ns a row (sums %g / %g)\n", n, g1.first, m1.first,
                (double) g1.second, (double) m1.second);
}
}  // namespace

int main() {
    std::printf("nemotron_cpu_experts_test\n");
    N::CpuExperts pool(0);
    std::printf("  %d threads\n", pool.threads());
    // the pairs of the UD-IQ3_XXS file (up / down): IQ3_XXS / IQ4_NL, IQ4_XS / IQ4_NL, IQ2_XS / Q2_0, IQ2_S / Q2_0, Q8_0 / Q8_0
    pair_case(GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, pool);
    pair_case(GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, pool);
    pair_case(GGML_TYPE_IQ2_XS, GGML_TYPE_Q2_0, pool);
    pair_case(GGML_TYPE_IQ2_S, GGML_TYPE_Q2_0, pool);
    pair_case(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, pool);
    own_kernel_case(GGML_TYPE_Q2_0, 2688);      // (the down width)
    {
        N::ExpertFmt f;
        std::string err;
        check(!N::expert_fmt((int) GGML_TYPE_IQ3_XXS, (int) GGML_TYPE_IQ4_XS, 1024, 2688, f, err),
              "IQ4_XS as the down type at ff 2688 is refused (not whole 256 blocks): " + err);
    }
    std::printf(g_fail ? "FAILED: %d\n" : "all ok\n", g_fail);
    return g_fail ? 1 : 0;
}
