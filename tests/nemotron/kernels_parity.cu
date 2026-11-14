// tests/nemotron/kernels_parity.cu - the nemotron CUDA kernels (src/nemotron/kernels.cu) against the fp32 CPU reference
// (include/strata/nemotron/reference.hpp), on synthetic weights: the tiny geometry, and the real model's Mamba2 / attention sizes
// with a small hidden width (tens of MB of GPU memory, a few seconds).  Exit 77 (skipped) without a GPU with 256 MB free.
#include "../../src/nemotron/kernels.cuh"
#include "strata/nemotron/reference.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace R = strata::nemotron::ref;
namespace G = strata::nemotron::gpu;
using strata::nemotron::BlockKind;
using strata::nemotron::Geometry;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-100s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
#define CK(x)                                                                                                    \
    do {                                                                                                         \
        cudaError_t e_ = (x);                                                                                    \
        if (e_ != cudaSuccess) {                                                                                 \
            std::fprintf(stderr, "%s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__);    \
            std::exit(1);                                                                                        \
        }                                                                                                        \
    } while (0)

template <class T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    CK(cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T)));
    if (!h.empty()) CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}
std::vector<float> rnd(size_t n, uint32_t seed, float a = 1.0f) {
    std::mt19937 r(seed);
    std::uniform_real_distribution<float> u(-a, a);
    std::vector<float> v(n);
    for (float& x : v) x = u(r);
    return v;
}
/// max |a - b| / max(1, max |b|)
float rel(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        d = std::max(d, std::fabs(a[i] - b[i]));
        m = std::max(m, std::fabs(b[i]));
    }
    return d / std::max(1.0f, m);
}
std::string fmt(const char* what, float v) {
    char b[200];
    std::snprintf(b, sizeof b, "%s (rel %.2e)", what, (double) v);
    return b;
}

/// The Mamba2 block `l` of `m` on the GPU against ref::mamba2, over T tokens in two pieces (T1, T - T1), then a verify window.
void mamba_case(const R::Model& m, int64_t l, int T, int T1, const char* name) {
    const Geometry& g = m.g;
    const R::Block& b = m.blocks[(size_t) l];
    const int N = (int) g.n_embd, DI = (int) g.ssm_d_inner, W_IN = (int) g.ssm_in_dim(), C = (int) g.ssm_conv_dim(), K = (int) g.ssm_d_conv;
    const int H = (int) g.ssm_n_head, P = (int) g.ssm_head_dim(), S = (int) g.ssm_d_state, GR = (int) g.ssm_n_group;
    const std::vector<float> h = rnd((size_t) T * N, 100 + (uint32_t) l);
    std::vector<float> proj((size_t) T * W_IN), want((size_t) T * N);
    R::State s = R::new_state(m);
    for (int t = 0; t < T; ++t) {
        R::matvec(b.ssm_in.data(), h.data() + (size_t) t * N, N, W_IN, proj.data() + (size_t) t * W_IN);
        R::mamba2(m, l, s, h.data() + (size_t) t * N, want.data() + (size_t) t * N);
    }
    float* dproj = up(proj);
    float* dconv = up(std::vector<float>((size_t) (K - 1) * C, 0.0f));
    float* dssm = up(std::vector<float>((size_t) DI * S, 0.0f));
    float *dw = up(b.conv_w), *db = up(b.conv_b), *ddt = up(b.dt_b), *dA = up(b.A), *dD = up(b.D), *dn = up(b.ssm_norm);
    float* dxbc = up(std::vector<float>((size_t) T * C));
    float* dy = up(std::vector<float>((size_t) T * DI));
    auto run = [&](int t0, int n, bool write) {
        G::mamba_conv(dproj + (size_t) t0 * W_IN, W_IN, DI, dconv, dw, db, dxbc, C, K, n, write, nullptr);
        G::mamba_scan(dxbc, C, dproj + (size_t) t0 * W_IN, W_IN, 0, DI + C, ddt, dA, dD, dssm, dy + (size_t) t0 * DI, H, P, S, GR, n, write,
                      nullptr);
    };
    run(0, T1, true);
    run(T1, T - T1, true);
    G::group_rms_rows(dy, dn, DI, DI / GR, T, g.rms_eps, nullptr);
    const std::vector<float> y = down(dy, (size_t) T * DI);
    std::vector<float> got((size_t) T * N);
    for (int t = 0; t < T; ++t) R::matvec(b.ssm_out.data(), y.data() + (size_t) t * DI, DI, N, got.data() + (size_t) t * N);
    const float e = rel(got, want);
    check(e < 2e-4f, fmt((std::string(name) + ": conv + scan + grouped norm vs the reference, in two pieces").c_str(), e));
    // the final states: the reference's after T tokens
    const std::vector<float> st = down(dssm, (size_t) DI * S), cs = down(dconv, (size_t) (K - 1) * C);
    check(rel(st, s.ssm[(size_t) l]) < 2e-4f && rel(cs, s.conv[(size_t) l]) == 0.0f,
          fmt((std::string(name) + ": the SSM state and the conv window after the tokens").c_str(), rel(st, s.ssm[(size_t) l])));
    // a verify window (write = false) leaves both states as they were
    run(0, std::min(T, 4), false);
    check(down(dssm, (size_t) DI * S) == st && down(dconv, (size_t) (K - 1) * C) == cs,
          std::string(name) + ": a verify window does not move the states");
    for (float* p : {dproj, dconv, dssm, dw, db, ddt, dA, dD, dn, dxbc, dy}) CK(cudaFree(p));
}

/// Attention block `l`: q / k / v from the reference's matrices on the host, the cache and the kernel on the GPU, Wo on the host.
void attention_case(const R::Model& m, int64_t l, int T, const char* name) {
    const Geometry& g = m.g;
    const R::Block& b = m.blocks[(size_t) l];
    const int N = (int) g.n_embd, Hq = (int) g.n_head, Hk = (int) g.n_head_kv, D = (int) g.head_dim;
    const std::vector<float> h = rnd((size_t) T * N, 300 + (uint32_t) l);
    std::vector<float> q((size_t) T * Hq * D), k((size_t) T * Hk * D), v((size_t) T * Hk * D), want((size_t) T * N);
    std::vector<float> kc, vc;
    for (int t = 0; t < T; ++t) {
        R::matvec(b.wq.data(), h.data() + (size_t) t * N, N, Hq * D, q.data() + (size_t) t * Hq * D);
        R::matvec(b.wk.data(), h.data() + (size_t) t * N, N, Hk * D, k.data() + (size_t) t * Hk * D);
        R::matvec(b.wv.data(), h.data() + (size_t) t * N, N, Hk * D, v.data() + (size_t) t * Hk * D);
        R::attention(m, l, kc, vc, t, h.data() + (size_t) t * N, want.data() + (size_t) t * N);
    }
    const int ctx = T + 8;
    float *dq = up(q), *dk = up(k), *dv = up(v);
    __half *dkc = nullptr, *dvc = nullptr;
    CK(cudaMalloc(&dkc, (size_t) Hk * ctx * D * 2));
    CK(cudaMalloc(&dvc, (size_t) Hk * ctx * D * 2));
    float* dout = up(std::vector<float>((size_t) T * Hq * D));
    float* scr = up(std::vector<float>(G::attn_scratch_floats(T, Hq, D, 5)));
    const float scale = 1.0f / std::sqrt((float) D);
    auto finish = [&](const char* how, float tol = 3e-3f) {
        const std::vector<float> o = down(dout, (size_t) T * Hq * D);
        std::vector<float> got((size_t) T * N);
        for (int t = 0; t < T; ++t) R::matvec(b.wo.data(), o.data() + (size_t) t * Hq * D, Hq * D, N, got.data() + (size_t) t * N);
        const float e = rel(got, want);
        check(e < tol, fmt((std::string(name) + ": " + how).c_str(), e));   // the cache is fp16
    };
    // the prompt form: every row at once, one range each
    G::kv_store(dk, dv, dkc, dvc, ctx, 0, T, Hk, D, nullptr);
    G::attention(dq, dkc, dvc, ctx, 0, T, Hq, Hk, D, scale, 1, nullptr, dout, nullptr);
    finish("prompt form (all rows, one range)");
    if (G::attention_tc_supported(Hq, Hk, D)) {   // the prompt path on tensor cores: fp16 Q and P too
        CK(cudaMemset(dout, 0, (size_t) T * Hq * D * 4));
        G::attention_tc(dq, dkc, dvc, ctx, 0, T, Hq, Hk, D, scale, dout, nullptr);
        finish("tensor cores, all rows", 1e-2f);
        CK(cudaMemset(dout, 0, (size_t) T * Hq * D * 4));
        const int pa = T / 3;   // a chunk after a prefix (rows pa .. T - 1 at positions pa ..)
        G::attention_tc(dq, dkc, dvc, ctx, 0, pa, Hq, Hk, D, scale, dout, nullptr);
        G::attention_tc(dq + (size_t) pa * Hq * D, dkc, dvc, ctx, pa, T - pa, Hq, Hk, D, scale, dout + (size_t) pa * Hq * D, nullptr);
        finish("tensor cores, a chunk after a prefix", 1e-2f);
    }
    // the decode form: one row at a time on a growing cache, the cells cut into 1..5 ranges and merged
    CK(cudaMemset(dout, 0, (size_t) T * Hq * D * 4));
    for (int t = 0; t < T; ++t) {
        G::kv_store(dk + (size_t) t * Hk * D, dv + (size_t) t * Hk * D, dkc, dvc, ctx, t, 1, Hk, D, nullptr);
        G::attention(dq + (size_t) t * Hq * D, dkc, dvc, ctx, t, 1, Hq, Hk, D, scale, 1 + t % 5, scr, dout + (size_t) t * Hq * D,
                     nullptr);
    }
    finish("decode form (one row at a time, split into 1..5 ranges)");
    // a window of 4 rows after a prefix, split in 3
    CK(cudaMemset(dout, 0, (size_t) T * Hq * D * 4));
    const int p0 = T - 4;
    G::attention(dq, dkc, dvc, ctx, 0, p0, Hq, Hk, D, scale, 1, nullptr, dout, nullptr);
    G::attention(dq + (size_t) p0 * Hq * D, dkc, dvc, ctx, p0, 4, Hq, Hk, D, scale, 3, scr, dout + (size_t) p0 * Hq * D, nullptr);
    finish("a 4-row window after the prefix (split in 3)");
    for (void* p : {(void*) dq, (void*) dk, (void*) dv, (void*) dkc, (void*) dvc, (void*) dout, (void*) scr}) CK(cudaFree(p));
}

void router_case(int E, int K, int T, float scale) {
    std::vector<float> lg = rnd((size_t) T * E, 7, 3.0f), bias = rnd((size_t) E, 8, 0.05f);
    lg[3] = lg[5];   // a tie between ids 3 and 5 of token 0
    int32_t* dids = nullptr;
    float* dw = nullptr;
    CK(cudaMalloc(&dids, (size_t) T * K * 4));
    CK(cudaMalloc(&dw, (size_t) T * K * 4));
    float *dl = up(lg), *db = up(bias);
    G::router_topk(dl, E, db, E, K, T, true, scale, dids, dw, nullptr);
    const std::vector<int32_t> ids = down(dids, (size_t) T * K);
    const std::vector<float> w = down(dw, (size_t) T * K);
    bool ok = true;
    float werr = 0;
    for (int t = 0; t < T; ++t) {
        std::vector<int> idx(E);
        std::vector<float> p(E), sel(E);
        for (int e = 0; e < E; ++e) {
            p[e] = 1.0f / (1.0f + std::exp(-lg[(size_t) t * E + e]));
            sel[e] = p[e] + bias[e];
            idx[e] = e;
        }
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return sel[a] != sel[b] ? sel[a] > sel[b] : a < b; });
        float sum = 0;
        for (int k = 0; k < K; ++k) sum += p[idx[k]];
        for (int k = 0; k < K; ++k) {
            ok &= ids[(size_t) t * K + k] == idx[k];
            werr = std::max(werr, std::fabs(w[(size_t) t * K + k] - p[idx[k]] / sum * scale));
        }
    }
    char buf[160];
    std::snprintf(buf, sizeof buf, "router: %d experts top-%d, %d tokens: the picks (lowest id on ties) and weights (err %.1e)", E, K, T,
                  (double) werr);
    check(ok && werr < 1e-5f, buf);
    for (void* q : {(void*) dids, (void*) dw, (void*) dl, (void*) db}) CK(cudaFree(q));
}
}  // namespace

int main() {
    std::printf("nemotron_kernels_parity\n");
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("  no CUDA device: skipped\n");
        return 77;
    }
    // the card with the most free memory (other jobs may hold the rest)
    int best = 0;
    size_t best_free = 0;
    for (int d = 0; d < n_dev; ++d) {
        CK(cudaSetDevice(d));
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        if (fr > best_free) {
            best_free = fr;
            best = d;
        }
    }
    if (best_free < ((size_t) 256 << 20)) {
        std::printf("  no GPU has 256 MB free (%zu MB at most): skipped\n", best_free >> 20);
        return 77;
    }
    CK(cudaSetDevice(best));
    std::printf("  GPU %d, %zu MB free\n", best, best_free >> 20);

    // ---- the tiny model's blocks
    const Geometry tg = R::tiny_geometry();
    const R::Model tm = R::random_model(tg, 99);
    mamba_case(tm, 0, 19, 7, "tiny Mamba2 (H 4, P 8, N 8, G 2)");
    attention_case(tm, 3, 21, "tiny attention (4:2 x 16)");

    // ---- the real model's Mamba2 and attention sizes, hidden 256
    Geometry rg = tg;
    rg.n_embd = 256;
    rg.ssm_d_inner = 8192;
    rg.ssm_n_head = 128;
    rg.ssm_d_state = 128;
    rg.ssm_n_group = 8;
    rg.n_head = 32;
    rg.n_head_kv = 2;
    rg.head_dim = 128;
    rg.n_vocab = 16;
    rg.kind = {BlockKind::Mamba, BlockKind::Attention};
    rg.n_layers = 2;
    rg.n_mtp = 0;
    const R::Model rm = R::random_model(rg, 5);
    check(G::mamba_scan_supported(128, 64, 128, 8), "the real Mamba2 sizes are supported (128 heads x 64, state 128, 8 groups)");
    mamba_case(rm, 0, 12, 5, "real-size Mamba2 (H 128, P 64, N 128, G 8)");
    attention_case(rm, 1, 40, "real-size attention (32:2 x 128)");
    attention_case(rm, 1, 150, "real-size attention, 150 rows (several blocks and tiles)");

    // ---- the router at the real sizes and the tiny ones
    router_case(512, 22, 3, 5.0f);
    router_case(8, 3, 2, 2.5f);

    // ---- the sampled requests' top-k (topk.cu) at the vocabulary's size, against a sort on the host
    for (const int K : {1, 40, 64, 256}) {
        const int n = 131072, rows = 7;
        std::vector<float> x((size_t) n * rows);   // distinct values (a shuffled ramp), negative and positive
        std::mt19937 rg2(300 + (uint32_t) K);
        for (int r = 0; r < rows; ++r) {
            std::vector<int> perm(n);
            for (int i = 0; i < n; ++i) perm[(size_t) i] = i;
            std::shuffle(perm.begin(), perm.end(), rg2);
            for (int i = 0; i < n; ++i) x[(size_t) r * n + i] = (float) (perm[(size_t) i] - n / 2) * 0.001f;
            x[(size_t) r * n + (size_t) (r * 977) % n] = -1e30f;   // (a huge negative orders right too)
        }
        float* dx = up(x);
        int32_t* di = up(std::vector<int32_t>((size_t) rows * K));
        float* dv = up(std::vector<float>((size_t) rows * K));
        G::topk_rows(dx, n, rows, K, di, dv, nullptr);
        const std::vector<int32_t> gi = down(di, (size_t) rows * K);
        const std::vector<float> gv = down(dv, (size_t) rows * K);
        bool ok = true;
        for (int r = 0; r < rows; ++r) {
            std::vector<int> idx(n);
            for (int i = 0; i < n; ++i) idx[(size_t) i] = i;
            const float* xr = x.data() + (size_t) r * n;
            std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return xr[a] > xr[b] || (xr[a] == xr[b] && a < b); });
            for (int k = 0; k < K; ++k)
                ok &= gi[(size_t) r * K + k] == idx[(size_t) k] && gv[(size_t) r * K + k] == xr[idx[(size_t) k]];
        }
        char buf[160];
        std::snprintf(buf, sizeof buf, "topk_rows: %d rows of %d, k %d: the same ids and values, largest first", rows, n, K);
        check(ok, buf);
        CK(cudaFree(dx));
        CK(cudaFree(di));
        CK(cudaFree(dv));
    }
    {   // ties at the k-th value: the count and the values are fixed (which of the equal ids is the atomics' order)
        const int n = 5000, K = 10;
        std::vector<float> x((size_t) n, 0.0f);
        for (int i = 0; i < 6; ++i) x[(size_t) (i * 101)] = 5.0f - (float) i;   // 5 4 3 2 1 0... and 4994 more zeros
        float* dx = up(x);
        int32_t* di = up(std::vector<int32_t>((size_t) K));
        float* dv = up(std::vector<float>((size_t) K));
        G::topk_rows(dx, n, 1, K, di, dv, nullptr);
        const std::vector<int32_t> gi = down(di, (size_t) K);
        const std::vector<float> gv = down(dv, (size_t) K);
        bool ok = true;
        for (int k = 0; k < 5; ++k) ok &= gi[(size_t) k] == k * 101 && gv[(size_t) k] == 5.0f - (float) k;
        std::vector<int32_t> rest(gi.begin() + 5, gi.end());
        std::sort(rest.begin(), rest.end());
        ok &= std::adjacent_find(rest.begin(), rest.end()) == rest.end();
        for (int k = 5; k < K; ++k) ok &= gv[(size_t) k] == 0.0f && x[(size_t) gi[(size_t) k]] == 0.0f;
        check(ok, "topk_rows: ties at the k-th value give k distinct ids of the right values");
        CK(cudaFree(dx));
        CK(cudaFree(di));
        CK(cudaFree(dv));
    }

    // ---- norms, relu^2, combine
    {
        const int n = 300, rows = 3;
        std::vector<float> x = rnd((size_t) n * rows, 1), w = rnd(n, 2), y((size_t) n * rows), ln((size_t) n * rows);
        for (int r = 0; r < rows; ++r) {
            R::rms_norm(x.data() + r * n, w.data(), n, 1e-5f, y.data() + r * n);
            R::layer_norm(x.data() + r * n, w.data(), n, 1e-5f, ln.data() + r * n);
        }
        float *dx = up(x), *dw = up(w), *dy = up(std::vector<float>((size_t) n * rows));
        G::rms_rows(dx, dw, dy, n, rows, 1e-5f, nullptr);
        check(rel(down(dy, (size_t) n * rows), y) < 1e-5f, "rms_rows vs the reference");
        G::layer_norm_rows(dx, dw, dy, n, rows, 1e-5f, nullptr);
        check(rel(down(dy, (size_t) n * rows), ln) < 1e-5f, "layer_norm_rows vs the reference");
        G::relu2(dx, (int64_t) n * rows, nullptr);
        const std::vector<float> r2 = down(dx, (size_t) n * rows);
        bool ok = true;
        for (size_t i = 0; i < x.size(); ++i) ok &= r2[i] == (x[i] > 0 ? x[i] * x[i] : 0.0f);
        check(ok, "relu2");
        for (float* p : {dx, dw, dy}) CK(cudaFree(p));
    }
    std::printf(g_fail ? "FAILED: %d\n" : "all ok\n", g_fail);
    return g_fail ? 1 : 0;
}
