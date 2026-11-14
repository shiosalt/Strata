// tests/nemotron/reference_test.cpp - the fp32 CPU reference of nemotron_h_moe against independent formulations (no model, no GPU).
//
//   * Mamba2: the token-by-token recurrence of ref::mamba2 against the whole-sequence quadratic ("dual", SSD) form, with the
//     convolution done over the zero-padded sequence instead of a rolling window;
//   * attention: one token at a time with a growing K / V cache against causal masked attention over the whole sequence, and the
//     GQA head mapping;
//   * MoE: the selection bias picks without weighing, the weights are normalised and scaled, relu^2 and the latent projections;
//   * the model: a state carried across a split equals one run, the MTP chain is deterministic, every logit is finite.
#include "strata/nemotron/reference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace R = strata::nemotron::ref;
using strata::nemotron::BlockKind;
using strata::nemotron::Geometry;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-92s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float d = a.size() == b.size() ? 0.0f : INFINITY;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}
float max_abs(const std::vector<float>& a) {
    float d = 0.0f;
    for (float x : a) d = std::max(d, std::fabs(x));
    return d;
}
bool finite(const std::vector<float>& a) {
    for (float x : a)
        if (!std::isfinite(x)) return false;
    return true;
}
float sigm(float x) { return 1.0f / (1.0f + std::exp(-x)); }
std::vector<float> inputs(int64_t T, int64_t N, uint32_t seed) {
    std::vector<float> h((size_t) (T * N));
    uint32_t s = seed;
    for (float& x : h) {
        s = s * 1664525u + 1013904223u;
        x = ((float) (s >> 8) / 16777216.0f) * 2.0f - 1.0f;
    }
    return h;
}

/// Mamba2 of block `l` over a whole sequence `h` [T][N], written without the recurrence: the conv over the zero-padded sequence,
/// then y_t = sum_{s <= t} exp(sum_{r = s+1..t} dt'_r A) (C_t . B_s) x_s dt'_s + D x_t.
std::vector<float> mamba2_dual(const R::Model& m, int64_t l, const std::vector<float>& h, int64_t T) {
    const Geometry& g = m.g;
    const R::Block& b = m.blocks[(size_t) l];
    const int64_t N = g.n_embd, DI = g.ssm_d_inner, S = g.ssm_d_state, G = g.ssm_n_group, H = g.ssm_n_head, P = g.ssm_head_dim();
    const int64_t C = g.ssm_conv_dim(), K = g.ssm_d_conv, W_IN = g.ssm_in_dim();
    std::vector<float> proj((size_t) (T * W_IN));
    for (int64_t t = 0; t < T; ++t) R::matvec(b.ssm_in.data(), h.data() + t * N, N, W_IN, proj.data() + t * W_IN);
    std::vector<float> xbc((size_t) (T * C));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t c = 0; c < C; ++c) {
            float acc = b.conv_b[(size_t) c];
            for (int64_t k = 0; k < K; ++k) {
                const int64_t src = t - (K - 1) + k;   // weight tap k sees token t - (K - 1 - k)
                if (src >= 0) acc += proj[(size_t) (src * W_IN + DI + c)] * b.conv_w[(size_t) (c * K + k)];
            }
            xbc[(size_t) (t * C + c)] = acc / (1.0f + std::exp(-acc));
        }
    std::vector<float> sp((size_t) (T * H));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t hh = 0; hh < H; ++hh) sp[(size_t) (t * H + hh)] = R::softplus(proj[(size_t) (t * W_IN + DI + C + hh)] + b.dt_b[(size_t) hh]);
    std::vector<float> out((size_t) (T * N)), y((size_t) DI);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t hh = 0; hh < H; ++hh) {
            const int64_t gi = hh / (H / G);
            for (int64_t p = 0; p < P; ++p) {
                double acc = 0.0;
                for (int64_t s = 0; s <= t; ++s) {
                    double decay = 0.0;
                    for (int64_t r = s + 1; r <= t; ++r) decay += (double) sp[(size_t) (r * H + hh)] * b.A[(size_t) hh];
                    double cb = 0.0;
                    for (int64_t n = 0; n < S; ++n)
                        cb += (double) xbc[(size_t) (t * C + DI + G * S + gi * S + n)] * xbc[(size_t) (s * C + DI + gi * S + n)];
                    acc += std::exp(decay) * cb * xbc[(size_t) (s * C + p + hh * P)] * sp[(size_t) (s * H + hh)];
                }
                const int64_t ii = p + hh * P;
                const float z = proj[(size_t) (t * W_IN + ii)];
                y[(size_t) ii] = ((float) acc + xbc[(size_t) (t * C + ii)] * b.D[(size_t) hh]) * (z / (1.0f + std::exp(-z)));
            }
        }
        const int64_t gw = DI / G;
        for (int64_t gi = 0; gi < G; ++gi) {
            double ss = 0.0;
            for (int64_t i = 0; i < gw; ++i) ss += (double) y[(size_t) (gi * gw + i)] * y[(size_t) (gi * gw + i)];
            const float sc = (float) (1.0 / std::sqrt(ss / (double) gw + g.rms_eps));
            for (int64_t i = 0; i < gw; ++i) y[(size_t) (gi * gw + i)] *= sc * b.ssm_norm[(size_t) (gi * gw + i)];
        }
        R::matvec(b.ssm_out.data(), y.data(), DI, N, out.data() + t * N);
    }
    return out;
}

/// Causal attention of block `l` over the whole sequence at once (the masked form).
std::vector<float> attention_full(const R::Model& m, int64_t l, const std::vector<float>& h, int64_t T) {
    const Geometry& g = m.g;
    const R::Block& b = m.blocks[(size_t) l];
    const int64_t N = g.n_embd, Hq = g.n_head, Hk = g.n_head_kv, D = g.head_dim;
    std::vector<float> q((size_t) (T * Hq * D)), k((size_t) (T * Hk * D)), v((size_t) (T * Hk * D)), out((size_t) (T * N));
    for (int64_t t = 0; t < T; ++t) {
        R::matvec(b.wq.data(), h.data() + t * N, N, Hq * D, q.data() + t * Hq * D);
        R::matvec(b.wk.data(), h.data() + t * N, N, Hk * D, k.data() + t * Hk * D);
        R::matvec(b.wv.data(), h.data() + t * N, N, Hk * D, v.data() + t * Hk * D);
    }
    std::vector<float> o((size_t) (Hq * D));
    for (int64_t t = 0; t < T; ++t) {
        std::fill(o.begin(), o.end(), 0.0f);
        for (int64_t hq = 0; hq < Hq; ++hq) {
            const int64_t hk = hq * Hk / Hq;   // GQA: query heads hq * Hk / Hq share a key head
            std::vector<double> w((size_t) T, 0.0);
            double mx = -1e300, sum = 0.0;
            for (int64_t s = 0; s < T; ++s) {
                if (s > t) continue;   // the causal mask
                double d = 0.0;
                for (int64_t i = 0; i < D; ++i) d += (double) q[(size_t) ((t * Hq + hq) * D + i)] * k[(size_t) ((s * Hk + hk) * D + i)];
                w[(size_t) s] = d / std::sqrt((double) D);
                mx = std::max(mx, w[(size_t) s]);
            }
            for (int64_t s = 0; s <= t; ++s) sum += (w[(size_t) s] = std::exp(w[(size_t) s] - mx));
            for (int64_t s = 0; s <= t; ++s)
                for (int64_t i = 0; i < D; ++i) o[(size_t) (hq * D + i)] += (float) (w[(size_t) s] / sum) * v[(size_t) ((s * Hk + hk) * D + i)];
        }
        R::matvec(b.wo.data(), o.data(), Hq * D, N, out.data() + t * N);
    }
    return out;
}
}  // namespace

int main() {
    std::printf("nemotron_reference_test\n");
    const Geometry g = R::tiny_geometry();
    const R::Model m = R::random_model(g, 1234);
    const int64_t N = g.n_embd, T = 24;
    check(g.pattern() == "MEM*EMX" && g.ssm_head_dim() == 8 && g.ssm_conv_dim() == 64, "the tiny geometry has the real structure");

    // ---- the pieces
    check(R::softplus(25.0f) == 25.0f && std::fabs(R::softplus(0.0f) - std::log(2.0f)) < 1e-6f, "softplus: ggml's threshold at 20");
    {
        std::vector<float> x = inputs(1, 37, 3), y(37);
        R::layer_norm(x.data(), nullptr, 37, 1e-6f, y.data());
        double mean = 0, var = 0;
        for (float v : y) mean += v;
        mean /= 37;
        for (float v : y) var += (v - mean) * (v - mean);
        check(std::fabs(mean) < 1e-5 && std::fabs(var / 37 - 1.0) < 1e-3, "layer_norm: zero mean, unit variance");
        R::rms_norm(x.data(), nullptr, 37, 1e-6f, y.data());
        double ms = 0;
        for (float v : y) ms += v * v;
        check(std::fabs(ms / 37 - 1.0) < 1e-3, "rms_norm: unit mean square");
    }

    // ---- Mamba2: recurrence == the dual form, for block 0 and block 2
    for (int64_t l : {0, 2}) {
        const std::vector<float> h = inputs(T, N, 10 + (uint32_t) l);
        R::State s = R::new_state(m);
        std::vector<float> rec((size_t) (T * N));
        for (int64_t t = 0; t < T; ++t) R::mamba2(m, l, s, h.data() + t * N, rec.data() + t * N);
        const std::vector<float> dual = mamba2_dual(m, l, h, T);
        const float d = max_abs_diff(rec, dual), scale = max_abs(dual);
        char buf[160];
        std::snprintf(buf, sizeof buf, "Mamba2 block %lld: recurrence vs the whole-sequence dual form, %lld tokens (max diff %.2e of %.2e)",
                      (long long) l, (long long) T, (double) d, (double) scale);
        check(d <= 2e-4f * std::max(1.0f, scale) && scale > 1e-3f, buf);
    }
    {
        // the state carried in two pieces equals one run (conv window and SSM state)
        const std::vector<float> h = inputs(T, N, 77);
        R::State a = R::new_state(m), b = R::new_state(m);
        std::vector<float> one((size_t) (T * N)), two((size_t) (T * N));
        for (int64_t t = 0; t < T; ++t) R::mamba2(m, 0, a, h.data() + t * N, one.data() + t * N);
        for (int64_t t = 0; t < 7; ++t) R::mamba2(m, 0, b, h.data() + t * N, two.data() + t * N);
        R::State c = b;   // a snapshot, as the engine keeps for a conversation
        for (int64_t t = 7; t < T; ++t) R::mamba2(m, 0, c, h.data() + t * N, two.data() + t * N);
        check(one == two, "Mamba2: a snapshot after 7 tokens continues bit for bit");
    }

    // ---- attention: incremental == masked full sequence; GQA mapping
    {
        const std::vector<float> h = inputs(T, N, 21);
        std::vector<float> k, v, inc((size_t) (T * N));
        for (int64_t t = 0; t < T; ++t) R::attention(m, 3, k, v, t, h.data() + t * N, inc.data() + t * N);
        const std::vector<float> full = attention_full(m, 3, h, T);
        char buf[160];
        std::snprintf(buf, sizeof buf, "attention (NoPE, GQA 4:2): incremental KV vs masked full sequence (max diff %.2e)",
                      (double) max_abs_diff(inc, full));
        check(max_abs_diff(inc, full) <= 1e-5f * std::max(1.0f, max_abs(full)), buf);
        // GQA: the value rows of KV head 1 reach only query heads 2 and 3 (Hq / Hk = 2 query heads per KV head).  With Wo restricted
        // to the columns of query heads 0, 1 the output does not change when KV head 1's values do; with heads 2, 3 it does.
        const int64_t D = g.head_dim, HqD = g.n_head * D;
        std::vector<float> x = inputs(1, N, 5);
        auto run = [&](bool scale_kv1, int64_t q0, int64_t q1) {
            R::Model mm = m;
            if (scale_kv1)
                for (int64_t i = D * N; i < 2 * D * N; ++i) mm.blocks[3].wv[(size_t) i] *= -3.0f;
            for (int64_t o = 0; o < N; ++o)
                for (int64_t c = 0; c < HqD; ++c)
                    if (c / D < q0 || c / D > q1) mm.blocks[3].wo[(size_t) (o * HqD + c)] = 0.0f;
            std::vector<float> kk, vv, out((size_t) N);
            R::attention(mm, 3, kk, vv, 0, x.data(), out.data());
            return out;
        };
        check(max_abs_diff(run(false, 0, 1), run(true, 0, 1)) == 0.0f && max_abs_diff(run(false, 2, 3), run(true, 2, 3)) > 1e-4f,
              "attention: KV head 1 serves query heads 2 and 3 only");
    }

    // ---- MoE: selection bias, weights, relu^2
    {
        R::Model mm = m;
        const int64_t l = 1;
        std::vector<float> x = inputs(1, N, 9), out(N);
        std::vector<int32_t> pick((size_t) g.n_expert_used);
        std::vector<float> w((size_t) g.n_expert_used);
        // the expert with the lowest probability, given a large selection bias, is picked; its weight stays its own probability
        std::vector<float> logit((size_t) g.n_expert);
        R::matvec(mm.blocks[l].router.data(), x.data(), N, g.n_expert, logit.data());
        const int64_t low = std::min_element(logit.begin(), logit.end()) - logit.begin();
        std::fill(mm.blocks[l].probs_b.begin(), mm.blocks[l].probs_b.end(), 0.0f);
        mm.blocks[l].probs_b[(size_t) low] = 10.0f;
        R::moe(mm, l, x.data(), out.data(), pick.data(), w.data());
        check(pick[0] == (int32_t) low, "MoE: the selection bias picks the least likely expert first");
        // the weights: sigmoid of the UNBIASED logits of the picked experts, normalised to 1, times expert_weights_scale
        double sum = 0.0;
        for (int32_t e : pick) sum += sigm(logit[(size_t) e]);
        bool ok = true;
        double wsum = 0.0;
        for (size_t k = 0; k < pick.size(); ++k) {
            const double want = sigm(logit[(size_t) pick[k]]) / sum * g.expert_weights_scale;
            ok &= std::fabs(w[k] - want) < 1e-5;
            wsum += w[k];
        }
        check(ok && std::fabs(wsum - g.expert_weights_scale) < 1e-5, "MoE: weights = unbiased sigmoid, normalised, x expert_weights_scale");
        // relu^2: negating every routed expert's up rows changes the routed part (the squares of the other half);
        // a zero latent gives only the shared expert
        R::Model z = mm;
        std::fill(z.blocks[l].latent_down.begin(), z.blocks[l].latent_down.end(), 0.0f);
        std::vector<float> shared_only(N), sh_ref(N), hidden((size_t) g.n_ff_shexp);
        R::moe(z, l, x.data(), shared_only.data());
        R::matvec(z.blocks[l].up_sh.data(), x.data(), N, g.n_ff_shexp, hidden.data());
        for (float& u : hidden) u = u > 0 ? u * u : 0.0f;
        R::matvec(z.blocks[l].down_sh.data(), hidden.data(), g.n_ff_shexp, N, sh_ref.data());
        check(max_abs_diff(shared_only, sh_ref) < 1e-5f, "MoE: with a zero latent only the relu^2 shared expert remains");
    }

    // ---- the model
    {
        const int64_t V = g.n_vocab;
        std::vector<int32_t> toks;
        for (int64_t t = 0; t < 20; ++t) toks.push_back((int32_t) ((t * 7 + 3) % V));
        R::State a = R::new_state(m);
        std::vector<float> la((size_t) V), all;
        for (int32_t t : toks) {
            R::forward(m, a, t, la.data(), nullptr);
            all.insert(all.end(), la.begin(), la.end());
        }
        check(finite(all) && max_abs(all) > 1e-3f, "the model: 20 tokens, every logit finite and not zero");
        R::State b = R::new_state(m);
        std::vector<float> lb((size_t) V), all_b;
        for (size_t i = 0; i < 9; ++i) {
            R::forward(m, b, toks[i], lb.data(), nullptr);
            all_b.insert(all_b.end(), lb.begin(), lb.end());
        }
        R::State c = b;   // the held state of a conversation
        for (size_t i = 9; i < toks.size(); ++i) {
            R::forward(m, c, toks[i], lb.data(), nullptr);
            all_b.insert(all_b.end(), lb.begin(), lb.end());
        }
        check(all == all_b && c.pos == 20, "the model: a held state after 9 tokens continues bit for bit");

        // MTP: seeded by h_nextn of position t and token t + 1; chained on its own output
        R::State s = R::new_state(m);
        std::vector<float> hn((size_t) N), lg((size_t) V), h1((size_t) N), h2((size_t) N), l1((size_t) V), l2((size_t) V);
        for (size_t i = 0; i + 1 < toks.size(); ++i) R::forward(m, s, toks[i], lg.data(), hn.data());
        R::MtpState ms = R::new_mtp_state(m);
        R::mtp(m, ms, toks.back(), hn.data(), l1.data(), h1.data());
        const int32_t d1 = (int32_t) (std::max_element(l1.begin(), l1.end()) - l1.begin());
        R::mtp(m, ms, d1, h1.data(), l2.data(), h2.data());
        R::MtpState ms2 = R::new_mtp_state(m);
        std::vector<float> l1b((size_t) V), h1b((size_t) N);
        R::mtp(m, ms2, toks.back(), hn.data(), l1b.data(), h1b.data());
        double mean = 0;
        for (float v : h1) mean += v;
        check(finite(l1) && finite(l2) && l1 == l1b && ms.pos == 2, "MTP: two chained drafts, finite and deterministic");
        check(std::fabs(mean / N) < 0.3, "MTP: its head norm is a LayerNorm (the seed it passes on is centred)");
    }
    std::printf(g_fail ? "FAILED: %d\n" : "all ok\n", g_fail);
    return g_fail ? 1 : 0;
}
