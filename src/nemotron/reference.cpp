// src/nemotron/reference.cpp - the fp32 CPU reference of nemotron_h_moe (see include/strata/nemotron/reference.hpp).
#include "strata/nemotron/reference.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace strata::nemotron::ref {

namespace {
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu(float x) { return x * sigmoid(x); }
float relu_sqr(float x) { return x > 0.0f ? x * x : 0.0f; }
}  // namespace

float softplus(float x) { return x > 20.0f ? x : std::log(1.0f + std::exp(x)); }

void rms_norm(const float* x, const float* w, int64_t n, float eps, float* out) {
    float ss = 0.0f;
    for (int64_t i = 0; i < n; ++i) ss += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(ss / (float) n + eps);
    for (int64_t i = 0; i < n; ++i) out[i] = x[i] * scale * (w ? w[i] : 1.0f);
}

void layer_norm(const float* x, const float* w, int64_t n, float eps, float* out) {
    float mean = 0.0f;
    for (int64_t i = 0; i < n; ++i) mean += x[i];
    mean /= (float) n;
    float var = 0.0f;
    for (int64_t i = 0; i < n; ++i) var += (x[i] - mean) * (x[i] - mean);
    var /= (float) n;
    const float scale = 1.0f / std::sqrt(var + eps);
    for (int64_t i = 0; i < n; ++i) out[i] = (x[i] - mean) * scale * (w ? w[i] : 1.0f);
}

void matvec(const float* W, const float* x, int64_t n_in, int64_t n_out, float* y) {
    for (int64_t o = 0; o < n_out; ++o) {
        const float* row = W + o * n_in;
        float acc = 0.0f;
        for (int64_t i = 0; i < n_in; ++i) acc += row[i] * x[i];
        y[o] = acc;
    }
}

State new_state(const Model& m) {
    const Geometry& g = m.g;
    State s;
    s.conv.resize((size_t) g.n_layers);
    s.ssm.resize((size_t) g.n_layers);
    s.k.resize((size_t) g.n_layers);
    s.v.resize((size_t) g.n_layers);
    for (int64_t l = 0; l < g.n_layers; ++l)
        if (g.kind[(size_t) l] == BlockKind::Mamba) {
            s.conv[(size_t) l].assign((size_t) ((g.ssm_d_conv - 1) * g.ssm_conv_dim()), 0.0f);
            s.ssm[(size_t) l].assign((size_t) (g.ssm_d_inner * g.ssm_d_state), 0.0f);
        }
    return s;
}

MtpState new_mtp_state(const Model&) { return {}; }

void mamba2(const Model& m, int64_t l, State& s, const float* h, float* out) {
    const Geometry& g = m.g;
    const Block& b = m.blocks[(size_t) l];
    const int64_t DI = g.ssm_d_inner, N = g.ssm_d_state, G = g.ssm_n_group, H = g.ssm_n_head, P = g.ssm_head_dim();
    const int64_t C = g.ssm_conv_dim(), K = g.ssm_d_conv;
    std::vector<float> zxbcdt((size_t) g.ssm_in_dim());
    matvec(b.ssm_in.data(), h, g.n_embd, g.ssm_in_dim(), zxbcdt.data());
    const float* z = zxbcdt.data();
    const float* xbc_in = z + DI;
    const float* dt = xbc_in + C;
    // causal depthwise conv over the xBC channels: [state (K - 1, oldest first), this token] . w
    std::vector<float>& cs = s.conv[(size_t) l];
    std::vector<float> xbc((size_t) C);
    for (int64_t c = 0; c < C; ++c) {
        const float* w = b.conv_w.data() + c * K;
        float acc = 0.0f;
        for (int64_t k = 0; k < K - 1; ++k) acc += cs[(size_t) (k * C + c)] * w[k];
        acc += xbc_in[c] * w[K - 1];
        xbc[(size_t) c] = silu(acc + b.conv_b[(size_t) c]);
    }
    for (int64_t k = 0; k + 1 < K - 1; ++k)   // shift the window by one token
        for (int64_t c = 0; c < C; ++c) cs[(size_t) (k * C + c)] = cs[(size_t) ((k + 1) * C + c)];
    for (int64_t c = 0; c < C; ++c) cs[(size_t) ((K - 2) * C + c)] = xbc_in[c];

    const float* x = xbc.data();
    const float* B = x + DI;
    const float* Cm = B + G * N;
    std::vector<float>& st = s.ssm[(size_t) l];
    std::vector<float> y((size_t) DI);
    for (int64_t hh = 0; hh < H; ++hh) {
        const float sp = softplus(dt[hh] + b.dt_b[(size_t) hh]);
        const float dA = std::exp(sp * b.A[(size_t) hh]);
        const int64_t gi = hh / (H / G);
        for (int64_t p = 0; p < P; ++p) {
            const int64_t ii = p + hh * P;
            const float xdt = x[ii] * sp;
            float* sv = st.data() + ii * N;
            float acc = 0.0f;
            for (int64_t n = 0; n < N; ++n) {
                sv[n] = sv[n] * dA + B[gi * N + n] * xdt;
                acc += sv[n] * Cm[gi * N + n];
            }
            y[(size_t) ii] = (acc + x[ii] * b.D[(size_t) hh]) * silu(z[ii]);
        }
    }
    const int64_t gw = DI / G;   // grouped RMS norm
    for (int64_t gi = 0; gi < G; ++gi)
        rms_norm(y.data() + gi * gw, b.ssm_norm.data() + gi * gw, gw, g.rms_eps, y.data() + gi * gw);
    matvec(b.ssm_out.data(), y.data(), DI, g.n_embd, out);
}

void attention(const Model& m, int64_t l, std::vector<float>& kc, std::vector<float>& vc, int64_t pos, const float* h, float* out) {
    const Geometry& g = m.g;
    const Block& b = m.blocks[(size_t) l];
    const int64_t Hq = g.n_head, Hk = g.n_head_kv, D = g.head_dim, rep = Hq / Hk;
    std::vector<float> q((size_t) (Hq * D)), k((size_t) (Hk * D)), v((size_t) (Hk * D));
    matvec(b.wq.data(), h, g.n_embd, Hq * D, q.data());
    matvec(b.wk.data(), h, g.n_embd, Hk * D, k.data());
    matvec(b.wv.data(), h, g.n_embd, Hk * D, v.data());
    if ((int64_t) kc.size() != pos * Hk * D) throw std::runtime_error("nemotron ref: the K cache is not at the position");
    kc.insert(kc.end(), k.begin(), k.end());
    vc.insert(vc.end(), v.begin(), v.end());
    const int64_t T = pos + 1;
    const float scale = 1.0f / std::sqrt((float) D);
    std::vector<float> o((size_t) (Hq * D), 0.0f), sc((size_t) T);
    for (int64_t hq = 0; hq < Hq; ++hq) {
        const int64_t hk = hq / rep;
        float mx = -INFINITY;
        for (int64_t t = 0; t < T; ++t) {
            const float* kt = kc.data() + (t * Hk + hk) * D;
            float d = 0.0f;
            for (int64_t i = 0; i < D; ++i) d += q[(size_t) (hq * D + i)] * kt[i];
            sc[(size_t) t] = d * scale;
            mx = std::max(mx, sc[(size_t) t]);
        }
        float sum = 0.0f;
        for (int64_t t = 0; t < T; ++t) sum += (sc[(size_t) t] = std::exp(sc[(size_t) t] - mx));
        for (int64_t t = 0; t < T; ++t) {
            const float* vt = vc.data() + (t * Hk + hk) * D;
            const float p = sc[(size_t) t] / sum;
            for (int64_t i = 0; i < D; ++i) o[(size_t) (hq * D + i)] += p * vt[i];
        }
    }
    matvec(b.wo.data(), o.data(), Hq * D, g.n_embd, out);
}

void moe(const Model& m, int64_t l, const float* h, float* out, int32_t* picked, float* weights, float* margin) {
    const Geometry& g = m.g;
    const Block& b = m.blocks[(size_t) l];
    const int64_t N = g.n_embd, E = g.n_expert, Kp = g.n_expert_used, L = g.moe_latent, F = g.n_ff_exp, FS = g.n_ff_shexp;
    std::vector<float> logit((size_t) E), prob((size_t) E), sel((size_t) E);
    matvec(b.router.data(), h, N, E, logit.data());
    for (int64_t e = 0; e < E; ++e) {
        prob[(size_t) e] = sigmoid(logit[(size_t) e]);
        sel[(size_t) e] = prob[(size_t) e] + b.probs_b[(size_t) e];   // the bias picks, it does not weigh
    }
    std::vector<int32_t> idx((size_t) E);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + std::min<int64_t>(Kp + 1, E), idx.end(), [&](int32_t a, int32_t c) {
        return sel[(size_t) a] != sel[(size_t) c] ? sel[(size_t) a] > sel[(size_t) c] : a < c;
    });
    if (margin && Kp < E) *margin = std::min(*margin, sel[(size_t) idx[(size_t) Kp - 1]] - sel[(size_t) idx[(size_t) Kp]]);
    std::vector<float> w((size_t) Kp);
    float wsum = 0.0f;
    for (int64_t k = 0; k < Kp; ++k) wsum += (w[(size_t) k] = prob[(size_t) idx[(size_t) k]]);
    if (g.expert_weights_norm)
        for (float& x : w) x /= std::max(wsum, 6.103515625e-5f);
    if (g.expert_weights_scale != 0.0f && g.expert_weights_scale != 1.0f)
        for (float& x : w) x *= g.expert_weights_scale;

    std::vector<float> lat((size_t) L), up((size_t) F), dn((size_t) L), acc((size_t) L, 0.0f);
    matvec(b.latent_down.data(), h, N, L, lat.data());
    for (int64_t k = 0; k < Kp; ++k) {
        const int64_t e = idx[(size_t) k];
        matvec(b.up_exps.data() + e * F * L, lat.data(), L, F, up.data());
        for (float& x : up) x = relu_sqr(x);
        matvec(b.down_exps.data() + e * L * F, up.data(), F, L, dn.data());
        for (int64_t i = 0; i < L; ++i) acc[(size_t) i] += w[(size_t) k] * dn[(size_t) i];
        if (picked) picked[k] = (int32_t) e;
        if (weights) weights[k] = w[(size_t) k];
    }
    std::vector<float> moe_out((size_t) N), sh((size_t) FS), sh_out((size_t) N);
    matvec(b.latent_up.data(), acc.data(), L, N, moe_out.data());
    matvec(b.up_sh.data(), h, N, FS, sh.data());
    for (float& x : sh) x = relu_sqr(x);
    matvec(b.down_sh.data(), sh.data(), FS, N, sh_out.data());
    for (int64_t i = 0; i < N; ++i) out[i] = moe_out[(size_t) i] + sh_out[(size_t) i];
}

void forward(const Model& m, State& s, int32_t token, float* logits, float* h_nextn) {
    const Geometry& g = m.g;
    const int64_t N = g.n_embd;
    if (token < 0 || token >= g.n_vocab) throw std::runtime_error("nemotron ref: token outside the vocabulary");
    std::vector<float> x(m.tok_embd.begin() + token * N, m.tok_embd.begin() + (token + 1) * N), hn((size_t) N), o((size_t) N);
    s.margin = INFINITY;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        rms_norm(x.data(), m.blocks[(size_t) l].attn_norm.data(), N, g.rms_eps, hn.data());
        switch (g.kind[(size_t) l]) {
        case BlockKind::Mamba: mamba2(m, l, s, hn.data(), o.data()); break;
        case BlockKind::Attention: attention(m, l, s.k[(size_t) l], s.v[(size_t) l], s.pos, hn.data(), o.data()); break;
        case BlockKind::Moe: moe(m, l, hn.data(), o.data(), nullptr, nullptr, &s.margin); break;
        case BlockKind::MtpAttentionMoe: throw std::runtime_error("nemotron ref: an MTP block among the main blocks");
        }
        for (int64_t i = 0; i < N; ++i) x[(size_t) i] += o[(size_t) i];
    }
    ++s.pos;
    rms_norm(x.data(), m.output_norm.data(), N, g.rms_eps, hn.data());
    if (h_nextn) std::copy(hn.begin(), hn.end(), h_nextn);
    if (logits) matvec(m.output.data(), hn.data(), N, g.n_vocab, logits);
}

void mtp(const Model& m, MtpState& s, int32_t token, const float* h, float* logits, float* h_out) {
    const Geometry& g = m.g;
    if (g.n_mtp < 1) throw std::runtime_error("nemotron ref: the model has no MTP block");
    const int64_t N = g.n_embd, l = g.n_layers;
    const Block& b = m.blocks[(size_t) l];
    std::vector<float> cat((size_t) (2 * N)), x((size_t) N), hn((size_t) N), o((size_t) N);
    rms_norm(m.tok_embd.data() + token * N, b.enorm.data(), N, g.rms_eps, cat.data());
    rms_norm(h, b.hnorm.data(), N, g.rms_eps, cat.data() + N);
    matvec(b.eh_proj.data(), cat.data(), 2 * N, N, x.data());
    rms_norm(x.data(), b.attn_norm.data(), N, g.rms_eps, hn.data());
    attention(m, l, s.k, s.v, s.pos, hn.data(), o.data());
    for (int64_t i = 0; i < N; ++i) x[(size_t) i] += o[(size_t) i];
    rms_norm(x.data(), b.post_norm.data(), N, g.rms_eps, hn.data());
    moe(m, l, hn.data(), o.data());
    for (int64_t i = 0; i < N; ++i) x[(size_t) i] += o[(size_t) i];
    ++s.pos;
    layer_norm(x.data(), b.shared_head_norm.data(), N, g.rms_eps, hn.data());
    if (h_out) std::copy(hn.begin(), hn.end(), h_out);
    if (logits) matvec(m.output.data(), hn.data(), N, g.n_vocab, logits);
}

Geometry tiny_geometry() {
    Geometry g;
    g.n_layers = 6;
    g.n_mtp = 1;
    g.n_embd = 64;
    g.n_vocab = 50;
    g.n_head = 4;
    g.n_head_kv = 2;
    g.head_dim = 16;
    g.n_rot = 16;
    g.n_expert = 8;
    g.n_expert_used = 3;
    g.n_ff_exp = 32;
    g.n_ff_shexp = 48;
    g.moe_latent = 16;
    g.expert_weights_scale = 2.5f;
    g.expert_weights_norm = true;
    g.ssm_d_conv = 4;
    g.ssm_d_state = 8;
    g.ssm_n_group = 2;
    g.ssm_n_head = 4;
    g.ssm_d_inner = 32;
    g.rms_eps = 1e-5f;
    g.kind = {BlockKind::Mamba, BlockKind::Moe, BlockKind::Mamba, BlockKind::Attention, BlockKind::Moe, BlockKind::Mamba,
              BlockKind::MtpAttentionMoe};
    return g;
}

Model random_model(const Geometry& g, uint64_t seed) {
    std::mt19937_64 rng(seed);
    auto uni = [&](float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); };
    auto mat = [&](int64_t n_in, int64_t n_out) {
        std::vector<float> w((size_t) (n_in * n_out));
        const float a = 1.0f / std::sqrt((float) n_in);
        for (float& x : w) x = uni(-a, a) * 1.7f;
        return w;
    };
    auto vec = [&](int64_t n, float lo, float hi) {
        std::vector<float> w((size_t) n);
        for (float& x : w) x = uni(lo, hi);
        return w;
    };
    Model m;
    m.g = g;
    const int64_t N = g.n_embd;
    m.tok_embd = vec(N * g.n_vocab, -1.0f, 1.0f);
    m.output_norm = vec(N, 0.8f, 1.2f);
    m.output = mat(N, g.n_vocab);
    m.blocks.resize((size_t) g.n_blocks());
    for (int64_t l = 0; l < g.n_blocks(); ++l) {
        Block& b = m.blocks[(size_t) l];
        b.attn_norm = vec(N, 0.8f, 1.2f);
        const BlockKind k = g.kind[(size_t) l];
        if (k == BlockKind::Mamba) {
            b.ssm_in = mat(N, g.ssm_in_dim());
            b.conv_w = vec(g.ssm_d_conv * g.ssm_conv_dim(), -0.6f, 0.6f);
            b.conv_b = vec(g.ssm_conv_dim(), -0.1f, 0.1f);
            b.dt_b = vec(g.ssm_n_head, -2.0f, 0.5f);
            b.A = vec(g.ssm_n_head, -2.0f, -0.05f);   // the GGUF stores A = -exp(A_log): negative
            b.D = vec(g.ssm_n_head, 0.5f, 1.5f);
            b.ssm_norm = vec(g.ssm_d_inner, 0.8f, 1.2f);
            b.ssm_out = mat(g.ssm_d_inner, N);
        }
        if (k == BlockKind::Attention || k == BlockKind::MtpAttentionMoe) {
            b.wq = mat(N, g.n_head * g.head_dim);
            b.wk = mat(N, g.n_head_kv * g.head_dim);
            b.wv = mat(N, g.n_head_kv * g.head_dim);
            b.wo = mat(g.n_head * g.head_dim, N);
        }
        if (k == BlockKind::Moe || k == BlockKind::MtpAttentionMoe) {
            b.router = mat(N, g.n_expert);
            b.probs_b = vec(g.n_expert, -0.05f, 0.05f);
            b.latent_down = mat(N, g.moe_latent);
            b.latent_up = mat(g.moe_latent, N);
            b.up_exps = mat(g.moe_latent, g.n_ff_exp * g.n_expert);   // n_expert matrices of [latent, ff]
            b.down_exps = mat(g.n_ff_exp, g.moe_latent * g.n_expert);
            b.up_sh = mat(N, g.n_ff_shexp);
            b.down_sh = mat(g.n_ff_shexp, N);
        }
        if (k == BlockKind::MtpAttentionMoe) {
            b.post_norm = vec(N, 0.8f, 1.2f);
            b.eh_proj = mat(2 * N, N);
            b.enorm = vec(N, 0.8f, 1.2f);
            b.hnorm = vec(N, 0.8f, 1.2f);
            b.shared_head_norm = vec(N, 0.8f, 1.2f);
        }
    }
    return m;
}

}  // namespace strata::nemotron::ref
