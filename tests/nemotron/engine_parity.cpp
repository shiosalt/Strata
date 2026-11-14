// tests/nemotron/engine_parity.cpp - the nemotron_h_moe engine (src/nemotron) end to end against the fp32 CPU reference, on a
// synthetic model with the real structure (Mamba2 / MoE / attention blocks + the MTP block) quantized by ggml into the types the
// Unsloth UD-IQ3_XXS file uses (Q6_K / Q5_K / Q4_K / Q8_0 dense, IQ3_XXS / IQ4_XS up and IQ4_NL / Q2_0 down experts) and written as a
// GGUF.  The reference runs on the dequantized weights, so the difference is the engine's activation rounding (q8_1, FP16 GEMMs,
// the fp16 K / V cache).  Checked: the prompt's logits (the prompt path, two chunks), every greedy step's logits (the decode path),
// with the experts all on the CPU, all on the GPU and split; the MTP drafts giving the same tokens as plain greedy decoding; a
// conversation that continues reusing the held state.  A few hundred MB of GPU memory, seconds.  Exit 77 without a GPU.
#include "../../src/nemotron/engine.hpp"
#include "strata/nemotron/reference.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace R = strata::nemotron::ref;
namespace N = strata::nemotron;
using strata::nemotron::BlockKind;
using strata::nemotron::Geometry;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-104s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

Geometry geometry() {
    Geometry g;
    g.n_layers = 6;
    g.n_mtp = 1;
    g.n_embd = 256;
    g.n_vocab = 512;
    g.n_head = 4;
    g.n_head_kv = 2;
    g.head_dim = 64;
    g.n_rot = 64;
    g.n_expert = 16;
    g.n_expert_used = 4;
    g.n_ff_exp = 192;
    g.n_ff_shexp = 512;
    g.moe_latent = 256;
    g.expert_weights_scale = 2.5f;
    g.expert_weights_norm = true;
    g.ssm_d_conv = 4;
    g.ssm_d_state = 32;
    g.ssm_n_group = 2;
    g.ssm_n_head = 8;
    g.ssm_d_inner = 512;
    g.rms_eps = 1e-5f;
    g.kind = {BlockKind::Mamba, BlockKind::Moe, BlockKind::Mamba, BlockKind::Attention, BlockKind::Moe, BlockKind::Mamba,
              BlockKind::MtpAttentionMoe};
    return g;
}

// ---------------------------------------------------------------------------------------------------------------- a GGUF writer
struct Out {
    std::vector<uint8_t> b;
    void put(const void* p, size_t n) { b.insert(b.end(), (const uint8_t*) p, (const uint8_t*) p + n); }
    void u32(uint32_t v) { put(&v, 4); }
    void u64(uint64_t v) { put(&v, 8); }
    void f32(float v) { put(&v, 4); }
    void str(const std::string& s) {
        u64(s.size());
        put(s.data(), s.size());
    }
};
struct Tensor {
    std::string name;
    std::vector<uint64_t> shape;
    ggml_type type;
    std::vector<uint8_t> data;
};

/// `w` (shape[0] values per row) quantized to `type`; `w` is replaced by the dequantized values.
std::vector<uint8_t> quantize(ggml_type type, std::vector<float>& w, int64_t n_per_row) {
    const int64_t rows = (int64_t) w.size() / n_per_row;
    if (type == GGML_TYPE_F32) {
        std::vector<uint8_t> q(w.size() * 4);
        std::memcpy(q.data(), w.data(), q.size());
        return q;
    }
    std::vector<float> imat((size_t) n_per_row, 1.0f);
    std::vector<uint8_t> q(ggml_row_size(type, n_per_row) * (size_t) rows);
    ggml_quantize_chunk(type, w.data(), q.data(), 0, rows, n_per_row, ggml_quantize_requires_imatrix(type) ? imat.data() : nullptr);
    ggml_get_type_traits(type)->to_float(q.data(), w.data(), (int64_t) w.size());
    return q;
}

/// The model as a GGUF (its matrices quantized; `m` gets the dequantized weights the file holds).
void write_gguf(const std::filesystem::path& path, R::Model& m) {
    const Geometry& g = m.g;
    std::vector<Tensor> ts;
    auto add = [&](const std::string& name, std::vector<float>& w, std::vector<uint64_t> shape, ggml_type type) {
        ts.push_back({name, shape, type, quantize(type, w, (int64_t) shape[0])});
    };
    const uint64_t Nn = (uint64_t) g.n_embd;
    add("token_embd.weight", m.tok_embd, {Nn, (uint64_t) g.n_vocab}, GGML_TYPE_Q6_K);
    add("output_norm.weight", m.output_norm, {Nn}, GGML_TYPE_F32);
    add("output.weight", m.output, {Nn, (uint64_t) g.n_vocab}, GGML_TYPE_Q8_0);
    for (int64_t l = 0; l < g.n_blocks(); ++l) {
        R::Block& b = m.blocks[(size_t) l];
        const std::string p = "blk." + std::to_string(l) + ".";
        const BlockKind k = g.kind[(size_t) l];
        add(p + "attn_norm.weight", b.attn_norm, {Nn}, GGML_TYPE_F32);
        if (k == BlockKind::Mamba) {
            add(p + "ssm_in.weight", b.ssm_in, {Nn, (uint64_t) g.ssm_in_dim()}, GGML_TYPE_Q6_K);
            add(p + "ssm_conv1d.weight", b.conv_w, {(uint64_t) g.ssm_d_conv, (uint64_t) g.ssm_conv_dim()}, GGML_TYPE_F32);
            add(p + "ssm_conv1d.bias", b.conv_b, {(uint64_t) g.ssm_conv_dim()}, GGML_TYPE_F32);
            add(p + "ssm_dt.bias", b.dt_b, {(uint64_t) g.ssm_n_head}, GGML_TYPE_F32);
            add(p + "ssm_a", b.A, {1, (uint64_t) g.ssm_n_head}, GGML_TYPE_F32);
            add(p + "ssm_d", b.D, {1, (uint64_t) g.ssm_n_head}, GGML_TYPE_F32);
            add(p + "ssm_norm.weight", b.ssm_norm, {(uint64_t) (g.ssm_d_inner / g.ssm_n_group), (uint64_t) g.ssm_n_group}, GGML_TYPE_F32);
            add(p + "ssm_out.weight", b.ssm_out, {(uint64_t) g.ssm_d_inner, Nn}, GGML_TYPE_Q8_0);
        }
        if (k == BlockKind::Attention || k == BlockKind::MtpAttentionMoe) {
            add(p + "attn_q.weight", b.wq, {Nn, (uint64_t) (g.n_head * g.head_dim)}, GGML_TYPE_Q8_0);
            add(p + "attn_k.weight", b.wk, {Nn, (uint64_t) (g.n_head_kv * g.head_dim)}, GGML_TYPE_Q8_0);
            add(p + "attn_v.weight", b.wv, {Nn, (uint64_t) (g.n_head_kv * g.head_dim)}, GGML_TYPE_Q8_0);
            add(p + "attn_output.weight", b.wo, {(uint64_t) (g.n_head * g.head_dim), Nn}, GGML_TYPE_Q5_K);
        }
        if (k == BlockKind::Moe || k == BlockKind::MtpAttentionMoe) {
            const bool first = l == 1, mtp = k == BlockKind::MtpAttentionMoe;
            const ggml_type ut = mtp ? GGML_TYPE_Q8_0 : first ? GGML_TYPE_IQ3_XXS : GGML_TYPE_IQ4_XS;
            const ggml_type dt = mtp ? GGML_TYPE_Q8_0 : first ? GGML_TYPE_IQ4_NL : GGML_TYPE_Q2_0;
            add(p + "ffn_gate_inp.weight", b.router, {Nn, (uint64_t) g.n_expert}, GGML_TYPE_F32);
            add(p + "exp_probs_b.bias", b.probs_b, {(uint64_t) g.n_expert}, GGML_TYPE_F32);
            add(p + "ffn_latent_down.weight", b.latent_down, {Nn, (uint64_t) g.moe_latent}, GGML_TYPE_Q8_0);
            add(p + "ffn_latent_up.weight", b.latent_up, {(uint64_t) g.moe_latent, Nn}, GGML_TYPE_Q8_0);
            add(p + "ffn_up_exps.weight", b.up_exps, {(uint64_t) g.moe_latent, (uint64_t) g.n_ff_exp, (uint64_t) g.n_expert}, ut);
            add(p + "ffn_down_exps.weight", b.down_exps, {(uint64_t) g.n_ff_exp, (uint64_t) g.moe_latent, (uint64_t) g.n_expert}, dt);
            add(p + "ffn_up_shexp.weight", b.up_sh, {Nn, (uint64_t) g.n_ff_shexp}, GGML_TYPE_Q4_K);
            add(p + "ffn_down_shexp.weight", b.down_sh, {(uint64_t) g.n_ff_shexp, Nn}, GGML_TYPE_Q5_K);
        }
        if (k == BlockKind::MtpAttentionMoe) {
            add(p + "post_attention_norm.weight", b.post_norm, {Nn}, GGML_TYPE_F32);
            add(p + "nextn.eh_proj.weight", b.eh_proj, {2 * Nn, Nn}, GGML_TYPE_Q8_0);
            add(p + "nextn.enorm.weight", b.enorm, {Nn}, GGML_TYPE_F32);
            add(p + "nextn.hnorm.weight", b.hnorm, {Nn}, GGML_TYPE_F32);
            add(p + "nextn.shared_head_norm.weight", b.shared_head_norm, {Nn}, GGML_TYPE_F32);
        }
    }
    // metadata
    Out o;
    const std::string a = "nemotron_h_moe.";
    struct Kv {
        std::string k;
        int kind;   // 0 u32, 1 f32, 2 string, 3 u32 array, 4 string array
        uint32_t u = 0;
        float f = 0;
        std::string s;
        std::vector<uint32_t> arr;
        std::vector<std::string> sarr;
    };
    std::vector<Kv> kv;
    auto ku = [&](const std::string& k, uint32_t v) { Kv x; x.k = k; x.kind = 0; x.u = v; kv.push_back(x); };
    auto kf = [&](const std::string& k, float v) { Kv x; x.k = k; x.kind = 1; x.f = v; kv.push_back(x); };
    Kv arch;
    arch.k = "general.architecture";
    arch.kind = 2;
    arch.s = "nemotron_h_moe";
    kv.push_back(arch);
    ku(a + "block_count", (uint32_t) g.n_blocks());
    ku(a + "embedding_length", (uint32_t) g.n_embd);
    ku(a + "attention.head_count", (uint32_t) g.n_head);
    Kv hkv, ffl;
    hkv.k = a + "attention.head_count_kv";
    hkv.kind = 3;
    ffl.k = a + "feed_forward_length";
    ffl.kind = 3;
    for (int64_t l = 0; l < g.n_blocks(); ++l) {
        const BlockKind k = g.kind[(size_t) l];
        hkv.arr.push_back(k == BlockKind::Attention || k == BlockKind::MtpAttentionMoe ? (uint32_t) g.n_head_kv : 0);
        ffl.arr.push_back(k == BlockKind::Moe || k == BlockKind::MtpAttentionMoe ? (uint32_t) g.n_ff_exp : 0);
    }
    kv.push_back(hkv);
    kv.push_back(ffl);
    kf(a + "attention.layer_norm_epsilon", g.rms_eps);
    ku(a + "attention.key_length", (uint32_t) g.head_dim);
    ku(a + "attention.value_length", (uint32_t) g.head_dim);
    ku(a + "rope.dimension_count", (uint32_t) g.head_dim);
    ku(a + "expert_count", (uint32_t) g.n_expert);
    ku(a + "expert_used_count", (uint32_t) g.n_expert_used);
    ku(a + "expert_feed_forward_length", (uint32_t) g.n_ff_exp);
    ku(a + "expert_shared_feed_forward_length", (uint32_t) g.n_ff_shexp);
    ku(a + "expert_weights_norm", 1);
    kf(a + "expert_weights_scale", g.expert_weights_scale);
    ku(a + "moe_latent_size", (uint32_t) g.moe_latent);
    ku(a + "nextn_predict_layers", (uint32_t) g.n_mtp);
    ku(a + "ssm.conv_kernel", (uint32_t) g.ssm_d_conv);
    ku(a + "ssm.state_size", (uint32_t) g.ssm_d_state);
    ku(a + "ssm.group_count", (uint32_t) g.ssm_n_group);
    ku(a + "ssm.time_step_rank", (uint32_t) g.ssm_n_head);
    ku(a + "ssm.inner_size", (uint32_t) g.ssm_d_inner);
    ku("tokenizer.ggml.eos_token_id", 11);
    Kv toks;
    toks.k = "tokenizer.ggml.tokens";
    toks.kind = 4;
    for (int64_t i = 0; i < g.n_vocab; ++i)
        toks.sarr.push_back(i == 10 ? "<|im_start|>" : i == 11 ? "<|im_end|>" : i == 18 ? "<image>" : i == 19 ? "<img>" : i == 20 ? "</img>"
                                                                                                                   : "t" + std::to_string(i));
    kv.push_back(toks);
    o.put("GGUF", 4);
    o.u32(3);
    o.u64(ts.size());
    o.u64(kv.size());
    for (const Kv& x : kv) {
        o.str(x.k);
        switch (x.kind) {
        case 0: o.u32(4); o.u32(x.u); break;
        case 1: o.u32(6); o.f32(x.f); break;
        case 2: o.u32(8); o.str(x.s); break;
        case 3: o.u32(9); o.u32(4); o.u64(x.arr.size()); for (uint32_t v : x.arr) o.u32(v); break;
        case 4: o.u32(9); o.u32(8); o.u64(x.sarr.size()); for (const std::string& s : x.sarr) o.str(s); break;
        }
    }
    uint64_t off = 0;
    std::vector<uint64_t> offs;
    for (const Tensor& t : ts) {
        o.str(t.name);
        o.u32((uint32_t) t.shape.size());
        for (uint64_t d : t.shape) o.u64(d);
        o.u32((uint32_t) t.type);
        o.u64(off);
        offs.push_back(off);
        off = (off + t.data.size() + 31) / 32 * 32;
    }
    o.b.resize((o.b.size() + 31) / 32 * 32, 0);
    const size_t base = o.b.size();
    for (size_t i = 0; i < ts.size(); ++i) {
        o.b.resize(base + offs[i], 0);
        o.put(ts[i].data.data(), ts[i].data.size());
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write((const char*) o.b.data(), (std::streamsize) o.b.size());
}

float rel_l2(const float* a, const float* b, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        num += (double) (a[i] - b[i]) * (a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return (float) std::sqrt(num / std::max(den, 1e-30));
}
int argmax(const float* x, size_t n) { return (int) (std::max_element(x, x + n) - x); }

/// The reference's logits after each of `toks` (from a fresh state); `margins` (optional) each token's smallest routing margin.
std::vector<std::vector<float>> ref_logits(const R::Model& m, const std::vector<int32_t>& toks, std::vector<float>* margins = nullptr) {
    R::State s = R::new_state(m);
    std::vector<std::vector<float>> out;
    std::vector<float> lg((size_t) m.g.n_vocab);
    if (margins) margins->clear();
    for (int32_t t : toks) {
        R::forward(m, s, t, lg.data(), nullptr);
        out.push_back(lg);
        if (margins) margins->push_back(s.margin);
    }
    return out;
}
// A routing margin under this is one a quantized run can flip (the engine's activations are a percent or two off the reference's):
// from such a token on, the two runs' states differ by a different expert, not by rounding, and are not compared.
constexpr float kFlipMargin = 0.005f;   // (measured: margins of 0.000 and 0.004 parted the runs; 0.013 did not)

struct Result {
    std::vector<float> prompt_logits;
    std::vector<int32_t> gen;
    std::vector<float> gen_logits;
};
}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("nemotron_engine_parity\n");
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("  no CUDA device: skipped\n");
        return 77;
    }
    size_t fr = 0, tot = 0;
    cudaSetDevice(0);
    cudaMemGetInfo(&fr, &tot);
    if (fr < ((size_t) 1 << 30)) {
        std::printf("  GPU 0 has %zu MB free (1 GB needed): skipped\n", fr >> 20);
        return 77;
    }
    const Geometry g = geometry();
    R::Model m = R::random_model(g, 2024);
    // a router with a real model's spread (the random one's logits are ~N(0, 1): 16 experts' top-4 ties at almost every token)
    for (int64_t l = 0; l < g.n_blocks(); ++l)
        for (float& w : m.blocks[(size_t) l].router) w *= 3.0f;
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "strata-nemotron-engine-parity";
    std::filesystem::create_directories(dir);
    const std::filesystem::path gguf = dir / "tiny-nemotron.gguf";
    write_gguf(gguf, m);   // (m now holds the dequantized weights the file has)
    std::printf("  wrote %s (%.1f MB)\n", gguf.string().c_str(), std::filesystem::file_size(gguf) / 1e6);

    // a prompt of 45 tokens (two prompt chunks of 32 + 13), one <|im_start|> in it
    std::vector<int32_t> prompt;
    for (int i = 0; i < 45; ++i) prompt.push_back(i == 30 ? 10 : (int32_t) (20 + (i * 37) % 480));
    const int n_gen = 12;
    const size_t V = (size_t) g.n_vocab;

    auto run = [&](int slots, int mtp, const char* name, bool want_logits, int chunk = 32, double pcie = 0, int mtp_slots = 0) {
        N::EngineOptions o;
        o.ctx = 512;
        o.chunk = chunk;
        o.mtp = mtp;
        o.mtp_slots = mtp_slots;
        o.expert_slots = slots;
        o.max_gpus = 1;
        o.warm = false;
        o.adapt = false;
        o.verbose = false;
        o.reserve_mib = 0;
        o.pcie_frac = pcie;   // (the CPU path; the PCIe share has its own case)
        std::string err;
        std::unique_ptr<N::Engine> e = N::Engine::load(gguf.string(), o, err);
        Result r;
        if (!e) {
            check(false, std::string(name) + ": load: " + err);
            return r;
        }
        e->feed(prompt, &r.prompt_logits);
        r.gen = e->greedy(argmax(r.prompt_logits.data(), V), n_gen, want_logits ? &r.gen_logits : nullptr);
        return r;
    };
    // the reference: the prompt, then teacher-forced on the engine's own tokens
    const std::vector<std::vector<float>> rp = ref_logits(m, prompt);
    {
        // the prompt read in windows of 8 rows: the decode path's kernels on every token
        const Result r = run(6, 0, "decode path", false, 8);
        if (!r.prompt_logits.empty()) {
            const float ep = rel_l2(r.prompt_logits.data(), rp.back().data(), V);
            char buf[200];
            std::snprintf(buf, sizeof buf, "the prompt in windows of 8 rows (the decode path) vs the reference: rel. L2 %.3f", (double) ep);
            check(ep < 0.08f, buf);
        }
        // the prompt path for 40 tokens, then one token at a time (each a held-state continuation of one row)
        N::EngineOptions o;
        o.ctx = 512;
        o.chunk = 32;
        o.expert_slots = 6;
        o.max_gpus = 1;
        o.warm = false;
        o.adapt = false;
        o.verbose = false;
        o.reserve_mib = 0;
        o.pcie_frac = 0;   // (the CPU path; the PCIe share has its own case)
        std::string err;
        std::unique_ptr<N::Engine> e = N::Engine::load(gguf.string(), o, err);
        if (e) {
            std::vector<float> lg;
            float worst = 0;
            std::string steps;
            for (size_t n = 40; n <= prompt.size(); ++n) {
                const int64_t rd = e->feed(std::vector<int32_t>(prompt.begin(), prompt.begin() + (long) n), &lg);
                const float er = rel_l2(lg.data(), rp[n - 1].data(), V);
                worst = std::max(worst, er);
                char b[48];
                std::snprintf(b, sizeof b, " %lld:%.3f", (long long) rd, (double) er);
                steps += b;
            }
            check(worst < 0.08f, "one row at a time after a prompt (read:rel. L2)" + steps);
            // greedy() against the same tokens read one at a time through feed()
            std::unique_ptr<N::Engine> e2 = N::Engine::load(gguf.string(), o, err);
            std::vector<float> l0, gl;
            e2->feed(prompt, &l0);
            const int32_t first = argmax(l0.data(), V);
            const std::vector<int32_t> gen = e2->greedy(first, 6, &gl);
            std::vector<int32_t> seq = prompt;
            seq.push_back(first);
            std::string s2;
            float w2 = 0;
            bool tied = false;
            std::vector<float> mg;
            const std::vector<std::vector<float>> rq = ref_logits(m, [&] {
                std::vector<int32_t> q = prompt;
                q.push_back(first);
                q.insert(q.end(), gen.begin(), gen.end());
                return q;
            }(), &mg);
            for (int i = 0; i < 6; ++i) {
                e->feed(seq, &lg);
                const float a = rel_l2(gl.data() + (size_t) i * V, lg.data(), V);
                const float b = rel_l2(lg.data(), rq[prompt.size() + (size_t) i].data(), V);
                if (mg[prompt.size() + (size_t) i] < kFlipMargin) tied = true;   // (from a routing tie on, the runs may part)
                if (!tied) w2 = std::max(w2, a);
                char bb[80];
                std::snprintf(bb, sizeof bb, " %.3f/%.3f(m%.3f)", (double) a, (double) b, (double) mg[prompt.size() + (size_t) i]);
                s2 += bb;
                seq.push_back(gen[(size_t) i]);
            }
            // (the two engines read the prompt differently - one chunk of 45 against 40 + five single rows - so their states differ by
            // rounding, as each does from the reference; the reference's margin says where a routing tie lets runs part)
            check(w2 < 0.06f, "greedy() vs feed() of its tokens, 6 steps (engine-vs-engine/feed-vs-ref(ref margin))" + s2);
        }
    }
    struct Cfg {
        int slots;
        const char* name;
        double pcie = 0;
    };
    for (const Cfg c : {Cfg{0, "experts on the CPU"}, Cfg{16, "experts on the GPU"}, Cfg{6, "experts split 6 GPU / 10 CPU"},
                        Cfg{6, "split, half the misses copied over PCIe", 0.5}, Cfg{0, "no slots, every miss over PCIe", 1.0}}) {
        const Result r = run(c.slots, 0, c.name, true, 32, c.pcie);
        if (r.prompt_logits.empty()) continue;
        const float ep = rel_l2(r.prompt_logits.data(), rp.back().data(), V);
        char buf[220];
        std::snprintf(buf, sizeof buf, "%s: prompt (two chunks, the prompt path) logits vs the reference: rel. L2 %.3f, top-1 %s", c.name,
                      (double) ep, argmax(r.prompt_logits.data(), V) == argmax(rp.back().data(), V) ? "same" : "differs");
        check(ep < 0.08f, buf);
        // teacher-forced: the reference reads prompt + the engine's tokens; step i's logits are after token i was read
        std::vector<int32_t> seq = prompt;
        seq.push_back(argmax(r.prompt_logits.data(), V));
        for (int i = 0; i + 1 < n_gen; ++i) seq.push_back(r.gen[(size_t) i]);
        std::vector<float> margins;
        const std::vector<std::vector<float>> rs = ref_logits(m, seq, &margins);
        float worst = 0;
        int same = 0, n_cmp = 0;
        for (int i = 0; i < n_gen; ++i) {
            const size_t at = prompt.size() + (size_t) i;
            if (margins[at] < kFlipMargin) break;   // a routing tie: the runs may part from here on
            const float* el = r.gen_logits.data() + (size_t) i * V;
            worst = std::max(worst, rel_l2(el, rs[at].data(), V));
            {   // the same pick, or a logit tie in the reference (the engine's pick within 1% of the logit range of the best)
                const float* rl = rs[at].data();
                const int a = argmax(el, V), b = argmax(rl, V);
                const float lo = *std::min_element(rl, rl + V), hi = rl[b];
                same += a == b || rl[a] >= hi - 0.01f * (hi - lo);
            }
            ++n_cmp;
        }
        std::snprintf(buf, sizeof buf, "%s: decode steps vs the reference until the first routing tie (%d of %d): worst rel. L2 %.3f, "
                      "top-1 same in %d", c.name, n_cmp, n_gen, (double) worst, same);
        check(n_cmp >= 1 && worst < 0.08f && same == n_cmp, buf);
    }
    // MTP drafts: the same greedy tokens as without (the verify window's rows compute what single steps compute)
    {
        const Result plain = run(6, 0, "plain", false);
        const Result spec = run(6, 2, "MTP x2", false);
        const Result spec3 = run(16, 3, "MTP x3", false);
        const Result plain16 = run(16, 0, "plain 16", false);
        const Result again = run(6, 0, "plain again", false);
        const Result spec_pcie = run(6, 2, "MTP x2 + PCIe", false, 32, 0.5);
        const Result plain_pcie = run(6, 0, "plain + PCIe", false, 32, 0.5);
        check(!plain_pcie.gen.empty() && plain_pcie.gen == spec_pcie.gen, "MTP with 2 drafts and the PCIe share: the same greedy tokens as plain decoding with it");
        // the MTP block's experts in slots of their own (--mtp-slots): the drafts' experts on the GPU, the verified tokens the same
        const Result spec_ms = run(6, 2, "MTP x2 + MTP slots", false, 32, 0, 4);
        const Result spec_ms_all = run(6, 3, "MTP x3 + all MTP experts in slots", false, 32, 0, 1 << 20);
        check(!spec_ms.gen.empty() && spec_ms.gen == plain.gen, "MTP with 2 drafts and 4 MTP slots: the same greedy tokens as plain decoding");
        check(!spec_ms_all.gen.empty() && spec_ms_all.gen == plain.gen, "MTP with 3 drafts and every MTP expert in a slot: the same greedy tokens");
        check(!again.gen.empty() && again.prompt_logits == plain.prompt_logits && again.gen == plain.gen,
              "the same engine settings twice: bit-identical prompt logits and greedy tokens (deterministic)");
        if (spec.prompt_logits != plain.prompt_logits) std::printf("    the MTP x2 run's prompt logits differ from the plain run's\n");
        auto show = [](const char* what, const std::vector<int32_t>& v) {
            std::printf("    %s:", what);
            for (int32_t t : v) std::printf(" %d", t);
            std::printf("\n");
        };
        if (plain.gen != spec.gen) { show("plain", plain.gen); show("mtp 2", spec.gen); }
        if (plain16.gen != spec3.gen) { show("plain 16", plain16.gen); show("mtp 3", spec3.gen); }
        check(!plain.gen.empty() && plain.gen == spec.gen, "MTP with 2 drafts (experts split): the same greedy tokens as plain decoding");
        check(!plain16.gen.empty() && plain16.gen == spec3.gen, "MTP with 3 drafts (experts on the GPU): the same greedy tokens as plain decoding");
    }
    // the prefill pipeline (two GPUs, the layers split): each GPU reads a different chunk at the same time, and the prompt's logits and
    // the greedy tokens are those of one chunk through both GPUs at a time, bit for bit
    {
        int n_dev = 0;
        if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 2) {
            std::printf("  the prefill pipeline: one GPU only, skipped\n");
        } else {
            auto run2 = [&](bool pipe, int chunk, int mtp) {
                N::EngineOptions o;
                o.ctx = 512;
                o.chunk = chunk;
                o.mtp = mtp;
                o.expert_slots = 6;
                o.max_gpus = 2;
                o.warm = false;
                o.adapt = false;
                o.verbose = false;
                o.reserve_mib = 0;
                o.pcie_frac = 0;
                o.prefill_pipe = pipe;
                std::string err;
                std::unique_ptr<N::Engine> e = N::Engine::load(gguf.string(), o, err);
                Result r;
                if (!e) {
                    check(false, "two GPUs: load: " + err);
                    return r;
                }
                e->feed(prompt, &r.prompt_logits);
                r.gen = e->greedy(argmax(r.prompt_logits.data(), V), n_gen);
                return r;
            };
            // chunks of 16: [0,16) [16,30) in the pipeline (the checkpoint at 30 ends it), [30,45) alone; of 9: [0,9) [9,18) [18,27)
            // in it, then [27,30) (3 rows: the decode path) and the rest one at a time
            for (const int chunk : {16, 9}) {
                const Result seq = run2(false, chunk, 0), pip = run2(true, chunk, 0), pip_mtp = run2(true, chunk, 2);
                char buf[200];
                std::snprintf(buf, sizeof buf, "two GPUs, chunks of %d: the prefill pipeline's prompt logits and greedy tokens are those of one "
                              "chunk at a time (bit for bit)", chunk);
                check(!seq.gen.empty() && pip.prompt_logits == seq.prompt_logits && pip.gen == seq.gen, buf);
                std::snprintf(buf, sizeof buf, "two GPUs, chunks of %d: the pipeline with 2 MTP drafts, the same greedy tokens", chunk);
                check(!pip_mtp.gen.empty() && pip_mtp.gen == seq.gen, buf);
            }
        }
    }
    // a conversation that continues: the held state is reused, and its logits are those of a fresh read
    {
        N::EngineOptions o;
        o.ctx = 512;
        o.chunk = 32;
        o.expert_slots = 6;
        o.max_gpus = 1;
        o.warm = false;
        o.adapt = false;
        o.verbose = false;
        o.reserve_mib = 0;
        o.pcie_frac = 0;   // (the CPU path; the PCIe share has its own case)
        std::string err;
        std::unique_ptr<N::Engine> e = N::Engine::load(gguf.string(), o, err);
        if (!e) {
            check(false, "the conversation engine: load: " + err);
            std::printf("FAILED: %d\n", g_fail);
            return 1;
        }
        std::vector<float> lg;
        e->feed(prompt, &lg);
        const int32_t first = argmax(lg.data(), V);
        const std::vector<int32_t> gen = e->greedy(first, 6);
        std::vector<int32_t> next = prompt;
        next.push_back(first);
        next.insert(next.end(), gen.begin(), gen.end() - 1);   // the last token was picked, not read
        for (int i = 0; i < 9; ++i) next.push_back((int32_t) (100 + i * 13));
        std::vector<float> lg2;
        const int64_t read = e->feed(next, &lg2);
        check(read == 9, "a continued conversation reads only its 9 new tokens (read " + std::to_string(read) + ")");
        const std::vector<std::vector<float>> rn = ref_logits(m, next);
        const float ec = rel_l2(lg2.data(), rn.back().data(), V);
        char buf[160];
        std::snprintf(buf, sizeof buf, "... and its logits match the reference's fresh read (rel. L2 %.3f)", (double) ec);
        check(ec < 0.08f, buf);
        // a re-rendered turn (a token changed after the <|im_start|> at 30): the checkpoint there is restored, 15+ tokens read
        std::vector<int32_t> other = prompt;
        other[40] = 7;
        std::vector<float> lg3;
        const int64_t read3 = e->feed(other, &lg3);
        const std::vector<std::vector<float>> ro = ref_logits(m, other);
        check(read3 == (int64_t) (other.size() - 30), "a changed tail resumes from the checkpoint at the last <|im_start|> (read " +
                                                         std::to_string(read3) + ")");
        std::snprintf(buf, sizeof buf, "... and its logits match the reference's (rel. L2 %.3f)", (double) rel_l2(lg3.data(), ro.back().data(), V));
        check(rel_l2(lg3.data(), ro.back().data(), V) < 0.08f, buf);
    }
    // two conversations in turn: B writes over A's cells; A then continues.  Parked (in RAM with its cells' K / V): only A's new tokens
    // are read.  Not parked: A's checkpoints no longer match the cells, so A is read again (never from a checkpoint B wrote over).
    for (const bool parking : {true, false}) {
        N::EngineOptions o;
        o.ctx = 512;
        o.chunk = 32;
        o.expert_slots = 6;
        o.max_gpus = 1;
        o.warm = false;
        o.adapt = false;
        o.verbose = false;
        o.reserve_mib = 0;
        o.pcie_frac = 0;
        o.park_min = parking ? 8 : 1 << 30;
        o.park_mib = parking ? 64 : 0;
        std::string err;
        std::unique_ptr<N::Engine> e = N::Engine::load(gguf.string(), o, err), fresh = N::Engine::load(gguf.string(), o, err);
        if (!e || !fresh) {
            check(false, "the two-conversation engines: load: " + err);
            continue;
        }
        std::vector<int32_t> A = prompt, B(prompt.begin(), prompt.begin() + 10);
        for (int i = 0; i < 60; ++i) B.push_back((int32_t) (30 + (i * 53) % 470));
        std::vector<float> la, lb, la2, lf;
        e->feed(A, &la);
        e->feed(B, &lb);
        std::vector<int32_t> A2 = A;
        for (int i = 0; i < 5; ++i) A2.push_back((int32_t) (300 + i * 7));
        const int64_t read = e->feed(A2, &la2);
        // the reference reads the same chunks without B between (A, then A2's 5 tokens): parking must change nothing; without parking
        // A2 is read again whole, so its reference reads it whole
        if (parking) fresh->feed(A, &lf);
        fresh->feed(A2, &lf);
        const float d = rel_l2(la2.data(), lf.data(), V);
        char buf[200];
        if (parking) {
            std::snprintf(buf, sizeof buf, "A, then B over its cells, then A again: parked, it reads only its 5 new tokens (read %lld), "
                          "logits vs the same reads without B rel. L2 %.4f", (long long) read, (double) d);
            check(read == 5 && d == 0.0f, buf);
        } else {
            std::snprintf(buf, sizeof buf, "... not parked: A is read again from where its cells still hold it (read %lld of %zu), "
                          "logits vs a fresh read rel. L2 %.4f", (long long) read, A2.size(), (double) d);
            check(read > 5 && d < 0.02f, buf);
        }
    }
    // pictures (GENI): rows in place of <image> cells.  A "picture" whose rows are the token embeddings of 12 prompt tokens must read
    // exactly as those tokens (the same arithmetic after the embedding), in the prompt path and in the MTP block's fill; the held state
    // is reused for the same picture and not for another one.
    {
        N::EngineOptions o;
        o.ctx = 512;
        o.chunk = 16;   // chunks [0,16) [16,30) [30,45): the picture's run crosses the first boundary
        o.expert_slots = 6;
        o.max_gpus = 1;
        o.warm = false;
        o.adapt = false;
        o.verbose = false;
        o.reserve_mib = 0;
        o.pcie_frac = 0;   // (the CPU path; the PCIe share has its own case)
        o.mtp = 2;
        std::string err;
        std::unique_ptr<N::Engine> e = N::Engine::load(gguf.string(), o, err);
        if (!e) {
            check(false, "the picture engine: load: " + err);
            std::printf("FAILED: %d\n", g_fail);
            return 1;
        }
        const int64_t Ne = e->n_embd();
        const int64_t a = 10, n_img = 12;   // cells 10 .. 21 (the <|im_start|> at 30 stays: both reads are chunked alike)
        std::vector<int32_t> under(prompt.begin() + a, prompt.begin() + a + n_img);
        const std::vector<float> img = e->token_rows(under);
        std::vector<float> lt, li;
        e->reset();
        e->feed(prompt, &lt);
        const int32_t ft = argmax(lt.data(), V);
        const std::vector<int32_t> gt = e->greedy(ft, 8);
        // the same prompt with the picture: identity ids in its cells, the rows from `img`
        std::vector<int32_t> pic = prompt;
        std::vector<const float*> rows(pic.size(), nullptr);
        for (int64_t j = 0; j < n_img; ++j) {
            pic[(size_t) (a + j)] = N::image_cell_id(0x1234, j);
            rows[(size_t) (a + j)] = img.data() + (size_t) j * (size_t) Ne;
        }
        e->reset();
        const int64_t read_i = e->feed(pic, rows, &li);
        const int32_t fi = argmax(li.data(), V);
        const std::vector<int32_t> gi = e->greedy(fi, 8);
        check(read_i == (int64_t) pic.size() && lt == li, "a picture whose rows are token embeddings: bit-identical prompt logits");
        check(ft == fi && gt == gi, "... and the same greedy tokens with 2 MTP drafts (the MTP block reads the picture's rows)");
        // the next turn with the same picture continues the held state; one with another picture (other identity ids) does not
        std::vector<int32_t> next = pic;
        next.push_back(fi);
        next.insert(next.end(), gi.begin(), gi.end() - 1);
        for (int i = 0; i < 5; ++i) next.push_back((int32_t) (200 + i * 11));
        std::vector<const float*> rows2 = rows;
        rows2.resize(next.size(), nullptr);
        std::vector<float> l2;
        const int64_t read2 = e->feed(next, rows2, &l2);
        check(read2 == 5, "the next turn with the same picture reads only its 5 new tokens (read " + std::to_string(read2) + ")");
        // a fresh read of that turn, then the same tokens with another picture (other identity ids, the same floats): no held state or
        // checkpoint starts with it (they all hold the first picture), so all of it is read again - with the same logits as the fresh read
        std::vector<float> lf;
        e->reset();
        e->feed(next, rows2, &lf);
        std::vector<int32_t> other = next;
        for (int64_t j = 0; j < n_img; ++j) other[(size_t) (a + j)] = N::image_cell_id(0x5678, j);
        std::vector<float> l3;
        const int64_t read3 = e->feed(other, rows2, &l3);
        check(read3 == (int64_t) other.size(), "another picture in the same place: nothing held is reused (read " + std::to_string(read3) +
                                                   " of " + std::to_string(other.size()) + ")");
        check(l3 == lf, "... and, its rows being the same floats, its logits are those of the fresh read");
    }
    // STRATA_NEMO_KEEP_GGUF=<path>: the synthetic model is kept there (the protocol test, tests/nemotron/test_proto.py, serves it)
    std::error_code ec;
    if (const char* keep = std::getenv("STRATA_NEMO_KEEP_GGUF")) {
        std::filesystem::copy_file(gguf, keep, std::filesystem::copy_options::overwrite_existing, ec);
        std::printf("  kept the synthetic model at %s%s\n", keep, ec ? " (copy failed)" : "");
    }
    std::filesystem::remove_all(dir, ec);
    std::printf(g_fail ? "FAILED: %d\n" : "all ok\n", g_fail);
    return g_fail ? 1 : 0;
}
