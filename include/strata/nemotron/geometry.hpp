// include/strata/nemotron/geometry.hpp - NVIDIA-Nemotron-3.5-Super-VL-120B-A12B (GGUF architecture nemotron_h_moe):
// the geometry read from the metadata, the per-block layer kinds, the tensor layout table and its check.
//
// The model is 88 blocks + 1 MTP block, each block ONE mixer or ONE feed-forward (no block has both, except the MTP block):
//   * Mamba2   (ssm_in / ssm_conv1d / ssm_a / ssm_d / ssm_dt / ssm_norm / ssm_out)   - no per-block feed-forward
//   * Attention (attn_q / k / v / output, GQA 32 : 2, no q/k norm, no gate)          - no per-block feed-forward
//   * MoE      (LatentMoE: router -> latent_down -> 512 routed relu^2 experts -> latent_up, plus one shared expert)
// The kind of block l is read from the two per-block lists of the metadata: attention.head_count_kv[l] > 0 = attention,
// feed_forward_length[l] > 0 = MoE, both 0 = Mamba2.  The MTP block (index n_layers) is attention + MoE.
//
// Everything here is header-only and has no GPU dependency: the C++ tests, the CPU reference and the CUDA engine share it.
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_family.hpp"

#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace strata::nemotron {

enum class BlockKind { Mamba, Attention, Moe, MtpAttentionMoe };

struct Geometry {
    int64_t n_layers = 0;        ///< the main blocks (block_count - nextn_predict_layers)
    int64_t n_mtp = 0;           ///< blocks after them that are the in-file MTP block(s)
    int64_t n_embd = 0;
    int64_t n_vocab = 0;
    // attention
    int64_t n_head = 0, n_head_kv = 0, head_dim = 0, n_rot = 0;
    // MoE
    int64_t n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0, moe_latent = 0;
    float expert_weights_scale = 1.0f;
    bool expert_weights_norm = false;
    // Mamba2
    int64_t ssm_d_conv = 0, ssm_d_state = 0, ssm_n_group = 0, ssm_n_head = 0, ssm_d_inner = 0;
    float rms_eps = 1e-5f;
    std::vector<BlockKind> kind;   ///< one per block, main blocks then MTP blocks

    int64_t n_blocks() const { return n_layers + n_mtp; }
    int64_t ssm_head_dim() const { return ssm_d_inner / ssm_n_head; }
    int64_t ssm_conv_dim() const { return ssm_d_inner + 2 * ssm_n_group * ssm_d_state; }          ///< x, B, C
    int64_t ssm_in_dim() const { return 2 * ssm_d_inner + 2 * ssm_n_group * ssm_d_state + ssm_n_head; }  ///< z, x, B, C, dt
    int64_t n_of(BlockKind k) const {
        int64_t n = 0;
        for (BlockKind x : kind) n += x == k;
        return n;
    }
    bool has_attention(int64_t l) const { return kind[(size_t) l] == BlockKind::Attention || kind[(size_t) l] == BlockKind::MtpAttentionMoe; }
    bool has_moe(int64_t l) const { return kind[(size_t) l] == BlockKind::Moe || kind[(size_t) l] == BlockKind::MtpAttentionMoe; }
    /// The layer string the model card style uses: M = Mamba2, * = attention, E = MoE, X = the MTP block.
    std::string pattern() const {
        std::string s;
        for (BlockKind k : kind) s += k == BlockKind::Mamba ? 'M' : k == BlockKind::Attention ? '*' : k == BlockKind::Moe ? 'E' : 'X';
        return s;
    }
};

/// Fills `g` from the metadata shard `meta`.  Returns false with the first problem in `err`.
inline bool geometry_from_gguf(const strata::GgufFile& meta, Geometry& g, std::string& err) {
    err = strata::check_architecture(meta);
    if (!err.empty()) return false;
    if (strata::model_family(meta) != strata::ModelFamily::NemotronHMoe) {
        err = "not a nemotron_h_moe file";
        return false;
    }
    g = Geometry{};
    const std::string p = "nemotron_h_moe.";
    auto num = [&](const char* key, int64_t& dst) {
        if (const strata::MetaValue* v = meta.get(p + key))
            if (v->is_num()) dst = (int64_t) v->u;
    };
    int64_t blocks = 0, mtp = 0, key_len = 0, val_len = 0, used = 0;
    num("block_count", blocks);
    num("nextn_predict_layers", mtp);
    num("embedding_length", g.n_embd);
    num("attention.head_count", g.n_head);
    num("attention.key_length", key_len);
    num("attention.value_length", val_len);
    num("rope.dimension_count", g.n_rot);
    num("expert_count", g.n_expert);
    num("expert_used_count", used);
    num("expert_feed_forward_length", g.n_ff_exp);
    num("expert_shared_feed_forward_length", g.n_ff_shexp);
    num("moe_latent_size", g.moe_latent);
    num("ssm.conv_kernel", g.ssm_d_conv);
    num("ssm.state_size", g.ssm_d_state);
    num("ssm.group_count", g.ssm_n_group);
    num("ssm.time_step_rank", g.ssm_n_head);
    num("ssm.inner_size", g.ssm_d_inner);
    g.n_expert_used = used;
    g.head_dim = key_len;
    g.n_layers = blocks - mtp;
    g.n_mtp = mtp;
    if (const strata::MetaValue* v = meta.get(p + "attention.layer_norm_epsilon"))
        if (v->is_num() && v->num() > 0) g.rms_eps = (float) v->num();
    if (const strata::MetaValue* v = meta.get(p + "expert_weights_scale"))
        if (v->is_num()) g.expert_weights_scale = (float) v->num();
    if (const strata::MetaValue* v = meta.get(p + "expert_weights_norm")) g.expert_weights_norm = v->is_num() && v->u != 0;
    if (const strata::MetaValue* v = meta.get("tokenizer.ggml.tokens"))
        if (v->type == strata::MetaType::ARRAY && v->count > 0) g.n_vocab = (int64_t) v->count;
    if (g.n_vocab == 0) num("vocab_size", g.n_vocab);

    char buf[256];
    auto bad = [&](const char* what) {
        std::snprintf(buf, sizeof buf, "%s: %s", meta.path().c_str(), what);
        err = buf;
        return false;
    };
    if (g.n_layers <= 0 || g.n_embd <= 0 || g.n_head <= 0 || g.head_dim <= 0 || g.n_expert <= 0 || used <= 0 || used > g.n_expert ||
        g.n_ff_exp <= 0 || g.n_ff_shexp <= 0 || g.moe_latent <= 0 || g.ssm_d_conv <= 0 || g.ssm_d_state <= 0 || g.ssm_n_group <= 0 ||
        g.ssm_n_head <= 0 || g.ssm_d_inner <= 0 || g.n_vocab <= 0)
        return bad("a geometry key is zero or out of range");
    if (val_len != key_len) return bad("attention.key_length != attention.value_length");
    if (g.ssm_d_inner % g.ssm_n_head) return bad("ssm.inner_size is not a multiple of ssm.time_step_rank");
    if (g.ssm_d_inner % g.ssm_n_group) return bad("ssm.inner_size is not a multiple of ssm.group_count");
    if (g.ssm_n_head % g.ssm_n_group) return bad("ssm.time_step_rank is not a multiple of ssm.group_count");
    if (g.n_rot <= 0 || g.n_rot > g.head_dim || g.n_rot % 2) g.n_rot = g.head_dim;   // the key is optional; rotate the whole head then

    // the per-block lists
    const strata::MetaValue* kv = meta.get(p + "attention.head_count_kv");
    const strata::MetaValue* ff = meta.get(p + "feed_forward_length");
    g.kind.assign((size_t) blocks, BlockKind::Mamba);
    for (int64_t l = 0; l < blocks; ++l) {
        const int64_t a = (int64_t) kv->items[(size_t) l].u, f = (int64_t) ff->items[(size_t) l].u;
        const bool mtp_block = l >= g.n_layers;
        if (a > 0) {
            if (g.n_head_kv == 0) g.n_head_kv = a;
            if (a != g.n_head_kv) return bad("attention.head_count_kv differs between the attention blocks");
        }
        if (mtp_block) {
            if (a <= 0 || f <= 0) return bad("the MTP block is expected to be attention + MoE");
            g.kind[(size_t) l] = BlockKind::MtpAttentionMoe;
        } else if (a > 0 && f > 0) {
            return bad("a main block has both attention and a feed-forward");
        } else {
            g.kind[(size_t) l] = a > 0 ? BlockKind::Attention : f > 0 ? BlockKind::Moe : BlockKind::Mamba;
        }
        if (f > 0 && f != g.n_ff_exp) return bad("feed_forward_length differs from expert_feed_forward_length");
    }
    if (g.n_head_kv <= 0 || g.n_head % g.n_head_kv) return bad("attention.head_count is not a multiple of attention.head_count_kv");
    return true;
}

struct TensorWant {
    std::string name;
    std::vector<int64_t> shape;   ///< GGUF order: dim 0 varies fastest
};

/// Every tensor of block `l` (name without the blk.N. prefix).
inline std::vector<TensorWant> layer_tensors(const Geometry& g, int64_t l) {
    const int64_t N = g.n_embd;
    std::vector<TensorWant> t;
    auto add = [&](const char* s, std::vector<int64_t> shape) { t.push_back({s, std::move(shape)}); };
    add("attn_norm.weight", {N});
    switch (g.kind[(size_t) l]) {
    case BlockKind::Mamba:
        add("ssm_in.weight", {N, g.ssm_in_dim()});
        add("ssm_conv1d.weight", {g.ssm_d_conv, g.ssm_conv_dim()});
        add("ssm_conv1d.bias", {g.ssm_conv_dim()});
        add("ssm_dt.bias", {g.ssm_n_head});
        add("ssm_a", {1, g.ssm_n_head});
        add("ssm_d", {1, g.ssm_n_head});
        add("ssm_norm.weight", {g.ssm_d_inner / g.ssm_n_group, g.ssm_n_group});
        add("ssm_out.weight", {g.ssm_d_inner, N});
        break;
    case BlockKind::Attention:
        add("attn_q.weight", {N, g.n_head * g.head_dim});
        add("attn_k.weight", {N, g.n_head_kv * g.head_dim});
        add("attn_v.weight", {N, g.n_head_kv * g.head_dim});
        add("attn_output.weight", {g.n_head * g.head_dim, N});
        break;
    case BlockKind::Moe:
    case BlockKind::MtpAttentionMoe:
        if (g.kind[(size_t) l] == BlockKind::MtpAttentionMoe) {
            add("attn_q.weight", {N, g.n_head * g.head_dim});
            add("attn_k.weight", {N, g.n_head_kv * g.head_dim});
            add("attn_v.weight", {N, g.n_head_kv * g.head_dim});
            add("attn_output.weight", {g.n_head * g.head_dim, N});
        }
        add("ffn_gate_inp.weight", {N, g.n_expert});
        add("exp_probs_b.bias", {g.n_expert});
        add("ffn_latent_down.weight", {N, g.moe_latent});
        add("ffn_latent_up.weight", {g.moe_latent, N});
        add("ffn_up_exps.weight", {g.moe_latent, g.n_ff_exp, g.n_expert});     // relu^2, no gate
        add("ffn_down_exps.weight", {g.n_ff_exp, g.moe_latent, g.n_expert});
        add("ffn_up_shexp.weight", {N, g.n_ff_shexp});
        add("ffn_down_shexp.weight", {g.n_ff_shexp, N});
        break;
    }
    if (g.kind[(size_t) l] == BlockKind::MtpAttentionMoe) {
        // the in-file MTP block: eh_proj over concat(enorm(embedding), hnorm(hidden)), then the block, then the shared head norm
        add("post_attention_norm.weight", {N});
        add("nextn.eh_proj.weight", {2 * N, N});
        add("nextn.enorm.weight", {N});
        add("nextn.hnorm.weight", {N});
        add("nextn.shared_head_norm.weight", {N});
    }
    return t;
}

inline std::vector<TensorWant> global_tensors(const Geometry& g) {
    return {{"token_embd.weight", {g.n_embd, g.n_vocab}}, {"output_norm.weight", {g.n_embd}}, {"output.weight", {g.n_embd, g.n_vocab}}};
}

/// Checks all shards of `m` against the layout table: every table tensor is present with its shape and inside its shard, and no
/// blk.* / global tensor of the file is outside the table.  `checked` counts the tensors found.
inline bool check_gguf_layout(const strata::GgufModel& m, const Geometry& g, std::string& err, size_t* checked = nullptr) {
    std::set<std::string> named;
    size_t n = 0;
    auto check = [&](const std::string& name, const std::vector<int64_t>& shape) {
        named.insert(name);
        size_t s = 0;
        const strata::TensorInfo* t = m.find(name, &s);
        if (!t) {
            err = "missing " + name;
            return false;
        }
        bool same = t->shape.size() == shape.size();
        for (size_t i = 0; same && i < shape.size(); ++i) same = t->shape[i] == (uint64_t) shape[i];
        if (!same) {
            std::string got, want;
            for (uint64_t d : t->shape) got += (got.empty() ? "" : ", ") + std::to_string(d);
            for (int64_t d : shape) want += (want.empty() ? "" : ", ") + std::to_string(d);
            err = name + " is [" + got + "], the Nemotron layout wants [" + want + "]";
            return false;
        }
        if (!m.in_bounds(*t, s)) {
            err = name + " (" + t->type_name() + ") has no known size inside " + m.shard(s).path();
            return false;
        }
        ++n;
        return true;
    };
    for (const TensorWant& w : global_tensors(g))
        if (!check(w.name, w.shape)) return false;
    for (int64_t l = 0; l < g.n_blocks(); ++l)
        for (const TensorWant& w : layer_tensors(g, l))
            if (!check("blk." + std::to_string(l) + "." + w.name, w.shape)) return false;
    for (size_t s = 0; s < m.size(); ++s)
        for (const strata::TensorInfo& t : m.shard(s).tensors())
            if (!named.count(t.name)) {
                err = t.name + " is not in the Nemotron layout";
                return false;
            }
    if (checked) *checked = n;
    return true;
}

}  // namespace strata::nemotron
