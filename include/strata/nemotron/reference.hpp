// include/strata/nemotron/reference.hpp - the fp32 CPU reference of NVIDIA-Nemotron-3.5-Super-VL (nemotron_h_moe).
//
// One token at a time, every weight dequantized to fp32, no SIMD, no threads: slow and exact by construction, the oracle that the
// CUDA kernels and the hybrid CPU expert path of src/nemotron are checked against.  It follows llama.cpp's graph at the pin
// 3cf03257 (src/models/nemotron-h.cpp, nemotron-h-moe.cpp, mamba-base.cpp; ggml-cpu ssm_conv / ssm_scan), which is what the
// GGUF was made for:
//
//   block l:  x += mixer_l(rms(x) * attn_norm_l)            (every block, also the MoE ones, uses attn_norm)
//     Mamba2: zxBCdt = ssm_in h;  xBC = silu(conv1d(xBC) + b);  dt' = softplus(dt + dt_bias);  s = s*exp(dt'*A) + B (x dt');
//             y = s.C + D x;  y *= silu(z);  y = grouped_rms(y) * ssm_norm;  out = ssm_out y
//     attention: GQA, NO rotary embedding (NoPE), no q/k norm, scale 1/sqrt(head_dim), causal
//     MoE (LatentMoE): p = sigmoid(router h); top-k by p + exp_probs_b; w = p[top-k] / sum, * expert_weights_scale;
//             lat = latent_down h;  moe = latent_up sum_k w_k down_k relu(up_k lat)^2;  shared = down_sh relu(up_sh h)^2
//   head:     hf = rms(x) * output_norm;  logits = output hf;  h_nextn = hf (the MTP block's seed)
//   MTP:      x = eh_proj [rms(embd[t]) * enorm, rms(h) * hnorm];  x += attn(rms(x) * attn_norm);
//             x += moe(rms(x) * post_attention_norm);  h' = layernorm(x) * shared_head_norm;  logits = output h'
//
// Weight layout is the GGUF's: a matrix [n_in, n_out] is n_out rows of n_in floats (y[o] = sum_i W[o * n_in + i] x[i]).
#pragma once

#include "strata/nemotron/geometry.hpp"

#include <cstdint>
#include <vector>

namespace strata::nemotron::ref {

/// One block's weights, fp32.  Only the members of the block's kind are filled.
struct Block {
    std::vector<float> attn_norm;
    // Mamba2
    std::vector<float> ssm_in, conv_w, conv_b, dt_b, A, D, ssm_norm, ssm_out;
    // attention (also the MTP block)
    std::vector<float> wq, wk, wv, wo;
    // MoE (also the MTP block)
    std::vector<float> router, probs_b, latent_down, latent_up, up_exps, down_exps, up_sh, down_sh;
    // the MTP block
    std::vector<float> post_norm, eh_proj, enorm, hnorm, shared_head_norm;
};

struct Model {
    Geometry g;
    std::vector<Block> blocks;   ///< main blocks then the MTP block(s)
    std::vector<float> tok_embd, output_norm, output;
};

/// The recurrent and attention state of one sequence (the main blocks).
struct State {
    std::vector<std::vector<float>> conv;   ///< per block: (d_conv - 1) x conv channels, oldest first; empty for non-Mamba blocks
    std::vector<std::vector<float>> ssm;    ///< per block: n_head x head_dim x d_state; empty for non-Mamba blocks
    std::vector<std::vector<float>> k, v;   ///< per block: pos x n_head_kv x head_dim; empty for non-attention blocks
    int64_t pos = 0;
    /// The last token's smallest routing margin over the MoE blocks: selection score K-th minus (K+1)-th.  A small one means a
    /// quantized run may route that token differently (a flipped expert changes the output by far more than its rounding).
    float margin = 0.0f;
};

/// The MTP block's own state: its attention K / V (one row per token it has read).
struct MtpState {
    std::vector<float> k, v;
    int64_t pos = 0;
};

State new_state(const Model& m);
MtpState new_mtp_state(const Model& m);

// ---- the pieces (also used by the tests one by one)
float softplus(float x);   ///< ggml's: x > 20 ? x : log(1 + exp(x))
void rms_norm(const float* x, const float* w, int64_t n, float eps, float* out);
void layer_norm(const float* x, const float* w, int64_t n, float eps, float* out);   ///< no bias
void matvec(const float* W, const float* x, int64_t n_in, int64_t n_out, float* y);

/// The Mamba2 mixer of block `l` for one token (input already normed).
void mamba2(const Model& m, int64_t l, State& s, const float* h, float* out);
/// The attention mixer of block `l` (or the MTP block, with its K / V in `k`, `v`) for one token at position `pos`.
void attention(const Model& m, int64_t l, std::vector<float>& k, std::vector<float>& v, int64_t pos, const float* h, float* out);
/// The LatentMoE feed-forward of block `l` for one token; `picked` / `weights` (n_expert_used each, optional) get the routing.
void moe(const Model& m, int64_t l, const float* h, float* out, int32_t* picked = nullptr, float* weights = nullptr,
         float* margin = nullptr);

/// One token through the main blocks.  `logits` (n_vocab) and `h_nextn` (n_embd, the MTP seed) are optional.
void forward(const Model& m, State& s, int32_t token, float* logits, float* h_nextn);
/// One MTP step: `token` is the token after the position whose seed is `h`.  `logits` and `h_out` (the next step's seed) optional.
void mtp(const Model& m, MtpState& s, int32_t token, const float* h, float* logits, float* h_out);

/// A model of geometry `g` with random weights (deterministic for `seed`), sized so activations stay O(1): for tests.
Model random_model(const Geometry& g, uint64_t seed);
/// A small geometry with the real model's structure (Mamba2 / attention / MoE blocks + the MTP block), for tests.
Geometry tiny_geometry();

}  // namespace strata::nemotron::ref
