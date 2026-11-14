// tests/core/nemotron_geometry_test.cpp - NVIDIA-Nemotron-3.5-Super-VL (nemotron_h_moe): the architecture guard, the geometry from
// metadata, the block kinds and the layout table, on synthetic GGUFs (no model, no GPU).
//
// `nemotron_geometry_test --real SHARD`: the same view of a real model (headers only, no weights are read): geometry, the layer
// pattern, the tensor-type census and the layout check over every shard.
#include "gguf_fixture.hpp"

#include "strata/core/family_layout.hpp"
#include "strata/nemotron/geometry.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using fixture::Kv;
using fixture::Tensor;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-nemotron-geometry-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

// A tiny model: 6 main blocks  M E M * E M  + the MTP block (attention + MoE).  Every tensor F32.
constexpr uint64_t kBlocks = 7, kEmbd = 64, kHeads = 4, kHeadsKv = 2, kKey = 16, kExperts = 8, kUsed = 2, kFf = 32, kFfSh = 48,
                   kLatent = 16, kConv = 4, kState = 8, kGroups = 2, kSsmHeads = 4, kInner = 32, kVocab = 20;

std::vector<Kv> metadata(const char* arch = "nemotron_h_moe") {
    const std::string p = std::string(arch) + ".";
    std::vector<Kv> kv = {fixture::str("general.architecture", arch),
                          fixture::u32(p + "block_count", kBlocks),
                          fixture::u32(p + "embedding_length", kEmbd),
                          fixture::u32(p + "attention.head_count", kHeads),
                          fixture::u32s(p + "attention.head_count_kv", {0, 0, 0, kHeadsKv, 0, 0, kHeadsKv}),
                          fixture::f32(p + "attention.layer_norm_epsilon", 1e-5f),
                          fixture::u32(p + "attention.key_length", kKey),
                          fixture::u32(p + "attention.value_length", kKey),
                          fixture::u32(p + "rope.dimension_count", kKey),
                          fixture::u32s(p + "feed_forward_length", {0, kFf, 0, 0, kFf, 0, kFf}),
                          fixture::u32(p + "expert_count", kExperts),
                          fixture::u32(p + "expert_used_count", kUsed),
                          fixture::u32(p + "expert_feed_forward_length", kFf),
                          fixture::u32(p + "expert_shared_feed_forward_length", kFfSh),
                          fixture::f32(p + "expert_weights_scale", 5.0f),
                          fixture::u32(p + "moe_latent_size", kLatent),
                          fixture::u32(p + "nextn_predict_layers", 1),
                          fixture::u32(p + "ssm.conv_kernel", kConv),
                          fixture::u32(p + "ssm.state_size", kState),
                          fixture::u32(p + "ssm.group_count", kGroups),
                          fixture::u32(p + "ssm.time_step_rank", kSsmHeads),
                          fixture::u32(p + "ssm.inner_size", kInner)};
    std::vector<std::string> tokens;
    for (uint64_t i = 0; i < kVocab; ++i) tokens.push_back("t" + std::to_string(i));
    kv.push_back(fixture::str_array("tokenizer.ggml.tokens", tokens));
    return kv;
}

std::vector<Tensor> tensors_of(const strata::nemotron::Geometry& g) {
    std::vector<Tensor> ts;
    uint8_t seed = 1;
    auto add = [&](const std::string& name, const std::vector<int64_t>& shape) {
        ts.push_back({name, std::vector<uint64_t>(shape.begin(), shape.end()), 0, seed++});
    };
    for (const auto& w : strata::nemotron::global_tensors(g)) add(w.name, w.shape);
    for (int64_t l = 0; l < g.n_blocks(); ++l)
        for (const auto& w : strata::nemotron::layer_tensors(g, l)) add("blk." + std::to_string(l) + "." + w.name, w.shape);
    return ts;
}

bool geometry_of(const fs::path& p, strata::nemotron::Geometry& g, std::string& err) {
    const strata::GgufFile f(p.string());
    return strata::nemotron::geometry_from_gguf(f, g, err);
}

std::string layout_error(const fs::path& dir, const strata::nemotron::Geometry& g, void (*mutate)(std::vector<Tensor>&)) {
    std::vector<Tensor> ts = tensors_of(g);
    if (mutate) mutate(ts);
    const fs::path model = dir / "tiny.gguf";
    fixture::write(model, metadata(), ts);
    const strata::GgufModel m({model.string()});
    std::string err;
    size_t n = 0;
    if (!strata::nemotron::check_gguf_layout(m, g, err, &n)) return err;
    return {};
}

int real_model(const std::string& any) {
    try {
        const strata::GgufModel m = strata::GgufModel::open(any);
        std::string err = strata::check_architecture(m.meta());
        std::printf("  %zu shard(s); architecture guard: %s\n", m.size(), err.empty() ? "ok" : err.c_str());
        if (!err.empty()) return 1;
        strata::nemotron::Geometry g;
        if (!strata::nemotron::geometry_from_gguf(m.meta(), g, err)) {
            std::printf("  geometry: %s\n", err.c_str());
            return 1;
        }
        std::printf("  blocks %lld (+%lld MTP): Mamba2 %lld, attention %lld, MoE %lld; hidden %lld, vocab %lld\n", (long long) g.n_layers,
                    (long long) g.n_mtp, (long long) g.n_of(strata::nemotron::BlockKind::Mamba),
                    (long long) g.n_of(strata::nemotron::BlockKind::Attention), (long long) g.n_of(strata::nemotron::BlockKind::Moe),
                    (long long) g.n_embd, (long long) g.n_vocab);
        std::printf("  attention %lld/%lld x %lld; MoE %lld experts top-%lld ff %lld shared %lld latent %lld scale %g norm %d; "
                    "Mamba2 inner %lld heads %lld (x%lld) groups %lld state %lld conv %lld\n",
                    (long long) g.n_head, (long long) g.n_head_kv, (long long) g.head_dim, (long long) g.n_expert,
                    (long long) g.n_expert_used, (long long) g.n_ff_exp, (long long) g.n_ff_shexp, (long long) g.moe_latent,
                    (double) g.expert_weights_scale, (int) g.expert_weights_norm, (long long) g.ssm_d_inner,
                    (long long) g.ssm_n_head, (long long) g.ssm_head_dim(), (long long) g.ssm_n_group, (long long) g.ssm_d_state,
                    (long long) g.ssm_d_conv);
        std::printf("  pattern: %s\n", g.pattern().c_str());
        std::map<std::string, int> census;
        for (size_t s = 0; s < m.size(); ++s)
            for (const strata::TensorInfo& t : m.shard(s).tensors())
                if (t.name.find("_exps.weight") != std::string::npos) ++census[std::string(t.type_name())];
        for (const auto& kv : census) std::printf("  expert tensors of type %s: %d\n", kv.first.c_str(), kv.second);
        size_t n = 0;
        const bool ok = strata::nemotron::check_gguf_layout(m, g, err, &n);
        std::printf("  layout: %s (%zu tensors checked)\n", ok ? "ok" : err.c_str(), n);
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("  refused: %s\n", e.what());
        return 1;
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--real") return real_model(argv[2]);
    std::printf("nemotron_geometry_test\n");
    TempDir d;
    const fs::path meta = d.path / "meta.gguf";
    fixture::write(meta, metadata(), {});
    {
        const strata::GgufFile f(meta.string());
        check(strata::model_family(f) == strata::ModelFamily::NemotronHMoe, "model_family: nemotron_h_moe");
        check(strata::check_architecture(f).empty(), "the architecture guard accepts a complete nemotron_h_moe file");
        check(!strata::check_engine_architecture(f).empty(), "the engine guard says there is no engine path yet (until src/nemotron)");
    }
    {
        // the Qwen reader never takes a Nemotron file for one of its own shapes
        const strata::GgufFile f(meta.string());
        strata::core::ModelGeometry qg;
        int64_t k = 0;
        std::string e;
        check(!strata::core::geometry_from_gguf(f, qg, k, e) && e.find("nemotron_h_moe") != std::string::npos,
              "the Qwen geometry reader refuses it: " + e);
    }
    strata::nemotron::Geometry g;
    std::string err;
    check(geometry_of(meta, g, err), "geometry_from_gguf: " + err);
    check(g.n_layers == 6 && g.n_mtp == 1 && g.n_blocks() == 7, "6 main blocks + 1 MTP block");
    check(g.pattern() == "MEM*EMX", "the layer pattern comes from the two per-block lists: " + g.pattern());
    check(g.n_head_kv == (int64_t) kHeadsKv && g.head_dim == (int64_t) kKey && g.n_expert_used == (int64_t) kUsed, "attention and router sizes");
    check(g.ssm_head_dim() == 8 && g.ssm_conv_dim() == 64 && g.ssm_in_dim() == 100, "Mamba2 head width 8, conv channels 64, in_proj rows 100");
    check(g.expert_weights_scale == 5.0f && !g.expert_weights_norm && g.moe_latent == (int64_t) kLatent, "router scale and latent width");
    check(g.n_vocab == (int64_t) kVocab, "vocabulary from the token list");
    check(layout_error(d.path, g, nullptr).empty(), "a complete tiny model passes the layout check");
    check(layout_error(d.path, g, [](std::vector<Tensor>& ts) { ts.erase(ts.begin() + 6); }).find("missing") == 0,
          "a missing tensor is refused: " + layout_error(d.path, g, [](std::vector<Tensor>& ts) { ts.erase(ts.begin() + 6); }));
    check(layout_error(d.path, g, [](std::vector<Tensor>& ts) {
              for (auto& t : ts)
                  if (t.name == "blk.1.ffn_up_exps.weight") t.shape[0] += 1;
          }).find("blk.1.ffn_up_exps.weight is [") == 0,
          "a wrong shape is refused, naming the tensor");
    check(layout_error(d.path, g, [](std::vector<Tensor>& ts) { ts.push_back({"blk.0.surprise.weight", {4}, 0, 9}); })
                  .find("surprise") != std::string::npos,
          "a tensor the table does not name is refused");

    // refusals of the guard
    auto expect_refused = [&](const char* name, std::vector<Kv> kv, const char* part) {
        const fs::path p = d.path / name;
        fixture::write(p, kv, {});
        const strata::GgufFile f(p.string());
        const std::string e = strata::check_architecture(f);
        check(e.find(part) != std::string::npos, std::string(name) + ": " + e);
    };
    {
        std::vector<Kv> kv = metadata();
        kv.erase(std::remove_if(kv.begin(), kv.end(), [](const Kv& k) { return k.key == "nemotron_h_moe.moe_latent_size"; }), kv.end());
        expect_refused("no-latent.gguf", kv, "missing nemotron_h_moe.moe_latent_size");
        kv = metadata();
        for (Kv& k : kv)
            if (k.key == "nemotron_h_moe.feed_forward_length") k.arr.pop_back();
        expect_refused("short-list.gguf", kv, "one entry for each");
        kv = metadata();
        for (Kv& k : kv)
            if (k.key == "nemotron_h_moe.nextn_predict_layers") k.u = kBlocks;
        expect_refused("mtp-all.gguf", kv, "not smaller");
    }
    {
        // a main block with both attention and a feed-forward is not this model
        std::vector<Kv> kv = metadata();
        for (Kv& k : kv)
            if (k.key == "nemotron_h_moe.feed_forward_length") k.arr[3] = kFf;
        const fs::path p = d.path / "both.gguf";
        fixture::write(p, kv, {});
        strata::nemotron::Geometry bad;
        std::string e;
        check(!geometry_of(p, bad, e) && e.find("both attention and a feed-forward") != std::string::npos, "attention + feed-forward in a main block: " + e);
    }
    {
        // the real model's lists are 89 entries long: the reader once kept only the first 64 entries of a list
        // the 88 main blocks as the real file's tensors give them (A = attention, E = MoE, M = Mamba2); the MTP block follows
        const char* real = "MEMEMEMAEMEMEMEMAEMEMEMEMAEMEMEMEMEMAEMEMEMEMEMAEMEMEMEMEMAEMEMEMEMEMAEMEMEMEMAEMEMEMEME";
        check(std::string(real).size() == 88, "the test's block pattern has 88 main blocks");
        std::vector<uint64_t> kvh, ffl;
        for (const char* c = real; *c; ++c) {
            kvh.push_back(*c == 'A' ? kHeadsKv : 0);
            ffl.push_back(*c == 'E' ? kFf : 0);
        }
        kvh.push_back(kHeadsKv);   // the MTP block
        ffl.push_back(kFf);
        std::vector<Kv> kv = metadata();
        for (Kv& k : kv) {
            if (k.key == "nemotron_h_moe.block_count") k.u = kvh.size();
            if (k.key == "nemotron_h_moe.attention.head_count_kv") k.arr = kvh;
            if (k.key == "nemotron_h_moe.feed_forward_length") k.arr = ffl;
        }
        const fs::path p = d.path / "long.gguf";
        fixture::write(p, kv, {});
        const strata::GgufFile f(p.string());
        check(f.get("nemotron_h_moe.feed_forward_length")->items.size() == 89, "a numeric list of 89 entries is kept whole");
        strata::nemotron::Geometry lg;
        std::string e;
        check(geometry_of(p, lg, e) && lg.n_layers == 88 && lg.n_of(strata::nemotron::BlockKind::Mamba) == 40 &&
                  lg.n_of(strata::nemotron::BlockKind::Moe) == 40 && lg.n_of(strata::nemotron::BlockKind::Attention) == 8,
              "the real block count: 88 = 40 Mamba2 + 40 MoE + 8 attention, + the MTP block " + e);
    }
    {
        // the other families are untouched
        const fs::path p = d.path / "llama.gguf";
        fixture::write(p, {fixture::str("general.architecture", "llama")}, {});
        const strata::GgufFile f(p.string());
        check(strata::model_family(f) == strata::ModelFamily::Unknown, "another architecture is still Unknown");
    }
    std::printf(g_fail ? "FAILED: %d\n" : "all ok\n", g_fail);
    return g_fail ? 1 : 0;
}


