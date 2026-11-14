// src/nemotron/cpu_experts.hpp - nemotron_h_moe's routed experts on the CPU: up (latent -> ff) -> relu^2 -> down (ff -> latent),
// no gate, the weights in their GGUF blocks (any type ggml-cpu has a vec_dot for: IQ3_XXS, IQ2_XS, IQ2_S, IQ4_XS, IQ4_NL, Q2_0,
// Q8_0, ...).  The activations are quantized to each weight type's vec_dot_type, as llama.cpp's CPU backend does.  Strata's CPU
// expert pool (src/kernels/cpu) is SwiGLU-only, so this has its own small pool of persistent workers.
//
// The rows of all jobs of one layer are spread over the workers in two phases (every up row, then every down row), so a layer's
// dozen experts keep all cores busy whatever their count.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::nemotron {

/// One layer's expert format: up [latent -> ff] and down [ff -> latent] in their GGUF types.
struct ExpertFmt {
    int up_type = -1, down_type = -1;
    int64_t latent = 0, ff = 0;
    size_t up_row = 0, down_row = 0;      ///< bytes of one row (latent values / ff values)
    size_t up_bytes = 0, down_bytes = 0;  ///< one expert's matrix
    int up_act = -1, down_act = -1;       ///< the vec_dot types the activations are quantized to
    size_t up_act_bytes = 0, down_act_bytes = 0;
};
/// Fills `f`; false with the reason when ggml-cpu cannot dot these types at these sizes.
bool expert_fmt(int up_type, int down_type, int64_t latent, int64_t ff, ExpertFmt& f, std::string& err);
/// Tests: the dot product the experts use for one row of weights of `type` (n values) against its quantized activations - this
/// file's own kernel where it has one for the type and the CPU (`own` true), else ggml's.
float row_dot(int type, int n, const void* w, const void* act, bool& own);

/// One routed entry: expert weights (its up rows and its down rows), the token row of the latent input, the output row (latent
/// floats, not weighted: the caller combines).
struct CpuJob {
    const uint8_t* up = nullptr;
    const uint8_t* down = nullptr;
    int tok = 0;
    float* out = nullptr;
};

class CpuExperts {
public:
    /// `threads` <= 0: every hardware thread but one (the caller's).  `cores` (optional): worker i runs on logical processor`n    /// cores[i + 1] and the caller on cores[0]; the worker count is then cores.size() - 1.
    explicit CpuExperts(int threads = 0, const std::vector<int>& cores = {});
    /// one logical processor per physical core (the first of each core's hardware threads), lowest first; empty if unknown
    static std::vector<int> physical_cores();
    /// `2,4,6` / `2-14:2` / `physical`: the list, or empty (with `err`) when it cannot be read
    static std::vector<int> parse_cores(const std::string& spec, std::string& err);
    ~CpuExperts();
    CpuExperts(const CpuExperts&) = delete;
    CpuExperts& operator=(const CpuExperts&) = delete;

    int threads() const { return (int) workers_.size() + 1; }
    /// The jobs of one layer: `lat` [T][f.latent] floats (the routed experts' input), every job's out row written.
    void run(const ExpertFmt& f, const float* lat, int T, const CpuJob* jobs, int n);
    /// Benchmark (--bench-cpu): read n byte ranges with every worker, in 64 KiB tasks; returns a checksum so the reads stay.
    uint64_t read_all(const uint8_t* const* ptr, const size_t* bytes, int n);
    /// Benchmark: time run()'s phases per weight type from now on (timing on), and the summary of what was timed.
    void time_phases(bool on) { timing_ = on; phase_.clear(); }
    std::string phase_report() const;

private:
    void parallel(int tasks, const std::function<void(int)>& fn);
    void worker();

    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<int> next_{0}, done_{0}, active_{0};
    int tasks_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
    const std::function<void(int)>* fn_ = nullptr;
    // scratch
    std::vector<uint8_t> act_, hq_;
    std::vector<float> h_;
    // --bench-cpu's phase timing: per weight type, ms and weight bytes of the up / down products (type -1: the activations' quantization)
    bool timing_ = false;
    struct Phase { double ms = 0; double bytes = 0; long long rows = 0; };
    std::vector<std::pair<std::string, Phase>> phase_;
    void add_phase(const std::string& k, double ms, double bytes, long long rows);
};

}  // namespace strata::nemotron
