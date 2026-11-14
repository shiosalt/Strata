// src/nemotron/cpu_experts.cpp - see cpu_experts.hpp.
#include "cpu_experts.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#include <fstream>
#endif

#include "cpu_kernels_avx2.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#define NEMO_HAVE_X86_KERNELS 1
#endif

namespace strata::nemotron {

namespace {
constexpr int kRowBlock = 64;   // rows per task

#if defined(NEMO_HAVE_X86_KERNELS)
bool cpu_avx2_fma_f16c() {
    int r[4] = {0, 0, 0, 0};
#if defined(_MSC_VER) && !defined(__clang__)
    __cpuid(r, 1);
    const bool fma = (r[2] >> 12) & 1, f16c = (r[2] >> 29) & 1, osx = (r[2] >> 27) & 1;
    __cpuidex(r, 7, 0);
    const bool avx2 = (r[1] >> 5) & 1;
    const bool ymm = osx && ((_xgetbv(0) & 6) == 6);
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid(1, a, b, c, d);
    const bool fma = (c >> 12) & 1, f16c = (c >> 29) & 1, osx = (c >> 27) & 1;
    __cpuid_count(7, 0, a, b, c, d);
    const bool avx2 = (b >> 5) & 1;
    unsigned lo = 0, hi = 0;
    if (osx) __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    const bool ymm = osx && ((lo & 6) == 6);
    (void) r;
#endif
    return avx2 && fma && f16c && ymm;
}

#endif

using DotFn = void (*)(int, float*, size_t, const void*, size_t, const void*, size_t, int);
/// The dot product run() uses for weights of `type`: this file's own where it has one (STRATA_NEMO_CPU_KERNELS=0: always ggml's).
DotFn dot_for(int type, DotFn ggml_dot) {
#if defined(NEMO_HAVE_X86_KERNELS)
    static const bool own = cpu_avx2_fma_f16c() &&
                            !(std::getenv("STRATA_NEMO_CPU_KERNELS") && std::atoi(std::getenv("STRATA_NEMO_CPU_KERNELS")) == 0);
    if (own && type == (int) GGML_TYPE_Q2_0) return vec_dot_q2_0_q8_0_avx2;
    // (IQ3_XXS stays ggml's AVX2 kernel: an AVX2 gather of the grid and signs computed instead of looked up were both slower on a
    // Zen 3 core - 144 and 115 against 88-105 ns a 1024-value row, measured 2026-11-13 - and at ~4.5 GB/s a thread it is already
    // close to what the RAM gives 8 cores)
#endif
    (void) type;
    return ggml_dot;
}

void init_ggml() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}
const ggml_type_traits_cpu* cpu_traits(int type) {
    return type >= 0 && type < (int) GGML_TYPE_COUNT ? ggml_get_type_traits_cpu((ggml_type) type) : nullptr;
}
bool whole_blocks(int type, int64_t n) {
    const ggml_type_traits* t = ggml_get_type_traits((ggml_type) type);
    return t && t->blck_size > 0 && n % t->blck_size == 0;
}
}  // namespace

bool expert_fmt(int up_type, int down_type, int64_t latent, int64_t ff, ExpertFmt& f, std::string& err) {
    init_ggml();
    f = ExpertFmt{};
    for (int ty : {up_type, down_type}) {
        const ggml_type_traits_cpu* t = cpu_traits(ty);
        if (!t || !t->vec_dot) {
            err = "ggml-cpu has no dot product for expert type " + std::to_string(ty);
            return false;
        }
        const ggml_type_traits_cpu* a = cpu_traits((int) t->vec_dot_type);
        if (!a || !a->from_float) {
            err = "ggml-cpu cannot quantize the activations of expert type " + std::to_string(ty);
            return false;
        }
    }
    const int ua = (int) cpu_traits(up_type)->vec_dot_type, da = (int) cpu_traits(down_type)->vec_dot_type;
    if (!whole_blocks(up_type, latent) || !whole_blocks(ua, latent)) {
        err = "the latent width " + std::to_string(latent) + " is not whole blocks of the up experts' type";
        return false;
    }
    if (!whole_blocks(down_type, ff) || !whole_blocks(da, ff)) {
        err = "the expert width " + std::to_string(ff) + " is not whole blocks of the down experts' type";
        return false;
    }
    f.up_type = up_type;
    f.down_type = down_type;
    f.latent = latent;
    f.ff = ff;
    f.up_row = ggml_row_size((ggml_type) up_type, latent);
    f.down_row = ggml_row_size((ggml_type) down_type, ff);
    f.up_bytes = f.up_row * (size_t) ff;
    f.down_bytes = f.down_row * (size_t) latent;
    f.up_act = ua;
    f.down_act = da;
    f.up_act_bytes = ggml_row_size((ggml_type) ua, latent);
    f.down_act_bytes = ggml_row_size((ggml_type) da, ff);
    return true;
}

namespace {
void pin_this_thread(int cpu) {
    if (cpu < 0) return;
#if defined(_WIN32)
    if (cpu < 64) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << cpu);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#endif
}
}  // namespace

std::vector<int> CpuExperts::physical_cores() {
    std::vector<int> out;
#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<uint8_t> buf(len);
    if (len == 0 || !GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) return out;
    for (DWORD off = 0; off < len;) {
        const auto* p = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) (buf.data() + off);
        if (p->Relationship == RelationProcessorCore && p->Processor.GroupCount >= 1 && p->Processor.GroupMask[0].Group == 0) {
            const KAFFINITY m = p->Processor.GroupMask[0].Mask;
            for (int b = 0; b < 64; ++b)
                if (m & ((KAFFINITY) 1 << b)) {
                    out.push_back(b);
                    break;
                }
        }
        off += p->Size;
    }
#else
    for (int cpu = 0; cpu < 1024; ++cpu) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/thread_siblings_list");
        if (!f) break;
        int first = -1;
        f >> first;
        if (first == cpu) out.push_back(cpu);
    }
#endif
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<int> CpuExperts::parse_cores(const std::string& spec, std::string& err) {
    if (spec == "physical") {
        std::vector<int> p = physical_cores();
        if (p.empty()) err = "the physical cores are not known on this system";
        return p;
    }
    std::vector<int> out;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        int a = -1, b = -1, st = 1;
        char d1 = 0, d2 = 0;
        std::stringstream is(item);
        if (!(is >> a)) {
            err = "cannot read '" + item + "'";
            return {};
        }
        if (is >> d1) {
            if (d1 != '-' || !(is >> b)) {
                err = "cannot read '" + item + "'";
                return {};
            }
            if (is >> d2 && (d2 != ':' || !(is >> st) || st < 1)) {
                err = "cannot read '" + item + "'";
                return {};
            }
        } else b = a;
        for (int x = a; x <= b; x += st) out.push_back(x);
    }
    if (out.empty()) err = "no logical processors in '" + spec + "'";
    return out;
}

CpuExperts::CpuExperts(int threads, const std::vector<int>& cores) {
    init_ggml();
    if (!cores.empty()) {
        threads = (int) cores.size();
        pin_this_thread(cores[0]);   // the caller (the engine's host thread) on the first
    }
    if (threads <= 0) threads = (int) std::max(1u, std::thread::hardware_concurrency() - 1u);
    for (int i = 1; i < threads; ++i)
        workers_.emplace_back([this, cpu = cores.empty() ? -1 : cores[(size_t) i]] {
            pin_this_thread(cpu);
            worker();
        });
}

CpuExperts::~CpuExperts() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
        ++gen_;
    }
    cv_.notify_all();
    for (std::thread& t : workers_) t.join();
}

void CpuExperts::worker() {
    uint64_t seen = 0;
    for (;;) {
        const std::function<void(int)>* fn = nullptr;
        int tasks = 0;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
            if (stop_) return;
            seen = gen_;
            fn = fn_;
            tasks = tasks_;
            active_.fetch_add(1);   // under the lock: parallel() does not start the next generation while a worker is in this one
        }
        for (int i; (i = next_.fetch_add(1)) < tasks;) {
            (*fn)(i);
            done_.fetch_add(1);
        }
        active_.fetch_sub(1);
    }
}

void CpuExperts::parallel(int tasks, const std::function<void(int)>& fn) {
    if (tasks <= 0) return;
    if (workers_.empty() || tasks == 1) {
        for (int i = 0; i < tasks; ++i) fn(i);
        return;
    }
    for (;;) {   // a worker still inside the previous generation would take this one's task indices with that one's function
        std::unique_lock<std::mutex> lk(mu_);
        if (active_.load() == 0) {
            fn_ = &fn;
            tasks_ = tasks;
            next_.store(0);
            done_.store(0);
            ++gen_;
            break;
        }
        lk.unlock();
        std::this_thread::yield();
    }
    cv_.notify_all();
    for (int i; (i = next_.fetch_add(1)) < tasks;) {
        fn(i);
        done_.fetch_add(1);
    }
    // every task has run, and no worker is left in the loop (fn is the caller's and goes out of scope on return)
    while (done_.load() < tasks || active_.load() != 0) std::this_thread::yield();
}

void CpuExperts::run(const ExpertFmt& f, const float* lat, int T, const CpuJob* jobs, int n) {
    if (n <= 0) return;
    const ggml_type_traits_cpu* up = cpu_traits(f.up_type);
    const ggml_type_traits_cpu* dn = cpu_traits(f.down_type);
    const ggml_from_float_t q_up = cpu_traits(f.up_act)->from_float;
    const ggml_from_float_t q_dn = cpu_traits(f.down_act)->from_float;
    const DotFn up_dot = dot_for(f.up_type, (DotFn) up->vec_dot), dn_dot = dot_for(f.down_type, (DotFn) dn->vec_dot);
    const int64_t L = f.latent, F = f.ff;
    act_.resize((size_t) T * f.up_act_bytes);
    h_.resize((size_t) n * F);
    hq_.resize((size_t) n * f.down_act_bytes);
    // the latent rows of the tokens the jobs use
    std::vector<char> used((size_t) T, 0);
    for (int j = 0; j < n; ++j) used[(size_t) jobs[j].tok] = 1;
    for (int t = 0; t < T; ++t)
        if (used[(size_t) t]) q_up(lat + (size_t) t * L, act_.data() + (size_t) t * f.up_act_bytes, L);
    // the jobs of one expert together: each weight row is read once (from RAM) and applied to every token of that expert while it is in
    // the cache - an expert with r rows costs about one read of its weights, not r (the prompt path's CPU experts have up to dozens)
    std::vector<int> order((size_t) n);
    for (int j = 0; j < n; ++j) order[(size_t) j] = j;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return jobs[a].up < jobs[b].up; });
    std::vector<int> gstart;   // group g: order[gstart[g] .. gstart[g + 1])
    for (int i = 0; i < n; ++i)
        if (i == 0 || jobs[order[(size_t) i]].up != jobs[order[(size_t) i - 1]].up) gstart.push_back(i);
    const int ng = (int) gstart.size();
    gstart.push_back(n);
    const int up_blocks = (int) ((F + kRowBlock - 1) / kRowBlock);
    const std::function<void(int)> up_task = [&](int i) {
        const int gi = i / up_blocks, r0 = (i % up_blocks) * kRowBlock, r1 = (int) std::min<int64_t>(F, r0 + kRowBlock);
        const uint8_t* w = jobs[order[(size_t) gstart[(size_t) gi]]].up;
        for (int r = r0; r < r1; ++r) {
            const uint8_t* wr = w + (size_t) r * f.up_row;
            for (int q = gstart[(size_t) gi]; q < gstart[(size_t) gi + 1]; ++q) {
                const int j = order[(size_t) q];
                float s = 0.0f;
                up_dot((int) L, &s, 0, wr, 0, act_.data() + (size_t) jobs[j].tok * f.up_act_bytes, 0, 1);
                h_[(size_t) j * F + r] = s > 0.0f ? s * s : 0.0f;
            }
        }
    };
    using BClock = std::chrono::steady_clock;
    auto ms_of = [](BClock::time_point a) { return std::chrono::duration<double, std::milli>(BClock::now() - a).count(); };
    auto tp = BClock::now();
    parallel(ng * up_blocks, up_task);
    if (timing_) {
        add_phase(std::string("up ") + ggml_type_name((ggml_type) f.up_type), ms_of(tp), (double) ng * (double) f.up_bytes, (long long) n * F);
        tp = BClock::now();
    }
    const std::function<void(int)> q_task = [&](int j) {
        q_dn(h_.data() + (size_t) j * F, hq_.data() + (size_t) j * f.down_act_bytes, F);
    };
    parallel(n, q_task);
    if (timing_) {
        add_phase("quantize", ms_of(tp), 0, n);
        tp = BClock::now();
    }
    const int dn_blocks = (int) ((L + kRowBlock - 1) / kRowBlock);
    const std::function<void(int)> dn_task = [&](int i) {
        const int gi = i / dn_blocks, r0 = (i % dn_blocks) * kRowBlock, r1 = (int) std::min<int64_t>(L, r0 + kRowBlock);
        const uint8_t* w = jobs[order[(size_t) gstart[(size_t) gi]]].down;
        for (int r = r0; r < r1; ++r) {
            const uint8_t* wr = w + (size_t) r * f.down_row;
            for (int q = gstart[(size_t) gi]; q < gstart[(size_t) gi + 1]; ++q) {
                const int j = order[(size_t) q];
                float s = 0.0f;
                dn_dot((int) F, &s, 0, wr, 0, hq_.data() + (size_t) j * f.down_act_bytes, 0, 1);
                jobs[j].out[r] = s;
            }
        }
    };
    parallel(ng * dn_blocks, dn_task);
    if (timing_) add_phase(std::string("down ") + ggml_type_name((ggml_type) f.down_type), ms_of(tp), (double) ng * (double) f.down_bytes, (long long) n * L);
}

void CpuExperts::add_phase(const std::string& k, double ms, double bytes, long long rows) {
    for (auto& [name, p] : phase_)
        if (name == k) {
            p.ms += ms;
            p.bytes += bytes;
            p.rows += rows;
            return;
        }
    phase_.push_back({k, Phase{ms, bytes, rows}});
}

std::string CpuExperts::phase_report() const {
    std::string s;
    char b[200];
    for (const auto& [name, p] : phase_) {
        std::snprintf(b, sizeof b, "\n    %-16s %9.1f ms %8.2f GB %6.1f GB/s %8.1f ns a row", name.c_str(), p.ms, p.bytes / 1e9,
                      p.ms > 0 ? p.bytes / p.ms / 1e6 : 0.0, p.rows ? p.ms * 1e6 / (double) p.rows : 0.0);
        s += b;
    }
    return s;
}

float row_dot(int type, int n, const void* w, const void* act, bool& own) {
    init_ggml();
    const DotFn g = (DotFn) cpu_traits(type)->vec_dot;
    const DotFn d = dot_for(type, g);
    own = d != g;
    float s = 0.0f;
    d(n, &s, 0, w, 0, act, 0, 1);
    return s;
}

uint64_t CpuExperts::read_all(const uint8_t* const* ptr, const size_t* bytes, int n) {
    constexpr size_t kPiece = (size_t) 64 << 10;
    std::vector<std::pair<const uint8_t*, size_t>> pieces;
    for (int i = 0; i < n; ++i)
        for (size_t o = 0; o < bytes[i]; o += kPiece) pieces.emplace_back(ptr[i] + o, std::min(kPiece, bytes[i] - o));
    std::atomic<uint64_t> sum{0};
    const std::function<void(int)> task = [&](int i) {
        const uint64_t* p = (const uint64_t*) pieces[(size_t) i].first;
        const size_t m = pieces[(size_t) i].second / 8;
        uint64_t a = 0, b = 0, c = 0, d = 0;
        size_t k = 0;
        for (; k + 4 <= m; k += 4) {
            a += p[k];
            b ^= p[k + 1];
            c += p[k + 2];
            d ^= p[k + 3];
        }
        for (; k < m; ++k) a += p[k];
        sum.fetch_add(a + b + c + d, std::memory_order_relaxed);
    };
    parallel((int) pieces.size(), task);
    return sum.load();
}

}  // namespace strata::nemotron
