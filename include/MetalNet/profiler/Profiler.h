#pragma once
// =====================================================================
//  MetalNet Profiler  -  header-only, zero external deps
// ---------------------------------------------------------------------
//  Records per-layer forward/backward timings, FLOPs, bytes, memory and
//  step-level metrics (loss / throughput). Exposes live ring-buffers for
//  any UI and a Chrome/Perfetto trace exporter.
//
//  Core engine stays untouched unless compiled with -DMETALNET_PROFILE.
//  This header pulls in Layer.h only (Tensor + Layer), no SIMD headers,
//  so it builds on any architecture (used by the GUI/demo, not the kernels).
// =====================================================================
#include <chrono>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <algorithm>

#include "../Layers/Layer.h"

#if defined(_WIN32)
  #include <windows.h>
  #include <psapi.h>
#else
  #include <sys/resource.h>
#endif

namespace MetalNet {
namespace prof {

using clk = std::chrono::high_resolution_clock;

inline double ms_between(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
inline double us_between(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

// Cross-platform peak resident set size in MB.
inline double peak_ram_mb() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    return 0.0;
#else
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
  #if defined(__APPLE__)
    return (double)ru.ru_maxrss / (1024.0 * 1024.0);   // bytes on macOS
  #else
    return (double)ru.ru_maxrss / 1024.0;              // kilobytes on Linux
  #endif
#endif
}

// Per-layer accumulated statistics.
struct LayerStat {
    std::string      name;
    std::vector<int> out_shape;
    long long        params    = 0;
    long long        out_elems = 0;
    long long        bytes_fwd = 0;     // approx data touched in fwd
    double           fwd_flops = 0.0;   // per call
    double           bwd_flops = 0.0;
    double           fwd_ms      = 0.0; // last
    double           bwd_ms      = 0.0;
    double           fwd_ms_ewma = 0.0; // smoothed
    double           bwd_ms_ewma = 0.0;
    bool             init = false;
};

// Fixed-capacity scrolling series for live plots (contiguous for ImPlot).
struct ScrollBuf {
    int                cap;
    std::vector<float> xs, ys;
    explicit ScrollBuf(int c = 1200) : cap(c) {}
    inline void push(float x, float y) {
        xs.push_back(x); ys.push_back(y);
        if ((int)xs.size() > cap) { xs.erase(xs.begin()); ys.erase(ys.begin()); }
    }
    inline int  size() const { return (int)xs.size(); }
    inline void clear() { xs.clear(); ys.clear(); }
};

// Chrome Trace Event (ph "X" = complete duration event).
struct TraceEvent {
    std::string name;
    std::string cat;     // "fwd" | "bwd"
    double      ts_us;    // relative to origin
    double      dur_us;
    int         tid;      // 0 = forward lane, 1 = backward lane
};

class Profiler {
public:
    static Profiler& get() { static Profiler p; return p; }

    std::mutex        mtx;
    bool              enabled    = false;
    double            ewma_alpha = 0.2;
    clk::time_point   origin     = clk::now();

    // context
    int  n_threads = 1;
    int  batch     = 0;

    // per-layer
    std::vector<LayerStat> layers;

    // step-level (last values + history)
    long long step           = 0;
    double    last_step_ms    = 0.0;
    double    last_fwd_ms     = 0.0;
    double    last_bwd_ms     = 0.0;
    double    last_loss       = 0.0;
    double    last_throughput = 0.0;

    ScrollBuf loss_hist, thrpt_hist, step_ms_hist, mem_hist;

    // perfetto / timeline capture
    bool                    tracing = false;
    std::vector<TraceEvent> events;

    // optional one-shot thread-scaling sweep results (threads -> img/s)
    std::vector<std::pair<int, double>> thread_scaling;

    // ---- lifecycle -------------------------------------------------
    inline void start(int threads, int batch_size) {
        std::lock_guard<std::mutex> lk(mtx);
        enabled   = true;
        n_threads = threads;
        batch     = batch_size;
        origin    = clk::now();
    }
    inline void reset() {
        std::lock_guard<std::mutex> lk(mtx);
        layers.clear(); events.clear();
        loss_hist.clear(); thrpt_hist.clear(); step_ms_hist.clear(); mem_hist.clear();
        thread_scaling.clear();
        step = 0;
    }

    // ---- per-layer hooks (called from Model under -DMETALNET_PROFILE)
    inline void record_fwd(int idx, Layer* l, clk::time_point t0) {
        if (!enabled) return;
        auto   t1  = clk::now();
        double dur = ms_between(t0, t1);
        std::lock_guard<std::mutex> lk(mtx);
        LayerStat& s = at(idx);
        refresh_meta(s, l);
        s.fwd_ms      = dur;
        s.fwd_ms_ewma = (s.fwd_ms_ewma == 0.0) ? dur : s.fwd_ms_ewma * (1 - ewma_alpha) + dur * ewma_alpha;
        if (tracing)
            events.push_back({s.name, "fwd", us_between(origin, t0), dur * 1000.0, 0});
    }

    inline void record_bwd(int idx, Layer* l, clk::time_point t0) {
        if (!enabled) return;
        auto   t1  = clk::now();
        double dur = ms_between(t0, t1);
        std::lock_guard<std::mutex> lk(mtx);
        LayerStat& s = at(idx);
        refresh_meta(s, l);
        s.bwd_ms      = dur;
        s.bwd_ms_ewma = (s.bwd_ms_ewma == 0.0) ? dur : s.bwd_ms_ewma * (1 - ewma_alpha) + dur * ewma_alpha;
        if (tracing)
            events.push_back({s.name, "bwd", us_between(origin, t0), dur * 1000.0, 1});
    }

    // ---- step-level hook (called by training driver) ---------------
    inline void record_step(double loss, double throughput, double fwd_ms, double bwd_ms) {
        std::lock_guard<std::mutex> lk(mtx);
        ++step;
        last_loss       = loss;
        last_throughput = throughput;
        last_fwd_ms     = fwd_ms;
        last_bwd_ms     = bwd_ms;
        last_step_ms    = fwd_ms + bwd_ms;
        float x = (float)step;
        loss_hist.push(x, (float)loss);
        thrpt_hist.push(x, (float)throughput);
        step_ms_hist.push(x, (float)last_step_ms);
        mem_hist.push(x, (float)peak_ram_mb());
    }

    // ---- perfetto / chrome trace ----------------------------------
    inline void begin_trace() {
        std::lock_guard<std::mutex> lk(mtx);
        events.clear();
        origin  = clk::now();
        tracing = true;
    }
    inline void end_trace() {
        std::lock_guard<std::mutex> lk(mtx);
        tracing = false;
    }
    // Writes chrome://tracing / ui.perfetto.dev compatible JSON.
    inline bool export_chrome_trace(const std::string& path) {
        std::lock_guard<std::mutex> lk(mtx);
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        std::fprintf(f, "{\"traceEvents\":[\n");
        std::fprintf(f, "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"tid\":0,\"args\":{\"name\":\"MetalNet\"}},\n");
        std::fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":0,\"args\":{\"name\":\"forward\"}},\n");
        std::fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":1,\"args\":{\"name\":\"backward\"}}%s\n",
                     events.empty() ? "" : ",");
        for (size_t i = 0; i < events.size(); ++i) {
            const TraceEvent& e = events[i];
            std::fprintf(f,
                "{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\",\"ts\":%.3f,\"dur\":%.3f,\"pid\":1,\"tid\":%d}%s\n",
                e.name.c_str(), e.cat.c_str(), e.ts_us, e.dur_us, e.tid,
                (i + 1 < events.size()) ? "," : "");
        }
        std::fprintf(f, "]}\n");
        std::fclose(f);
        return true;
    }

    // ---- derived aggregates ---------------------------------------
    inline double total_fwd_flops() const { double s = 0; for (auto& l : layers) s += l.fwd_flops; return s; }
    inline double total_bwd_flops() const { double s = 0; for (auto& l : layers) s += l.bwd_flops; return s; }
    inline long long total_params() const { long long s = 0; for (auto& l : layers) s += l.params; return s; }

private:
    Profiler() = default;

    inline LayerStat& at(int idx) {
        if ((int)layers.size() <= idx) layers.resize(idx + 1);
        return layers[idx];
    }
    // (Re)capture static metadata; recompute when output size changes (e.g. batch resize).
    inline void refresh_meta(LayerStat& s, Layer* l) {
        long long oe = (long long)l->output_buffer.size();
        if (s.init && s.out_elems == oe) return;
        s.name      = l->name();
        s.out_shape = l->output_buffer.shape;
        s.out_elems = oe;
        s.params    = 0;
        for (Tensor* p : l->get_parameters()) s.params += p->size();
        s.fwd_flops = l->prof_flops_fwd();
        s.bwd_flops = l->prof_flops_bwd();
        s.bytes_fwd = l->prof_bytes_fwd();
        s.init      = true;
    }
};

} // namespace prof
} // namespace MetalNet
