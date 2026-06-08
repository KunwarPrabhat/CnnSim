// =====================================================================
//  Profiler self-test  -  architecture-independent (no SIMD kernels)
// ---------------------------------------------------------------------
//  Exercises the profiler core (timers, EWMA, ring buffers, step stats,
//  Perfetto export) with a dummy layer graph so it builds and runs on
//  any CPU - used to validate the profiler on machines where the AVX2
//  engine itself cannot compile (e.g. arm64).
// =====================================================================
#include "MetalNet/profiler/Profiler.h"

#include <thread>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace MetalNet;
namespace P = MetalNet::prof;

// Minimal Layer with a fixed output shape; default cost model applies.
struct DummyLayer : Layer {
    std::string nm;
    DummyLayer(std::string n, std::vector<int> shape) : nm(std::move(n)) {
        output_buffer = Tensor(shape);
    }
    void compile(const std::vector<std::vector<int>>&) override {}
    std::string name() const override { return nm; }
};

int main() {
    std::vector<DummyLayer> layers = {
        {"Conv2D",      {64, 32, 32, 64}},
        {"BatchNorm2D", {64, 32, 32, 64}},
        {"LeakyReLU",   {64, 32, 32, 64}},
        {"MaxPool2D",   {64, 16, 16, 64}},
        {"Flatten",     {64, 16384}},
        {"Dense",       {64, 10}},
    };

    auto& prof = P::Profiler::get();
    prof.start(8, 64);
    prof.begin_trace();

    const int steps = 12;
    for (int s = 0; s < steps; ++s) {
        double fwd_total = 0, bwd_total = 0;

        for (size_t i = 0; i < layers.size(); ++i) {
            auto t0 = P::clk::now();
            std::this_thread::sleep_for(std::chrono::microseconds(150 + (int)i * 60));
            prof.record_fwd((int)i, &layers[i], t0);
            fwd_total += P::ms_between(t0, P::clk::now());
        }
        for (int i = (int)layers.size() - 1; i >= 0; --i) {
            auto t0 = P::clk::now();
            std::this_thread::sleep_for(std::chrono::microseconds(220 + i * 80));
            prof.record_bwd(i, &layers[i], t0);
            bwd_total += P::ms_between(t0, P::clk::now());
        }

        double loss = 2.5 / (1.0 + 0.15 * s);
        double thr  = 64.0 / ((fwd_total + bwd_total) / 1000.0);
        prof.record_step(loss, thr, fwd_total, bwd_total);
    }
    prof.end_trace();

    // --- Validate aggregates --------------------------------------
    std::printf("== Profiler self-test ==\n");
    std::printf("steps recorded   : %lld  (expected %d)\n", prof.step, steps);
    std::printf("layers tracked   : %zu  (expected %zu)\n", prof.layers.size(), layers.size());
    std::printf("total params     : %lld\n", prof.total_params());
    std::printf("fwd GFLOPs/step  : %.4f\n", prof.total_fwd_flops() / 1e9);
    std::printf("loss hist points : %d\n", prof.loss_hist.size());
    std::printf("last loss/thr    : %.4f / %.1f img/s\n", prof.last_loss, prof.last_throughput);
    std::printf("trace events     : %zu\n", prof.events.size());

    std::printf("\n%-3s %-14s %-20s %9s %9s %9s\n", "#", "layer", "out", "fwd(ms)", "bwd(ms)", "GFLOP/s");
    for (size_t i = 0; i < prof.layers.size(); ++i) {
        const auto& l = prof.layers[i];
        std::string sh = "[";
        for (size_t d = 0; d < l.out_shape.size(); ++d)
            sh += std::to_string(l.out_shape[d]) + (d + 1 < l.out_shape.size() ? "," : "");
        sh += "]";
        double g = l.fwd_ms_ewma > 0 ? (l.fwd_flops / (l.fwd_ms_ewma / 1000.0)) / 1e9 : 0;
        std::printf("%-3zu %-14s %-20s %9.4f %9.4f %9.3f\n",
                    i, l.name.c_str(), sh.c_str(), l.fwd_ms_ewma, l.bwd_ms_ewma, g);
    }

    const char* path = "selftest_trace.json";
    bool ok = prof.export_chrome_trace(path);
    std::printf("\nperfetto export  : %s -> %s\n", ok ? "OK" : "FAIL", path);

    // --- Assertions -----------------------------------------------
    // NB: dummy layers expose no parameters, so total_params() is 0 here by design.
    bool pass = (prof.step == steps)
             && (prof.layers.size() == layers.size())
             && (prof.loss_hist.size() == steps)
             && (prof.events.size() == steps * layers.size() * 2)
             && ok;
    std::printf("\nRESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
