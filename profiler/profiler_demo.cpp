// =====================================================================
//  MetalNet Profiler - headless demo
// ---------------------------------------------------------------------
//  Trains a small VGG-style net on synthetic CIFAR-shaped data with the
//  profiler enabled, prints a per-layer cost table, and exports a
//  Chrome/Perfetto trace ("metalnet_trace.json") of one training step.
//
//  Build with -DMETALNET_PROFILE (CMake target: metalnet_profiler_demo).
//  View the trace at https://ui.perfetto.dev  (Open trace file).
// =====================================================================
#include "MetalNet/MetalNet.h"
#include "MetalNet/profiler/Profiler.h"

#include <chrono>
#include <cstdio>
#include <vector>
#include <omp.h>

using namespace MetalNet;
namespace P = MetalNet::prof;

static Model build_vgg_cifar() {
    Model model;
    model << conv2d(3, 64, 3, 1, 1) << batchnorm2d(64) << leaky_relu(0.01f)
          << conv2d(64, 64, 3, 1, 1) << batchnorm2d(64) << leaky_relu(0.01f)
          << maxpool2d(2, 2)
          << conv2d(64, 128, 3, 1, 1) << batchnorm2d(128) << leaky_relu(0.01f)
          << conv2d(128, 128, 3, 1, 1) << batchnorm2d(128) << leaky_relu(0.01f)
          << maxpool2d(2, 2)
          << flatten() << dense(128 * 8 * 8, 10);
    return model;
}

int main(int argc, char** argv) {
    const int   batch   = (argc > 1) ? std::atoi(argv[1]) : 64;
    const int   threads = (argc > 2) ? std::atoi(argv[2]) : omp_get_max_threads();
    const int   steps   = (argc > 3) ? std::atoi(argv[3]) : 20;
    omp_set_num_threads(threads);

    std::printf("MetalNet Profiler Demo  |  batch=%d threads=%d steps=%d\n", batch, threads, steps);

    Model m = build_vgg_cifar();
    m.compile({batch, 3, 32, 32});
    m.train();

    Adam opt(0.001f);
    CrossEntropyLoss criterion;
    Tensor x(batch, 3, 32, 32); x.randomize();
    Tensor y(batch, 10); y.fill(0.0f);
    for (int i = 0; i < batch; ++i) y(i, i % 10) = 1.0f;

    auto& prof = P::Profiler::get();
    prof.start(threads, batch);

    // Warmup (not recorded as a step).
    Tensor preds = m.forward(x);
    float loss = criterion.forward(preds, y);
    Tensor dL = criterion.backward(preds, y);
    m.backward(dL);
    opt.step(m.nodes);

    // Capture one full step into a Perfetto trace.
    prof.begin_trace();

    for (int s = 0; s < steps; ++s) {
        auto t0 = P::clk::now();
        preds   = m.forward(x);
        auto t1 = P::clk::now();
        loss    = criterion.forward(preds, y);
        dL      = criterion.backward(preds, y);
        m.backward(dL);
        auto t2 = P::clk::now();
        opt.step(m.nodes);

        double fwd_ms = P::ms_between(t0, t1);
        double bwd_ms = P::ms_between(t1, t2);
        double thr    = batch / ((fwd_ms + bwd_ms) / 1000.0);
        prof.record_step(loss, thr, fwd_ms, bwd_ms);

        if (s == 0) prof.end_trace();   // keep only the first step in the trace file
    }

    // --- Per-layer cost table -------------------------------------
    std::printf("\n%-3s %-14s %-18s %10s %10s %8s %10s\n",
                "#", "Layer", "Output", "fwd(ms)", "bwd(ms)", "params", "GFLOP/s");
    std::printf("--------------------------------------------------------------------------------\n");
    double tot_fwd = 0, tot_bwd = 0;
    for (size_t i = 0; i < prof.layers.size(); ++i) {
        const auto& l = prof.layers[i];
        if (!l.init) continue;
        char shp[32] = {0};
        std::string s = "[";
        for (size_t d = 0; d < l.out_shape.size(); ++d)
            s += std::to_string(l.out_shape[d]) + (d + 1 < l.out_shape.size() ? "," : "");
        s += "]";
        std::snprintf(shp, sizeof(shp), "%s", s.c_str());
        double gflops = l.fwd_ms_ewma > 0 ? (l.fwd_flops / (l.fwd_ms_ewma / 1000.0)) / 1e9 : 0.0;
        std::printf("%-3zu %-14s %-18s %10.4f %10.4f %8lld %10.1f\n",
                    i, l.name.c_str(), shp, l.fwd_ms_ewma, l.bwd_ms_ewma, l.params, gflops);
        tot_fwd += l.fwd_ms_ewma;
        tot_bwd += l.bwd_ms_ewma;
    }
    std::printf("--------------------------------------------------------------------------------\n");
    std::printf("TOTAL  fwd=%.3f ms  bwd=%.3f ms  step=%.3f ms\n", tot_fwd, tot_bwd, tot_fwd + tot_bwd);
    std::printf("Params: %lld   Fwd GFLOPs/step: %.2f   Throughput: %.1f img/s   Peak RAM: %.1f MB\n",
                prof.total_params(), prof.total_fwd_flops() / 1e9,
                prof.last_throughput, P::peak_ram_mb());

    const char* trace_path = "metalnet_trace.json";
    if (prof.export_chrome_trace(trace_path))
        std::printf("\nPerfetto trace written: %s  (open at https://ui.perfetto.dev)\n", trace_path);

    return 0;
}
