#pragma once
#include "Layer.h"
#include "../arch/Simd.h"
#include <omp.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace MetalNet {

class Conv2D : public Layer {
public:
    int in_channels, out_channels, kernel_size, stride, padding;
    Tensor weights, biases;
    
    int N_batch, H_dim, W_dim, OH_dim, OW_dim;

    // [RAM FIX] Pre-allocated thread buffers for zero-allocation backward pass
    std::vector<Tensor> local_grad_weights;
    std::vector<Tensor> local_grad_biases;

    inline Conv2D(int in_c, int out_c, int k, int s=1, int p=0)
        : in_channels(in_c), out_channels(out_c), kernel_size(k), stride(s), padding(p) {
        weights = Tensor(std::vector<int>{k, k, in_c, out_c});
        biases  = Tensor(out_c, 1);
        int fan_in = in_c * k * k;
        weights.fill_he_init(fan_in);
        biases.fill(0.0f);
    }

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        N_batch = input_shapes[0][0];
        if (input_shapes[0].size() == 4 && input_shapes[0][1] == in_channels && input_shapes[0][3] != in_channels) {
            H_dim = input_shapes[0][2]; W_dim = input_shapes[0][3];
        } else if (input_shapes[0].size() == 4) {
            H_dim = input_shapes[0][1]; W_dim = input_shapes[0][2]; 
        } else {
             throw std::runtime_error("Conv2D: Malformed tensor limits.");
        }

        OH_dim = (H_dim + 2*padding - kernel_size) / stride + 1;
        OW_dim = (W_dim + 2*padding - kernel_size) / stride + 1;
        
        output_buffer = Tensor(N_batch, OH_dim, OW_dim, out_channels);
        grad_input_buffer = Tensor(input_shapes[0]);

        // Pre-allocate thread buffers ONCE to avoid malloc lock contention
        int max_threads = omp_get_max_threads();
        local_grad_weights.clear(); local_grad_biases.clear();
        for (int t = 0; t < max_threads; ++t) {
            local_grad_weights.emplace_back(std::vector<int>{kernel_size, kernel_size, in_channels, out_channels});
            local_grad_biases.emplace_back(out_channels, 1);
        }
    }
inline Tensor& forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const float* in_ptr_base = input.data.data();
        const float* w_ptr_base = weights.data.data();
        const float* b_ptr_base = biases.data.data();
        float* out_ptr_base = output_buffer.data.data();

        // [LATENCY FIX] Bypass OpenMP overhead entirely when Batch Size == 1
        #pragma omp parallel for collapse(2) schedule(static) if(N_batch > 1)
        for (int n = 0; n < N_batch; ++n) {
            for (int y = 0; y < OH_dim; ++y) {
                for (int x = 0; x < OW_dim; ++x) {
                    float* out_pixel = out_ptr_base + ((n * OH_dim + y) * OW_dim + x) * out_channels;
                    
                    // [THROUGHPUT FIX] Strict In-Register Accumulation
                    constexpr int V = simd::VLEN;
                    int co = 0;
                    // [ILP FIX] 8 independent accumulator chains to hide FMA latency.
                    for (; co + 8*V <= out_channels; co += 8*V) {
                        simd::vfloat a0 = simd::vload(b_ptr_base + co);
                        simd::vfloat a1 = simd::vload(b_ptr_base + co + V);
                        simd::vfloat a2 = simd::vload(b_ptr_base + co + 2*V);
                        simd::vfloat a3 = simd::vload(b_ptr_base + co + 3*V);
                        simd::vfloat a4 = simd::vload(b_ptr_base + co + 4*V);
                        simd::vfloat a5 = simd::vload(b_ptr_base + co + 5*V);
                        simd::vfloat a6 = simd::vload(b_ptr_base + co + 6*V);
                        simd::vfloat a7 = simd::vload(b_ptr_base + co + 7*V);
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            int iy = y * stride - padding + ky;
                            if (iy < 0 || iy >= H_dim) continue;
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int ix = x * stride - padding + kx;
                                if (ix < 0 || ix >= W_dim) continue;
                                const float* in_pixel = in_ptr_base + ((n * H_dim + iy) * W_dim + ix) * in_channels;
                                const float* w_block = w_ptr_base + ((ky * kernel_size + kx) * in_channels) * out_channels;
                                for (int ci = 0; ci < in_channels; ++ci) {
                                    simd::vfloat v_val = simd::vset1(in_pixel[ci]);
                                    const float* w_row = w_block + ci * out_channels + co;
                                    a0 = simd::vfma(v_val, simd::vload(w_row),       a0);
                                    a1 = simd::vfma(v_val, simd::vload(w_row + V),   a1);
                                    a2 = simd::vfma(v_val, simd::vload(w_row + 2*V), a2);
                                    a3 = simd::vfma(v_val, simd::vload(w_row + 3*V), a3);
                                    a4 = simd::vfma(v_val, simd::vload(w_row + 4*V), a4);
                                    a5 = simd::vfma(v_val, simd::vload(w_row + 5*V), a5);
                                    a6 = simd::vfma(v_val, simd::vload(w_row + 6*V), a6);
                                    a7 = simd::vfma(v_val, simd::vload(w_row + 7*V), a7);
                                }
                            }
                        }
                        simd::vstore(out_pixel + co,       a0);
                        simd::vstore(out_pixel + co + V,   a1);
                        simd::vstore(out_pixel + co + 2*V, a2);
                        simd::vstore(out_pixel + co + 3*V, a3);
                        simd::vstore(out_pixel + co + 4*V, a4);
                        simd::vstore(out_pixel + co + 5*V, a5);
                        simd::vstore(out_pixel + co + 6*V, a6);
                        simd::vstore(out_pixel + co + 7*V, a7);
                    }
                    for (; co + 4*V <= out_channels; co += 4*V) {
                        // 1. Initialize ACCUMULATOR REGISTERS
                        simd::vfloat acc0 = simd::vload(b_ptr_base + co);
                        simd::vfloat acc1 = simd::vload(b_ptr_base + co + V);
                        simd::vfloat acc2 = simd::vload(b_ptr_base + co + 2*V);
                        simd::vfloat acc3 = simd::vload(b_ptr_base + co + 3*V);

                        // 2. Pure spatial inner loop
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            int iy = y * stride - padding + ky;
                            if (iy < 0 || iy >= H_dim) continue;
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int ix = x * stride - padding + kx;
                                if (ix < 0 || ix >= W_dim) continue;

                                const float* in_pixel = in_ptr_base + ((n * H_dim + iy) * W_dim + ix) * in_channels;
                                const float* w_block = w_ptr_base + ((ky * kernel_size + kx) * in_channels) * out_channels;

                                for (int ci = 0; ci < in_channels; ++ci) {
                                    simd::vfloat v_val = simd::vset1(in_pixel[ci]);
                                    const float* w_row = w_block + ci * out_channels;

                                    // 3. Accumulate STRICTLY in registers (NO RAM THRESHING)
                                    acc0 = simd::vfma(v_val, simd::vload(w_row + co), acc0);
                                    acc1 = simd::vfma(v_val, simd::vload(w_row + co + V), acc1);
                                    acc2 = simd::vfma(v_val, simd::vload(w_row + co + 2*V), acc2);
                                    acc3 = simd::vfma(v_val, simd::vload(w_row + co + 3*V), acc3);
                                }
                            }
                        }
                        // 4. WRITE to memory exactly ONCE
                        simd::vstore(out_pixel + co, acc0);
                        simd::vstore(out_pixel + co + V, acc1);
                        simd::vstore(out_pixel + co + 2*V, acc2);
                        simd::vstore(out_pixel + co + 3*V, acc3);
                    }

                    for (; co + V <= out_channels; co += V) {
                        simd::vfloat acc0 = simd::vload(b_ptr_base + co);
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            int iy = y * stride - padding + ky;
                            if (iy < 0 || iy >= H_dim) continue;
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int ix = x * stride - padding + kx;
                                if (ix < 0 || ix >= W_dim) continue;
                                const float* in_pixel = in_ptr_base + ((n * H_dim + iy) * W_dim + ix) * in_channels;
                                const float* w_block = w_ptr_base + ((ky * kernel_size + kx) * in_channels) * out_channels;
                                for (int ci = 0; ci < in_channels; ++ci) {
                                    simd::vfloat v_val = simd::vset1(in_pixel[ci]);
                                    acc0 = simd::vfma(v_val, simd::vload(w_block + ci * out_channels + co), acc0);
                                }
                            }
                        }
                        simd::vstore(out_pixel + co, acc0);
                    }
                    for (; co < out_channels; ++co) {
                        float sum = b_ptr_base[co];
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            int iy = y * stride - padding + ky;
                            if (iy < 0 || iy >= H_dim) continue;
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int ix = x * stride - padding + kx;
                                if (ix < 0 || ix >= W_dim) continue;
                                const float* in_pixel = in_ptr_base + ((n * H_dim + iy) * W_dim + ix) * in_channels;
                                const float* w_block = w_ptr_base + ((ky * kernel_size + kx) * in_channels) * out_channels;
                                for (int ci = 0; ci < in_channels; ++ci) {
                                    sum += in_pixel[ci] * w_block[ci * out_channels + co];
                                }
                            }
                        }
                        out_pixel[co] = sum;
                    }
                }
            }
        }
        return output_buffer;
    }

    inline void backward(const Tensor& grad_output) override {
        grad_input_buffer.fill(0.0f);
        weights.zero_grad();
        biases.zero_grad();

        const float* go_ptr_base = grad_output.data.data();
        const float* in_ptr_base = cached_input_ptr->data.data();
        const float* w_ptr_base  = weights.data.data();
        float* din_ptr_base      = grad_input_buffer.data.data();
        float* dw_ptr_base       = weights.grad.data();
        float* db_ptr_base       = biases.grad.data();

        // 1. Calculate dInput
        #pragma omp parallel for collapse(2) schedule(static)
        for (int n = 0; n < N_batch; ++n) {
            for (int iy = 0; iy < H_dim; ++iy) {
                for (int ix = 0; ix < W_dim; ++ix) {
                    float* din_pixel = din_ptr_base + ((n * H_dim + iy) * W_dim + ix) * in_channels;
                    
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        int y_strided = iy + padding - ky;
                        if (y_strided % stride != 0) continue;
                        int y = y_strided / stride;
                        if (y < 0 || y >= OH_dim) continue;

                        for (int kx = 0; kx < kernel_size; ++kx) {
                            int x_strided = ix + padding - kx;
                            if (x_strided % stride != 0) continue;
                            int x = x_strided / stride;
                            if (x < 0 || x >= OW_dim) continue;

                            const float* go_pixel = go_ptr_base + ((n * OH_dim + y) * OW_dim + x) * out_channels;
                            const float* w_block = w_ptr_base + ((ky * kernel_size + kx) * in_channels) * out_channels;
                            
                            constexpr int V = simd::VLEN;
                            for (int ci = 0; ci < in_channels; ++ci) {
                                const float* w_row = w_block + ci * out_channels;

                                simd::vfloat acc_v = simd::vzero();
                                int co = 0;
                                for (; co + 4*V <= out_channels; co += 4*V) {
                                    acc_v = simd::vfma(simd::vload(go_pixel + co),       simd::vload(w_row + co),       acc_v);
                                    acc_v = simd::vfma(simd::vload(go_pixel + co + V),   simd::vload(w_row + co + V),   acc_v);
                                    acc_v = simd::vfma(simd::vload(go_pixel + co + 2*V), simd::vload(w_row + co + 2*V), acc_v);
                                    acc_v = simd::vfma(simd::vload(go_pixel + co + 3*V), simd::vload(w_row + co + 3*V), acc_v);
                                }
                                for (; co + V <= out_channels; co += V) {
                                    acc_v = simd::vfma(simd::vload(go_pixel + co), simd::vload(w_row + co), acc_v);
                                }

                                float sum = simd::vsum(acc_v);
                                for (; co < out_channels; ++co) sum += go_pixel[co] * w_row[co];
                                din_pixel[ci] += sum;
                            }
                        }
                    }
                }
            }
        }

        // 2. Calculate dWeights and dBias 
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            // ZERO-ALLOCATION FIX: Grab pre-allocated buffers and zero them safely with memset
            Tensor& t_dw = local_grad_weights[tid];
            Tensor& t_db = local_grad_biases[tid];
            std::memset(t_dw.data.data(), 0, t_dw.size() * sizeof(float));
            std::memset(t_db.data.data(), 0, t_db.size() * sizeof(float));

            float* local_dw = t_dw.data.data();
            float* local_db = t_db.data.data();

            #pragma omp for collapse(2) schedule(static)
            for (int n = 0; n < N_batch; ++n) {
                for (int y = 0; y < OH_dim; ++y) {
                    for (int x = 0; x < OW_dim; ++x) {
                        const float* go_pixel = go_ptr_base + ((n * OH_dim + y) * OW_dim + x) * out_channels;
                        
                        // Bias Accumulation
                        constexpr int V = simd::VLEN;
                        int co = 0;
                        for (; co + 4*V <= out_channels; co += 4*V) {
                            simd::vstore(local_db + co,       simd::vadd(simd::vload(local_db + co),       simd::vload(go_pixel + co)));
                            simd::vstore(local_db + co + V,   simd::vadd(simd::vload(local_db + co + V),   simd::vload(go_pixel + co + V)));
                            simd::vstore(local_db + co + 2*V, simd::vadd(simd::vload(local_db + co + 2*V), simd::vload(go_pixel + co + 2*V)));
                            simd::vstore(local_db + co + 3*V, simd::vadd(simd::vload(local_db + co + 3*V), simd::vload(go_pixel + co + 3*V)));
                        }
                        for (; co + V <= out_channels; co += V) {
                            simd::vstore(local_db + co, simd::vadd(simd::vload(local_db + co), simd::vload(go_pixel + co)));
                        }
                        for (; co < out_channels; ++co) local_db[co] += go_pixel[co];

                        // Weight Accumulation
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            int iy = y * stride - padding + ky;
                            if (iy < 0 || iy >= H_dim) continue;
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int ix = x * stride - padding + kx;
                                if (ix < 0 || ix >= W_dim) continue;
                                
                                const float* in_pixel = in_ptr_base + ((n * H_dim + iy) * W_dim + ix) * in_channels;
                                float* dw_block = local_dw + ((ky * kernel_size + kx) * in_channels) * out_channels;
                                
                                constexpr int V = simd::VLEN;
                                for (int ci = 0; ci < in_channels; ++ci) {
                                    simd::vfloat v_val = simd::vset1(in_pixel[ci]);
                                    float* dw_row = dw_block + ci * out_channels;

                                    int c = 0;
                                    for (; c + 4*V <= out_channels; c += 4*V) {
                                        simd::vfloat dw0 = simd::vload(dw_row + c);
                                        simd::vfloat dw1 = simd::vload(dw_row + c + V);
                                        simd::vfloat dw2 = simd::vload(dw_row + c + 2*V);
                                        simd::vfloat dw3 = simd::vload(dw_row + c + 3*V);

                                        dw0 = simd::vfma(v_val, simd::vload(go_pixel + c),       dw0);
                                        dw1 = simd::vfma(v_val, simd::vload(go_pixel + c + V),   dw1);
                                        dw2 = simd::vfma(v_val, simd::vload(go_pixel + c + 2*V), dw2);
                                        dw3 = simd::vfma(v_val, simd::vload(go_pixel + c + 3*V), dw3);

                                        simd::vstore(dw_row + c,       dw0);
                                        simd::vstore(dw_row + c + V,   dw1);
                                        simd::vstore(dw_row + c + 2*V, dw2);
                                        simd::vstore(dw_row + c + 3*V, dw3);
                                    }
                                    for (; c + V <= out_channels; c += V) {
                                        simd::vstore(dw_row + c, simd::vfma(v_val, simd::vload(go_pixel + c), simd::vload(dw_row + c)));
                                    }
                                    for (; c < out_channels; ++c) dw_row[c] += in_pixel[ci] * go_pixel[c];
                                }
                            }
                        }
                    }
                }
            }
        }

        // Consolidate thread local blocks back into the main gradient weights
        int max_threads = omp_get_max_threads();
        for (int t = 0; t < max_threads; ++t) {
            float* l_dw = local_grad_weights[t].data.data();
            float* l_db = local_grad_biases[t].data.data();
            #pragma omp simd
            for (int i = 0; i < weights.size(); ++i) dw_ptr_base[i] += l_dw[i];
            #pragma omp simd
            for (int i = 0; i < out_channels; ++i) db_ptr_base[i] += l_db[i];
        }
    }

    inline void update_weights(float lr) override {
        float* W = weights.data.data();
        float* dW = weights.grad.data();
        float* b = biases.data.data();  float* db = biases.grad.data();
        #pragma omp simd
        for (int i = 0; i < weights.size(); ++i) W[i] -= lr * dW[i];
        #pragma omp simd
        for (int i = 0; i < biases.size(); ++i)  b[i] -= lr * db[i];
        weights.zero_grad(); biases.zero_grad();
    }

    inline std::vector<Tensor*> get_parameters() override { return {&weights, &biases}; }
    inline std::string name() const override { return "Conv2D"; }

    // Profiler cost model: 2 flops per MAC, bwd ~= 2x fwd (dInput + dWeights).
    inline double prof_flops_fwd() const override {
        double macs = (double)N_batch * OH_dim * OW_dim * out_channels
                      * in_channels * kernel_size * kernel_size;
        return 2.0 * macs;
    }
    inline double prof_flops_bwd() const override { return 2.0 * prof_flops_fwd(); }
    inline long long prof_bytes_fwd() const override {
        long long in  = (long long)N_batch * H_dim  * W_dim  * in_channels;
        long long out = (long long)N_batch * OH_dim * OW_dim * out_channels;
        return (in + out + (long long)weights.size()) * 4;
    }
};

} // namespace MetalNet