#pragma once
#include "Layer.h"
#include "../arch/Simd.h"
#include <span>
#include <omp.h>

namespace MetalNet {

class Dense : public Layer {
public:
    int    input_size, output_size;
    Tensor weights, biases;
    std::vector<Tensor> local_grad_weights, local_grad_biases;

    inline Dense(int in_sz, int out_sz) : input_size(in_sz), output_size(out_sz) {
        weights = Tensor(in_sz, out_sz);
        biases  = Tensor(1, out_sz);
        int fan_in = in_sz;
        weights.fill_he_init(fan_in);
        biases.fill(0.0f);
    }

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        if (input_shapes.empty()) throw std::runtime_error("Dense: input_shapes is empty");
        int in_features = input_shapes[0].back(); 
        if (in_features != input_size) {
            throw std::runtime_error("Dense: dimension mismatch.");
        }
        const int N = input_shapes[0][0];
        output_buffer = Tensor(N, output_size);
        grad_input_buffer = Tensor(input_shapes[0]);

        int max_threads = omp_get_max_threads();
        local_grad_weights.clear(); local_grad_biases.clear();
        for (int t = 0; t < max_threads; ++t) {
            local_grad_weights.emplace_back(input_size, output_size);
            local_grad_biases.emplace_back(1, output_size);
        }
    }
inline Tensor& forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const int N = input.shape[0];
        const int K = input_size;
        const int M = output_size;
        
        const float* inp = input.data.data();
        const float* W   = weights.data.data();
        const float* b   = biases.data.data();
        float* out = output_buffer.data.data();
        
        constexpr int BLOCK_SIZE = 64;
        
        // [SCALING FIX] Parallelize directly over N. 
        // For Batch 16, exactly 16 threads will wake up and process 1 image each!
        constexpr int V = simd::VLEN;
        #pragma omp parallel for schedule(static) if(N > 1)
        for (int i = 0; i < N; ++i) {
            float* o_row = out + i * M;

            int j = 0;
            for (; j + V <= M; j += V) {
                simd::vstore(o_row + j, simd::vload(b + j));
            }
            for (; j < M; ++j) o_row[j] = b[j];

            const float* i_row = inp + i * K;
            // Keep blocking for Cache Locality, but inside the thread!
            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                int k_max = std::min(k0 + BLOCK_SIZE, K);
                for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                    int j_max = std::min(j0 + BLOCK_SIZE, M);
                    for (int k = k0; k < k_max; ++k) {
                        float val = i_row[k];
                        const float* w_row = W + k * M;

                        simd::vfloat v_val = simd::vset1(val);
                        int jj = j0;
                        for (; jj + 2*V <= j_max; jj += 2*V) {
                            simd::vstore(o_row + jj,     simd::vfma(v_val, simd::vload(w_row + jj),     simd::vload(o_row + jj)));
                            simd::vstore(o_row + jj + V, simd::vfma(v_val, simd::vload(w_row + jj + V), simd::vload(o_row + jj + V)));
                        }
                        for (; jj + V <= j_max; jj += V) {
                            simd::vstore(o_row + jj, simd::vfma(v_val, simd::vload(w_row + jj), simd::vload(o_row + jj)));
                        }
                        for (; jj < j_max; ++jj) o_row[jj] += val * w_row[jj];
                    }
                }
            }
        }
        return output_buffer;
    }

    inline void backward(const Tensor& grad_output) override {
        const int N = grad_output.shape[0];
        const int K = input_size;
        const int M = output_size;

        const float* inp = cached_input_ptr->data.data();
        const float* go  = grad_output.data.data();
        const float* W   = weights.data.data();
        float*       di  = grad_input_buffer.data.data();
        float*       dW  = weights.grad.data();
        float*       db  = biases.grad.data();
        
        grad_input_buffer.fill(0.0f);
        constexpr int BLOCK_SIZE = 64;

        int max_threads = omp_get_max_threads();
        for (int t = 0; t < max_threads; ++t) {
            local_grad_weights[t].fill(0.0f);
            local_grad_biases[t].fill(0.0f);
        }

        #pragma omp parallel for schedule(static)
        for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
            int i_max = std::min(i0 + BLOCK_SIZE, N);
            int tid = omp_get_thread_num();
            float* local_dW = local_grad_weights[tid].data.data();
            float* local_db = local_grad_biases[tid].data.data();

            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                int k_max = std::min(k0 + BLOCK_SIZE, K);
                for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                    int j_max = std::min(j0 + BLOCK_SIZE, M);
                    for (int i = i0; i < i_max; ++i) {
                        float* di_row = di + i * K;
                        const float* go_row = go + i * M;
                        for (int k = k0; k < k_max; ++k) {
                            const float* w_row = W + k * M;
                            float acc = 0.0f;
                            constexpr int V = simd::VLEN;
                            simd::vfloat v_acc = simd::vzero();
                            int j = j0;
                            for (; j + 4*V <= j_max; j += 4*V) {
                                v_acc = simd::vfma(simd::vload(go_row + j),       simd::vload(w_row + j),       v_acc);
                                v_acc = simd::vfma(simd::vload(go_row + j + V),   simd::vload(w_row + j + V),   v_acc);
                                v_acc = simd::vfma(simd::vload(go_row + j + 2*V), simd::vload(w_row + j + 2*V), v_acc);
                                v_acc = simd::vfma(simd::vload(go_row + j + 3*V), simd::vload(w_row + j + 3*V), v_acc);
                            }
                            for (; j + V <= j_max; j += V) {
                                v_acc = simd::vfma(simd::vload(go_row + j), simd::vload(w_row + j), v_acc);
                            }
                            acc += simd::vsum(v_acc);
                            for (; j < j_max; ++j) {
                                acc += go_row[j] * w_row[j];
                            }
                            di_row[k] += acc;
                        }
                    }
                }
            }

            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                int k_max = std::min(k0 + BLOCK_SIZE, K);
                for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                    int j_max = std::min(j0 + BLOCK_SIZE, M);
                    for (int k = k0; k < k_max; ++k) {
                        float* dw_row = local_dW + k * M;
                        constexpr int V = simd::VLEN;
                        for (int i = i0; i < i_max; ++i) {
                            float val = inp[i * K + k]; // inp^T
                            const float* go_row = go + i * M;
                            simd::vfloat v_val = simd::vset1(val);
                            int j = j0;
                            for (; j + V <= j_max; j += V) {
                                simd::vstore(dw_row + j, simd::vfma(v_val, simd::vload(go_row + j), simd::vload(dw_row + j)));
                            }
                            for (; j < j_max; ++j) {
                                dw_row[j] += val * go_row[j];
                            }
                        }
                    }
                }
            }

            constexpr int V = simd::VLEN;
            for (int i = i0; i < i_max; ++i) {
                const float* go_row = go + i * M;
                int j = 0;
                for (; j + V <= M; j += V) {
                    simd::vstore(local_db + j, simd::vadd(simd::vload(local_db + j), simd::vload(go_row + j)));
                }
                for (; j < M; ++j) {
                    local_db[j] += go_row[j];
                }
            }
        }

        for (int t = 0; t < max_threads; ++t) {
            const float* l_dW = local_grad_weights[t].data.data();
            const float* l_db = local_grad_biases[t].data.data();
            #pragma omp simd
            for (int i = 0; i < K * M; ++i) dW[i] += l_dW[i];
            #pragma omp simd
            for (int i = 0; i < M; ++i) db[i] += l_db[i];
        }
    }

    inline void update_weights(float lr) override {
        float* W  = weights.data.data(); float* dW = weights.grad.data();
        float* b  = biases.data.data();  float* db = biases.grad.data();
        #pragma omp simd
        for (int i = 0; i < weights.size(); ++i) W[i] -= lr * dW[i];
        #pragma omp simd
        for (int i = 0; i < biases.size();  ++i) b[i] -= lr * db[i];
        weights.zero_grad(); biases.zero_grad();
    }

    inline std::vector<Tensor*> get_parameters() override { return {&weights, &biases}; }
    inline std::string name() const override { return "Dense"; }

    // Profiler cost model: GEMM is N*K*M MACs, bwd ~= 2x fwd (dInput + dWeights).
    inline double prof_flops_fwd() const override {
        double N = output_buffer.shape.empty() ? 0.0 : (double)output_buffer.shape[0];
        return 2.0 * N * (double)input_size * (double)output_size;
    }
    inline double prof_flops_bwd() const override { return 2.0 * prof_flops_fwd(); }
    inline long long prof_bytes_fwd() const override {
        long long N = output_buffer.shape.empty() ? 0 : (long long)output_buffer.shape[0];
        return (N * input_size + N * output_size + (long long)weights.size()) * 4;
    }
};

} // namespace MetalNet
