#pragma once
#include "Layer.h"
#include <omp.h>
#include <immintrin.h>
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

        #pragma omp parallel for collapse(2) schedule(static)
        for (int n = 0; n < N_batch; ++n) {
            for (int y = 0; y < OH_dim; ++y) {
                for (int x = 0; x < OW_dim; ++x) {
                    float* out_pixel = out_ptr_base + ((n * OH_dim + y) * OW_dim + x) * out_channels;
                    
                    // [THROUGHPUT FIX] Strict In-Register Accumulation
                    int co = 0;
                    for (; co + 31 < out_channels; co += 32) {
                        // 1. Initialize ACCUMULATOR REGISTERS
                        __m256 acc0 = _mm256_loadu_ps(b_ptr_base + co);
                        __m256 acc1 = _mm256_loadu_ps(b_ptr_base + co + 8);
                        __m256 acc2 = _mm256_loadu_ps(b_ptr_base + co + 16);
                        __m256 acc3 = _mm256_loadu_ps(b_ptr_base + co + 24);

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
                                    __m256 v_val = _mm256_set1_ps(in_pixel[ci]);
                                    const float* w_row = w_block + ci * out_channels;
                                    
                                    // 3. Accumulate STRICTLY in registers (NO RAM THRESHING)
                                    acc0 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_row + co), acc0);
                                    acc1 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_row + co + 8), acc1);
                                    acc2 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_row + co + 16), acc2);
                                    acc3 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_row + co + 24), acc3);
                                }
                            }
                        }
                        // 4. WRITE to memory exactly ONCE
                        _mm256_storeu_ps(out_pixel + co, acc0);
                        _mm256_storeu_ps(out_pixel + co + 8, acc1);
                        _mm256_storeu_ps(out_pixel + co + 16, acc2);
                        _mm256_storeu_ps(out_pixel + co + 24, acc3);
                    }
                    
                    for (; co + 7 < out_channels; co += 8) {
                        __m256 acc0 = _mm256_loadu_ps(b_ptr_base + co);
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            int iy = y * stride - padding + ky;
                            if (iy < 0 || iy >= H_dim) continue;
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int ix = x * stride - padding + kx;
                                if (ix < 0 || ix >= W_dim) continue;
                                const float* in_pixel = in_ptr_base + ((n * H_dim + iy) * W_dim + ix) * in_channels;
                                const float* w_block = w_ptr_base + ((ky * kernel_size + kx) * in_channels) * out_channels;
                                for (int ci = 0; ci < in_channels; ++ci) {
                                    __m256 v_val = _mm256_set1_ps(in_pixel[ci]);
                                    acc0 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_block + ci * out_channels + co), acc0);
                                }
                            }
                        }
                        _mm256_storeu_ps(out_pixel + co, acc0);
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
                            
                            for (int ci = 0; ci < in_channels; ++ci) {
                                const float* w_row = w_block + ci * out_channels;
                                
                                __m256 acc_v = _mm256_setzero_ps();
                                int co = 0;
                                for (; co + 31 < out_channels; co += 32) {
                                    __m256 a0 = _mm256_mul_ps(_mm256_loadu_ps(go_pixel + co), _mm256_loadu_ps(w_row + co));
                                    __m256 a1 = _mm256_mul_ps(_mm256_loadu_ps(go_pixel + co + 8), _mm256_loadu_ps(w_row + co + 8));
                                    __m256 a2 = _mm256_mul_ps(_mm256_loadu_ps(go_pixel + co + 16), _mm256_loadu_ps(w_row + co + 16));
                                    __m256 a3 = _mm256_mul_ps(_mm256_loadu_ps(go_pixel + co + 24), _mm256_loadu_ps(w_row + co + 24));
                                    acc_v = _mm256_add_ps(acc_v, _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
                                }
                                for (; co + 7 < out_channels; co += 8) {
                                    acc_v = _mm256_fmadd_ps(_mm256_loadu_ps(go_pixel + co), _mm256_loadu_ps(w_row + co), acc_v);
                                }
                                
                                alignas(32) float acc_arr[8];
                                _mm256_store_ps(acc_arr, acc_v);
                                float sum = acc_arr[0] + acc_arr[1] + acc_arr[2] + acc_arr[3] + acc_arr[4] + acc_arr[5] + acc_arr[6] + acc_arr[7];
                                
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
                        int co = 0;
                        for (; co + 31 < out_channels; co += 32) {
                            _mm256_storeu_ps(local_db + co, _mm256_add_ps(_mm256_loadu_ps(local_db + co), _mm256_loadu_ps(go_pixel + co)));
                            _mm256_storeu_ps(local_db + co + 8, _mm256_add_ps(_mm256_loadu_ps(local_db + co + 8), _mm256_loadu_ps(go_pixel + co + 8)));
                            _mm256_storeu_ps(local_db + co + 16, _mm256_add_ps(_mm256_loadu_ps(local_db + co + 16), _mm256_loadu_ps(go_pixel + co + 16)));
                            _mm256_storeu_ps(local_db + co + 24, _mm256_add_ps(_mm256_loadu_ps(local_db + co + 24), _mm256_loadu_ps(go_pixel + co + 24)));
                        }
                        for (; co + 7 < out_channels; co += 8) {
                            _mm256_storeu_ps(local_db + co, _mm256_add_ps(_mm256_loadu_ps(local_db + co), _mm256_loadu_ps(go_pixel + co)));
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
                                
                                for (int ci = 0; ci < in_channels; ++ci) {
                                    __m256 v_val = _mm256_set1_ps(in_pixel[ci]);
                                    float* dw_row = dw_block + ci * out_channels;
                                    
                                    int c = 0;
                                    for (; c + 31 < out_channels; c += 32) {
                                        __m256 dw0 = _mm256_loadu_ps(dw_row + c);
                                        __m256 dw1 = _mm256_loadu_ps(dw_row + c + 8);
                                        __m256 dw2 = _mm256_loadu_ps(dw_row + c + 16);
                                        __m256 dw3 = _mm256_loadu_ps(dw_row + c + 24);

                                        dw0 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(go_pixel + c), dw0);
                                        dw1 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(go_pixel + c + 8), dw1);
                                        dw2 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(go_pixel + c + 16), dw2);
                                        dw3 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(go_pixel + c + 24), dw3);

                                        _mm256_storeu_ps(dw_row + c, dw0);
                                        _mm256_storeu_ps(dw_row + c + 8, dw1);
                                        _mm256_storeu_ps(dw_row + c + 16, dw2);
                                        _mm256_storeu_ps(dw_row + c + 24, dw3);
                                    }
                                    for (; c + 7 < out_channels; c += 8) {
                                        _mm256_storeu_ps(dw_row + c, _mm256_fmadd_ps(v_val, _mm256_loadu_ps(go_pixel + c), _mm256_loadu_ps(dw_row + c)));
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
};

} // namespace MetalNet