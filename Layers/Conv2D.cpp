#include "Conv2D.h"
#include <iostream>

Conv2D::Conv2D(int in_c, int out_c, int k_size, int s, int p)
    : in_channels(in_c), out_channels(out_c), kernel_size(k_size), stride(s), padding(p) {
    // Shape: (out_channels, in_channels, kernel_size, kernel_size)
    weights = Tensor(std::vector<int>{out_c, in_c, k_size, k_size});
    // Biases: shape (out_channels, 1) to use the 2D default for 1D arrays
    biases = Tensor(out_c, 1);

    weights.randomize();
    biases.fill(0.1f); // Small positive biases to help ReLU
}

Tensor Conv2D::forward(const Tensor& input) {
    cached_input = input; // Save input for backward pass

    int N = input.shape[0];
    int C = input.shape[1]; // Should match in_channels
    int H = input.shape[2];
    int W = input.shape[3];

    int out_h = (H + 2 * padding - kernel_size) / stride + 1;
    int out_w = (W + 2 * padding - kernel_size) / stride + 1;

    Tensor output(N, out_channels, out_h, out_w);

    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < out_channels; ++c_out) {
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    float sum = biases(c_out, 0);

                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int in_y = y * stride - padding + ky;
                                int in_x = x * stride - padding + kx;

                                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                                    // Using std::vector indexing helper for 4D weights
                                    int w_idx = weights.get_index({c_out, c_in, ky, kx});
                                    sum += input(n, c_in, in_y, in_x) * weights.data[w_idx];
                                }
                            }
                        }
                    }
                    output(n, c_out, y, x) = sum;
                }
            }
        }
    }

    return output;
}

Tensor Conv2D::backward(const Tensor& grad_output) {
    int N = cached_input.shape[0];
    int H = cached_input.shape[2];
    int W = cached_input.shape[3];

    int out_h = grad_output.shape[2];
    int out_w = grad_output.shape[3];

    Tensor d_input(N, in_channels, H, W);
    d_input.fill(0.0f);

    for (int n = 0; n < N; ++n) {
        for (int c_out = 0; c_out < out_channels; ++c_out) {
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    float g_y = grad_output(n, c_out, y, x);
                    
                    biases.grad[c_out] += g_y;

                    for (int c_in = 0; c_in < in_channels; ++c_in) {
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int in_y = y * stride - padding + ky;
                                int in_x = x * stride - padding + kx;

                                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                                    int w_idx = weights.get_index({c_out, c_in, ky, kx});
                                    
                                    weights.grad[w_idx] += cached_input(n, c_in, in_y, in_x) * g_y;
                                    d_input(n, c_in, in_y, in_x) += weights.data[w_idx] * g_y;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return d_input;
}

void Conv2D::update_weights(float learning_rate) {
    for (int i = 0; i < weights.size(); ++i) {
        weights.data[i] -= learning_rate * weights.grad[i];
    }
    for (int i = 0; i < biases.size(); ++i) {
        biases.data[i] -= learning_rate * biases.grad[i];
    }
    weights.zero_grad();
    biases.zero_grad();
}
