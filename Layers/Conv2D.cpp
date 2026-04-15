#include "Conv2D.h"
#include <iostream>

Conv2D::Conv2D(int in_c, int out_c, int k_size, int s, int p)
    : in_channels(in_c), out_channels(out_c), kernel_size(k_size), stride(s), padding(p) {
    
    weights = Tensor(std::vector<int>{out_c, in_c, k_size, k_size});
    biases = Tensor(out_c, 1);

    weights.randomize();
    biases.fill(0.1f); 
}

Tensor Conv2D::forward(const Tensor& input) {
    cached_input = input; 

    int batches = input.shape[0];
    int height = input.shape[2];
    int width = input.shape[3];

    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;

    Tensor output(batches, out_channels, out_h, out_w);

    for (int b = 0; b < batches; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    float sum = biases(oc, 0);

                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int in_y = y * stride - padding + ky;
                                int in_x = x * stride - padding + kx;

                                if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                                    sum += input(b, ic, in_y, in_x) * weights(oc, ic, ky, kx);
                                }
                            }
                        }
                    }
                    output(b, oc, y, x) = sum;
                }
            }
        }
    }

    return output;
}

Tensor Conv2D::backward(const Tensor& grad_output) {
    int batches = cached_input.shape[0];
    int height = cached_input.shape[2];
    int width = cached_input.shape[3];

    int out_h = grad_output.shape[2];
    int out_w = grad_output.shape[3];

    Tensor d_input(batches, in_channels, height, width);
    d_input.fill(0.0f);

    for (int b = 0; b < batches; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    float g = grad_output(b, oc, y, x);
                    
                    biases.grad[oc] += g;

                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int ky = 0; ky < kernel_size; ++ky) {
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                int in_y = y * stride - padding + ky;
                                int in_x = x * stride - padding + kx;

                                if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                                    weights.grad[weights.get_index({oc, ic, ky, kx})] += cached_input(b, ic, in_y, in_x) * g;
                                    d_input(b, ic, in_y, in_x) += weights(oc, ic, ky, kx) * g;
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
