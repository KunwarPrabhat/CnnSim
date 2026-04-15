#include "MaxPool2D.h"
#include <limits>
#include <iostream>

MaxPool2D::MaxPool2D(int p_size, int s) : pool_size(p_size), stride(s) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    cached_input = input;

    int batches = input.shape[0];
    int channels = input.shape[1];
    int height = input.shape[2];
    int width = input.shape[3];

    int out_h = (height - pool_size) / stride + 1;
    int out_w = (width - pool_size) / stride + 1;

    Tensor output(batches, channels, out_h, out_w);

    for (int b = 0; b < batches; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    float max_val = -std::numeric_limits<float>::infinity();

                    for (int py = 0; py < pool_size; ++py) {
                        for (int px = 0; px < pool_size; ++px) {
                            int in_y = y * stride + py;
                            int in_x = x * stride + px;

                            float val = input(b, c, in_y, in_x);
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                    output(b, c, y, x) = max_val;
                }
            }
        }
    }

    return output;
}

Tensor MaxPool2D::backward(const Tensor& grad_output) {
    int batches = cached_input.shape[0];
    int channels = cached_input.shape[1];
    int height = cached_input.shape[2];
    int width = cached_input.shape[3];

    int out_h = grad_output.shape[2];
    int out_w = grad_output.shape[3];

    Tensor d_input(batches, channels, height, width);
    d_input.fill(0.0f);

    for (int b = 0; b < batches; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    float g = grad_output(b, c, y, x);

                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_y = -1, max_x = -1;

                    for (int py = 0; py < pool_size; ++py) {
                        for (int px = 0; px < pool_size; ++px) {
                            int in_y = y * stride + py;
                            int in_x = x * stride + px;
                            float val = cached_input(b, c, in_y, in_x);
                            if (val > max_val) {
                                max_val = val;
                                max_y = in_y;
                                max_x = in_x;
                            }
                        }
                    }

                    if (max_y != -1 && max_x != -1) {
                        d_input(b, c, max_y, max_x) += g;
                    }
                }
            }
        }
    }

    return d_input;
}
