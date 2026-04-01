#include "MaxPool2D.h"
#include <limits>
#include <iostream>

MaxPool2D::MaxPool2D(int p_size, int s) : pool_size(p_size), stride(s) {}

Tensor MaxPool2D::forward(const Tensor& input) {
    cached_input = input;

    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];

    int out_h = (H - pool_size) / stride + 1;
    int out_w = (W - pool_size) / stride + 1;

    Tensor output(N, C, out_h, out_w);
    max_indices = Tensor(input.shape); // To store where the max came from (1 if max, 0 else)
    max_indices.fill(0.0f);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_y = -1;
                    int max_x = -1;

                    for (int py = 0; py < pool_size; ++py) {
                        for (int px = 0; px < pool_size; ++px) {
                            int in_y = y * stride + py;
                            int in_x = x * stride + px;

                            float val = input(n, c, in_y, in_x);
                            if (val > max_val) {
                                max_val = val;
                                max_y = in_y;
                                max_x = in_x;
                            }
                        }
                    }

                    output(n, c, y, x) = max_val;
                    if (max_y != -1 && max_x != -1) {
                         // We track the mask 1 for gradients to flow through.
                         // But if overlapping maxpools occur, they can accumulate.
                         max_indices(n, c, max_y, max_x) += 1.0f;
                    }
                }
            }
        }
    }

    return output;
}

Tensor MaxPool2D::backward(const Tensor& grad_output) {
    int N = cached_input.shape[0];
    int C = cached_input.shape[1];
    int H = cached_input.shape[2];
    int W = cached_input.shape[3];

    int out_h = grad_output.shape[2];
    int out_w = grad_output.shape[3];

    Tensor d_input(N, C, H, W);
    d_input.fill(0.0f);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < out_h; ++y) {
                for (int x = 0; x < out_w; ++x) {
                    float g_y = grad_output(n, c, y, x);

                    // Re-run the max search to map gradient. 
                    // Technically we can just do the max loop again to map the `g_y` to the max_y, max_x. 
                    // Let's do that for simplicity instead of the generic mask if overlapping pools occur.
                    
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_y = -1, max_x = -1;

                    for (int py = 0; py < pool_size; ++py) {
                        for (int px = 0; px < pool_size; ++px) {
                            int in_y = y * stride + py;
                            int in_x = x * stride + px;
                            float val = cached_input(n, c, in_y, in_x);
                            if (val > max_val) {
                                max_val = val;
                                max_y = in_y;
                                max_x = in_x;
                            }
                        }
                    }

                    if (max_y != -1 && max_x != -1) {
                        d_input(n, c, max_y, max_x) += g_y;
                    }
                }
            }
        }
    }

    return d_input;
}
