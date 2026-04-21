#include "BatchNorm2D.h"
#include <cmath>

BatchNorm2D::BatchNorm2D(int features, float momentum, float eps)
    : num_features(features), momentum(momentum), eps(eps) {
    gamma = Tensor(features, 1);
    beta = Tensor(features, 1);
    running_mean = Tensor(features, 1);
    running_var = Tensor(features, 1);

    gamma.fill(1.0f);
    beta.fill(0.0f);
    running_mean.fill(0.0f);
    running_var.fill(1.0f);
}

std::vector<Tensor*> BatchNorm2D::get_parameters() {
    return {&gamma, &beta};
}

std::vector<Tensor*> BatchNorm2D::get_states() {
    return {&gamma, &beta, &running_mean, &running_var};
}

Tensor BatchNorm2D::forward(const Tensor& input) {
    int batches = input.shape[0];
    int channels = input.shape[1];
    int height = input.shape[2];
    int width = input.shape[3];
    int m = batches * height * width;

    Tensor output(input.shape);
    
    if (is_training) {
        Tensor batch_mean(channels, 1);
        batch_var = Tensor(channels, 1);
        batch_mean.fill(0.0f);
        batch_var.fill(0.0f);

        // Compute mean
        for (int b = 0; b < batches; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        batch_mean(c, 0) += input(b, c, h, w);
                    }
                }
            }
        }
        for (int c = 0; c < channels; ++c) {
            batch_mean(c, 0) /= m;
        }

        // Compute variance
        for (int b = 0; b < batches; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        float diff = input(b, c, h, w) - batch_mean(c, 0);
                        batch_var(c, 0) += diff * diff;
                    }
                }
            }
        }
        for (int c = 0; c < channels; ++c) {
            batch_var(c, 0) /= m;
        }

        // Apply momentum to running stats
        for (int c = 0; c < channels; ++c) {
            running_mean(c, 0) = (1 - momentum) * running_mean(c, 0) + momentum * batch_mean(c, 0);
            running_var(c, 0) = (1 - momentum) * running_var(c, 0) + momentum * batch_var(c, 0);
        }

        x_norm = Tensor(input.shape);
        // Normalize and scale/shift
        for (int b = 0; b < batches; ++b) {
            for (int c = 0; c < channels; ++c) {
                float inv_std = 1.0f / std::sqrt(batch_var(c, 0) + eps);
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        x_norm(b, c, h, w) = (input(b, c, h, w) - batch_mean(c, 0)) * inv_std;
                        output(b, c, h, w) = gamma(c, 0) * x_norm(b, c, h, w) + beta(c, 0);
                    }
                }
            }
        }
    } else {
        // Inference uses running stats
        for (int b = 0; b < batches; ++b) {
            for (int c = 0; c < channels; ++c) {
                float inv_std = 1.0f / std::sqrt(running_var(c, 0) + eps);
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        float norm = (input(b, c, h, w) - running_mean(c, 0)) * inv_std;
                        output(b, c, h, w) = gamma(c, 0) * norm + beta(c, 0);
                    }
                }
            }
        }
    }

    return output;
}

Tensor BatchNorm2D::backward(const Tensor& grad_output) {
    int batches = grad_output.shape[0];
    int channels = grad_output.shape[1];
    int height = grad_output.shape[2];
    int width = grad_output.shape[3];
    int m = batches * height * width;

    Tensor d_input(grad_output.shape);
    
    // Accumulate gradients for gamma and beta
    for (int b = 0; b < batches; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    float g = grad_output(b, c, h, w);
                    gamma.grad[c] += g * x_norm(b, c, h, w);
                    beta.grad[c] += g;
                }
            }
        }
    }

    // Compute gradient for input using the stabilized formula
    for (int c = 0; c < channels; ++c) {
        float inv_std = 1.0f / std::sqrt(batch_var(c, 0) + eps);
        float gamma_val = gamma(c, 0);
        
        float sum_d_out = 0.0f;
        float sum_d_out_x_norm = 0.0f;
        
        for (int b = 0; b < batches; ++b) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    sum_d_out += grad_output(b, c, h, w);
                    sum_d_out_x_norm += grad_output(b, c, h, w) * x_norm(b, c, h, w);
                }
            }
        }
        
        for (int b = 0; b < batches; ++b) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    d_input(b, c, h, w) = (gamma_val * inv_std / m) * 
                        (m * grad_output(b, c, h, w) - sum_d_out - x_norm(b, c, h, w) * sum_d_out_x_norm);
                }
            }
        }
    }

    return d_input;
}
