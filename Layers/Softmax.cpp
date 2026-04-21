#include "Softmax.h"
#include <cmath>
#include <algorithm>
#include <limits>

Tensor Softmax::forward(const Tensor& input) {
    int batches = input.shape[0];
    int classes = input.shape[1];
    Tensor output(input.shape);

    for (int b = 0; b < batches; ++b) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < classes; ++j) {
            max_val = std::max(max_val, input(b, j));
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < classes; ++j) {
            sum_exp += std::exp(input(b, j) - max_val);
        }

        for (int j = 0; j < classes; ++j) {
            output(b, j) = std::exp(input(b, j) - max_val) / sum_exp;
        }
    }

    output_cache = output;
    return output;
}

Tensor Softmax::backward(const Tensor& grad_output) {
    int batches = grad_output.shape[0];
    int classes = grad_output.shape[1];
    Tensor d_input(grad_output.shape);

    for (int b = 0; b < batches; ++b) {
        float dot_product = 0.0f;
        for (int j = 0; j < classes; ++j) {
            dot_product += output_cache(b, j) * grad_output(b, j);
        }

        for (int j = 0; j < classes; ++j) {
            d_input(b, j) = output_cache(b, j) * (grad_output(b, j) - dot_product);
        }
    }

    return d_input;
}
