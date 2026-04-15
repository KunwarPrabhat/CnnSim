#include "ReLU.h"

Tensor ReLU::forward(const Tensor& input) {
    cached_input = input;

    Tensor output(input.shape);
    for (int i = 0; i < input.size(); ++i) {
        if (input.data[i] > 0) {
            output.data[i] = input.data[i];
        } else {
            output.data[i] = 0.0f;
        }
    }

    return output;
}

Tensor ReLU::backward(const Tensor& grad_output) {
    Tensor d_input(cached_input.shape);
    for (int i = 0; i < cached_input.size(); ++i) {
        if (cached_input.data[i] > 0) {
            d_input.data[i] = grad_output.data[i];
        } else {
            d_input.data[i] = 0.0f;
        }
    }

    return d_input;
}