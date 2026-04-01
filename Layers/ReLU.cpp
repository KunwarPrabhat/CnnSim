#include "ReLU.h"

Tensor ReLU::forward(const Tensor& input) {
    cached_input = input;

    Tensor output(input.shape);
    for (int i = 0; i < input.size(); ++i) {
        output.data[i] = input.data[i] > 0 ? input.data[i] : 0.0f;
    }

    return output;
}

Tensor ReLU::backward(const Tensor& grad_output) {
    Tensor d_input(cached_input.shape);
    for (int i = 0; i < cached_input.size(); ++i) {
        d_input.data[i] = cached_input.data[i] > 0 ? grad_output.data[i] : 0.0f;
    }

    return d_input;
}