#include "LeakyReLU.h"

LeakyReLU::LeakyReLU(float alpha) : negative_slope(alpha) {}

Tensor LeakyReLU::forward(const Tensor& input) {
    cached_input = input;
    Tensor output(input.shape);

    for (int i = 0; i < input.size(); ++i) {
        float val = input.data[i];
        output.data[i] = val > 0 ? val : val * negative_slope;
    }

    return output;
}

Tensor LeakyReLU::backward(const Tensor& grad_output) {
    Tensor d_input(grad_output.shape);

    for (int i = 0; i < grad_output.size(); ++i) {
        float val = cached_input.data[i];
        d_input.data[i] = val > 0 ? grad_output.data[i] : grad_output.data[i] * negative_slope;
    }

    return d_input;
}
