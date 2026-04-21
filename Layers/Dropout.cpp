#include "Dropout.h"

Dropout::Dropout(float drop_rate) : rate(drop_rate) {}

Tensor Dropout::forward(const Tensor& input) {
    if (!is_training) {
        return input;
    }

    mask = Tensor(input.shape);
    Tensor output(input.shape);
    float scale = 1.0f / (1.0f - rate);

    for (int i = 0; i < input.size(); ++i) {
        // Generate random float between 0 and 1
        float rand_val = (float)rand() / RAND_MAX;
        
        if (rand_val > rate) { // Keep with probability (1 - rate)
            mask.data[i] = 1.0f;
            output.data[i] = input.data[i] * scale;
        } else {
            mask.data[i] = 0.0f;
            output.data[i] = 0.0f;
        }
    }

    return output;
}

Tensor Dropout::backward(const Tensor& grad_output) {
    if (!is_training) {
        return grad_output;
    }

    Tensor d_input(grad_output.shape);
    float scale = 1.0f / (1.0f - rate);

    for (int i = 0; i < grad_output.size(); ++i) {
        d_input.data[i] = grad_output.data[i] * mask.data[i] * scale;
    }

    return d_input;
}
