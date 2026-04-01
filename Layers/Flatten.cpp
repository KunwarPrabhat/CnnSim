#include "Flatten.h"
#include <numeric>

Tensor Flatten::forward(const Tensor& input) {
    cached_input = input;

    int N = input.shape[0];
    int flat_dim = 1;
    for (size_t i = 1; i < input.shape.size(); ++i) {
        flat_dim *= input.shape[i];
    }

    Tensor output(std::vector<int>{N, flat_dim});
    output.data = input.data; // copy values

    return output;
}

Tensor Flatten::backward(const Tensor& grad_output) {
    Tensor d_input(cached_input.shape);
    d_input.data = grad_output.data; // copy back grads
    return d_input;
}
