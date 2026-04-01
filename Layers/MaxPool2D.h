#pragma once
#include "Layer.h"

class MaxPool2D : public Layer {
public:
    int pool_size;
    int stride;

    Tensor max_indices; // Store max locations for backward pass

    MaxPool2D(int p_size, int s = 2);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
};
