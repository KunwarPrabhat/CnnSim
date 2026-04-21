#pragma once
#include "Layer.h"

class Softmax : public Layer {
public:
    Tensor output_cache;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
};
