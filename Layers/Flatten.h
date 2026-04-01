#pragma once
#include "Layer.h"

class Flatten : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
};
