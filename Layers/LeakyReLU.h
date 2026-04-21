#pragma once
#include "Layer.h"

class LeakyReLU : public Layer {
public:
    float negative_slope;

    LeakyReLU(float alpha = 0.01f);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
};
