#pragma once
#include "Layer.h"
#include <cstdlib>

class Dropout : public Layer {
public:
    float rate;
    Tensor mask;

    Dropout(float drop_rate = 0.5f);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
};
