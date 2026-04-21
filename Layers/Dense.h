#pragma once
#include "Layer.h"

class Dense : public Layer {
public:
    int input_size;
    int output_size;

    Tensor weights;
    Tensor biases;

    Dense(int in_size, int out_size);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update_weights(float learning_rate) override;
    std::vector<Tensor*> get_parameters() override;
};