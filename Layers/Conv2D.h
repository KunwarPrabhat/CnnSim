#pragma once
#include "Layer.h"

class Conv2D : public Layer {
public:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;

    Tensor weights; // Shape (out_channels, in_channels, kernel_size, kernel_size)
    Tensor biases;  // Shape (out_channels)

    Conv2D(int in_c, int out_c, int k_size, int s = 1, int p = 0);
    
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    void update_weights(float learning_rate) override;
    std::vector<Tensor*> get_parameters() override;
};
