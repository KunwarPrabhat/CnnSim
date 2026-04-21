#pragma once
#include "Layer.h"
#include <vector>

class BatchNorm2D : public Layer {
public:
    int num_features;
    float momentum;
    float eps;

    Tensor gamma;
    Tensor beta;
    Tensor running_mean;
    Tensor running_var;

    // Cache for backward pass
    Tensor x_norm;
    Tensor batch_var;

    BatchNorm2D(int features, float momentum = 0.1f, float eps = 1e-5f);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    
    std::vector<Tensor*> get_parameters() override;
};
