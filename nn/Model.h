#pragma once
#include <vector>
#include "../Layers/Layer.h"
#include "Optimizer.h"
#include "Loss.h"

class Model {
public:
    std::vector<Layer*> layers;

    void add(Layer* layer);
    Tensor forward(const Tensor& input);
    void backward(const Tensor& grad_output);

    void train();
    void eval();
};