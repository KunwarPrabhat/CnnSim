#pragma once
#include "../core/Tensor.h"

class Layer {
public:
    virtual ~Layer() = default;

    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    
    // Default empty implementation for layers without parameters (e.g. ReLU, MaxPool)
    virtual void update_weights(float learning_rate) {}

protected:
    Tensor cached_input; // Used to store input during forward pass to compute gradients in backward pass

public:
    bool is_training = true;
    
    virtual void train() { is_training = true; }
    virtual void eval() { is_training = false; }
    
    virtual std::vector<Tensor*> get_parameters() {
        return {}; // Return empty by default
    }
};