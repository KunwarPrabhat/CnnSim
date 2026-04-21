#include "Model.h"

void Model::add(Layer* layer) {
    layers.push_back(layer);
}

Tensor Model::forward(const Tensor& input) {
    Tensor output = input;
    for (auto layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void Model::backward(const Tensor& grad_output) {
    Tensor d_out = grad_output;
    // Iterate backwards through all layers
    for (int i = layers.size() - 1; i >= 0; --i) {
        d_out = layers[i]->backward(d_out);
    }
}

void Model::train() {
    for (auto layer : layers) {
        layer->train();
    }
}

void Model::eval() {
    for (auto layer : layers) {
        layer->eval();
    }
}