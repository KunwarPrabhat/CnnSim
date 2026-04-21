#include "Model.h"
#include <fstream>
#include <stdexcept>

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

void Model::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for saving: " + filename);
    }
    for (auto layer : layers) {
        for (Tensor* state : layer->get_states()) {
            file.write(reinterpret_cast<const char*>(state->data.data()), state->data.size() * sizeof(float));
        }
    }
}

void Model::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for loading: " + filename);
    }
    for (auto layer : layers) {
        for (Tensor* state : layer->get_states()) {
            file.read(reinterpret_cast<char*>(state->data.data()), state->data.size() * sizeof(float));
        }
    }
}