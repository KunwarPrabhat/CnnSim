#include "Tensor.h"
#include <cstdlib>

Tensor::Tensor() : shape({}) {}

Tensor::Tensor(int rows, int cols) : shape({rows, cols}) {
    data.resize(rows * cols, 0.0f);
    grad.resize(rows * cols, 0.0f);
}

Tensor::Tensor(int n, int c, int h, int w) : shape({n, c, h, w}) {
    data.resize(n * c * h * w, 0.0f);
    grad.resize(n * c * h * w, 0.0f);
}

Tensor::Tensor(std::vector<int> target_shape) : shape(target_shape) {
    int total_size = 1;
    for (int dim : shape) {
        total_size *= dim;
    }
    data.resize(total_size, 0.0f);
    grad.resize(total_size, 0.0f);
}

void Tensor::randomize() {
    for (int i = 0; i < data.size(); ++i) {
        data[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * 0.01f;
    }
}

void Tensor::zero_grad() {
    for (int i = 0; i < grad.size(); ++i) {
        grad[i] = 0.0f;
    }
}

void Tensor::fill(float value) {
    for (int i = 0; i < data.size(); ++i) {
        data[i] = value;
    }
}

int Tensor::size() const {
    return data.size();
}

int Tensor::dims() const {
    return shape.size();
}

int Tensor::get_index(const std::vector<int>& indices) const {
    int idx = 0;
    int multiplier = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        idx += indices[i] * multiplier;
        multiplier *= shape[i];
    }
    return idx;
}

int Tensor::get_index(int i, int j) const {
    return i * shape[1] + j;
}

int Tensor::get_index(int n, int c, int h, int w) const {
    int C = shape[1], H = shape[2], W = shape[3];
    return n * (C * H * W) + c * (H * W) + h * W + w;
}

float& Tensor::operator()(int i, int j) {
    return data[get_index(i, j)];
}

float Tensor::operator()(int i, int j) const {
    return data[get_index(i, j)];
}

float& Tensor::operator()(int n, int c, int h, int w) {
    return data[get_index(n, c, h, w)];
}

float Tensor::operator()(int n, int c, int h, int w) const {
    return data[get_index(n, c, h, w)];
}
