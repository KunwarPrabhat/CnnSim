#pragma once
#include <vector>
#include <stdexcept>
#include <numeric>

class Tensor {
public:
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> grad;

    // Constructors
    Tensor();
    Tensor(int rows, int cols);
    Tensor(int n, int c, int h, int w);
    Tensor(std::vector<int> target_shape);

    // Initializers
    void randomize();
    void zero_grad();
    void fill(float value);

    // Getters
    int size() const;
    int dims() const;

    // Accessors
    int get_index(const std::vector<int>& indices) const;
    int get_index(int i, int j) const;
    int get_index(int n, int c, int h, int w) const;

    float& operator()(int i, int j);
    float operator()(int i, int j) const;

    float& operator()(int n, int c, int h, int w);
    float operator()(int n, int c, int h, int w) const;
};