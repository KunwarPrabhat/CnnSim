#pragma once
#include <vector>
#include "../Layers/Layer.h"

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(std::vector<Layer*>& layers) = 0;
};

class SGD : public Optimizer {
public:
    float learning_rate;

    SGD(float lr = 0.01f);
    void step(std::vector<Layer*>& layers) override;
};
