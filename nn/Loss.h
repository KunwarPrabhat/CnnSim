#pragma once
#include "../core/Tensor.h"

class Loss {
public:
    virtual ~Loss() = default;
    virtual float forward(const Tensor& preds, const Tensor& targets) = 0;
    virtual Tensor backward(const Tensor& preds, const Tensor& targets) = 0;
};

class MSELoss : public Loss {
public:
    float forward(const Tensor& preds, const Tensor& targets) override;
    Tensor backward(const Tensor& preds, const Tensor& targets) override;
};

class CrossEntropyLoss : public Loss {
public:
    // Expects integer class labels in targets for simplicity, or one-hot?
    // Let's assume one-hot encoded targets of same shape as preds (N, Classes).
    float forward(const Tensor& preds, const Tensor& targets) override;
    Tensor backward(const Tensor& preds, const Tensor& targets) override;
};
