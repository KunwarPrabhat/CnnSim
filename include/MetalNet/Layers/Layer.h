#pragma once
#include "../core/Tensor.h"
#include <vector>
#include <memory>
#include <stdexcept>
#include <string>

namespace MetalNet {

class Layer {
public:
    virtual ~Layer() = default;

    virtual void compile(const std::vector<std::vector<int>>& input_shapes) = 0;

    Tensor output_buffer;
    Tensor grad_input_buffer;
    Tensor grad_output_buffer;
    std::vector<Tensor> multi_grad_input_buffers;
    Tensor temp_add_buffer;
    
    std::vector<std::shared_ptr<Layer>> input_nodes;
    std::vector<std::shared_ptr<Layer>> output_nodes;
    std::vector<std::shared_ptr<Layer>> extra_input_nodes;
    std::vector<const Tensor*> extra_inputs;
    
    const Tensor* cached_input_ptr = nullptr;
    std::vector<const Tensor*> cached_input_ptrs;
    bool   has_grad_cache = false;
    bool   is_training    = true;

    inline void add_extra_input_source(std::shared_ptr<Layer> parent) {
        extra_input_nodes.push_back(parent);
    }

    virtual Tensor& forward(const Tensor& input) {
        throw std::runtime_error("forward(Tensor) not implemented");
    }
    virtual void backward(const Tensor& grad) {
        throw std::runtime_error("backward(Tensor) not implemented");
    }
    virtual Tensor& forward(const std::vector<const Tensor*>& inputs) {
        if (inputs.size() == 1) {
            return forward(*inputs[0]);
        }
        throw std::runtime_error("Multi-input forward not implemented");
    }
    inline Tensor& forward(std::initializer_list<const Tensor*> inputs) {
        return forward(std::vector<const Tensor*>(inputs));
    }
    virtual void backward_multi(const Tensor& grad) {
        backward(grad);
    }
    
    virtual void update_weights(float lr) {}
    virtual void train() { is_training = true; }
    virtual void eval()  { is_training = false; }
    virtual std::vector<Tensor*> get_parameters() { return {}; }
    virtual std::vector<Tensor*> get_states()     { return get_parameters(); }
    virtual std::string name() const = 0;
};

} // namespace MetalNet
