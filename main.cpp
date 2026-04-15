#include <iostream>
#include <iomanip>
#include "nn/Model.h"
#include "Layers/Dense.h"
#include "Layers/ReLU.h"
#include "Layers/Conv2D.h"
#include "Layers/MaxPool2D.h"
#include "Layers/Flatten.h"
#include "nn/Loss.h"
#include "nn/Optimizer.h"
#include "core/Tensor.h"

int main() {
    std::cout << "--- MetalNet CNN Training ---" << std::endl;

    Model model;
    
    // First Block: Conv -> ReLU -> Pool
    model.add(new Conv2D(1, 4, 3, 1, 1)); 
    model.add(new ReLU());
    model.add(new MaxPool2D(2, 2));       
    
    // Second Block: Conv -> ReLU -> Pool
    model.add(new Conv2D(4, 8, 3, 1, 1)); 
    model.add(new ReLU());
    model.add(new MaxPool2D(2, 2));       
    
    model.add(new Flatten());
    
    // Final Layers: Dense -> ReLU -> Dense
    model.add(new Dense(392, 64)); 
    model.add(new ReLU());
    model.add(new Dense(64, 10)); 
    
    MSELoss criterion;
    SGD optimizer(0.01f);
    
    // Input: 2 images of 28x28 (grayscale)
    Tensor batch_x(2, 1, 28, 28);
    batch_x.randomize(); 
    
    // Targets: One-hot encoded labels for 2 images
    Tensor batch_y(2, 10);
    batch_y.fill(0.0f);
    batch_y(0, 3) = 1.0f; 
    batch_y(1, 7) = 1.0f; 
    
    std::cout << "Starting Training loop..." << std::endl;
    for(int epoch = 1; epoch <= 10; epoch++) {
        Tensor preds = model.forward(batch_x);
        
        float loss = criterion.forward(preds, batch_y);
        
        Tensor grad_out = criterion.backward(preds, batch_y);
        model.backward(grad_out);
        
        optimizer.step(model.layers);
        
        std::cout << "Epoch " << epoch << " | Loss: " << std::fixed << std::setprecision(5) << loss << std::endl;
    }
    
    std::cout << "Done! Everything looks good." << std::endl;

    return 0;
}