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
    std::cout << "--- MetalNet CNN Pipeline ---" << std::endl;

    // 1. Build Model Pipeline
    Model model;
    
    // Conv -> ReLU -> Pool
    // padding=1, stride=1 keeps 28x28. Pool reduces to 14x14
    model.add(new Conv2D(1, 4, 3, 1, 1)); 
    model.add(new ReLU());
    model.add(new MaxPool2D(2, 2));       
    
    // Conv -> ReLU -> Pool
    // padding=1 keeps 14x14. Pool reduces to 7x7
    model.add(new Conv2D(4, 8, 3, 1, 1)); 
    model.add(new ReLU());
    model.add(new MaxPool2D(2, 2));       
    
    // Flatten
    model.add(new Flatten());
    
    // Dense -> ReLU -> Dense (output)
    // 8 channels * 7 height * 7 width = 392 features
    model.add(new Dense(392, 64)); 
    model.add(new ReLU());
    model.add(new Dense(64, 10)); // 10 classes
    
    // 2. Setup Training Components
    MSELoss criterion;
    SGD optimizer(0.01f);
    
    // 3. Create Dummy Data (e.g., 2 images of 28x28 pixels)
    Tensor batch_x(2, 1, 28, 28);
    batch_x.randomize(); 
    
    // Dummy labels (one-hot encoded)
    Tensor batch_y(2, 10);
    batch_y.fill(0.0f);
    batch_y(0, 3) = 1.0f; // Image 0 is target class 3
    batch_y(1, 7) = 1.0f; // Image 1 is target class 7
    
    // 4. Run Training Loop
    std::cout << "Starting Dummy Training Loop..." << std::endl;
    for(int epoch = 1; epoch <= 10; epoch++) {
        // Forward pass
        Tensor preds = model.forward(batch_x);
        
        // Calculate loss
        float loss = criterion.forward(preds, batch_y);
        
        // Backward pass (calculate gradients)
        Tensor grad_out = criterion.backward(preds, batch_y);
        model.backward(grad_out);
        
        // Step optimizer (update weights)
        optimizer.step(model.layers);
        
        std::cout << "Epoch " << epoch << "\t| MSE Loss: " << std::fixed << std::setprecision(5) << loss << std::endl;
    }
    
    std::cout << "Training complete! Pipeline works perfectly." << std::endl;

    return 0;
}