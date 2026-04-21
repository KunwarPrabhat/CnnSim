#include <iostream>
#include <iomanip>
#include "nn/Model.h"
#include "Layers/Dense.h"
#include "Layers/ReLU.h"
#include "Layers/Conv2D.h"
#include "Layers/MaxPool2D.h"
#include "Layers/Flatten.h"
#include "Layers/Dropout.h"
#include "Layers/BatchNorm2D.h"
#include "Layers/LeakyReLU.h"
#include "Layers/Softmax.h"
#include "nn/Loss.h"
#include "nn/Optimizer.h"
#include "nn/Adam.h"
#include "core/Tensor.h"

int main() {
    std::cout << "--- MetalNet CNN Training (Phase 1) ---" << std::endl;

    Model model;
    
    // First Block: Conv -> BatchNorm2D -> LeakyReLU -> Pool
    model.add(new Conv2D(1, 4, 3, 1, 1)); 
    model.add(new BatchNorm2D(4));
    model.add(new LeakyReLU(0.01f));
    model.add(new MaxPool2D(2, 2));       
    
    // Second Block: Conv -> BatchNorm2D -> LeakyReLU -> Pool
    model.add(new Conv2D(4, 8, 3, 1, 1)); 
    model.add(new BatchNorm2D(8));
    model.add(new LeakyReLU(0.01f));
    model.add(new MaxPool2D(2, 2));       
    
    model.add(new Flatten());
    
    // Final Layers: Dense -> Dropout -> LeakyReLU -> Dense
    model.add(new Dense(392, 64)); 
    model.add(new Dropout(0.5f));
    model.add(new LeakyReLU(0.01f));
    model.add(new Dense(64, 10)); 
    
    CrossEntropyLoss criterion;
    Adam optimizer(0.001f);
    
    // Input: 2 images of 28x28 (grayscale)
    Tensor batch_x(2, 1, 28, 28);
    batch_x.randomize(); 
    
    // Targets: One-hot encoded labels for 2 images
    Tensor batch_y(2, 10);
    batch_y.fill(0.0f);
    batch_y(0, 3) = 1.0f; 
    batch_y(1, 7) = 1.0f; 
    
    std::cout << "Starting Training loop..." << std::endl;
    model.train(); // Setup training mode (Dropout/BatchNorm behavior)

    for(int epoch = 1; epoch <= 10; epoch++) {
        Tensor preds = model.forward(batch_x);
        float loss = criterion.forward(preds, batch_y);
        
        Tensor grad_out = criterion.backward(preds, batch_y);
        model.backward(grad_out);
        optimizer.step(model.layers);
        
        std::cout << "Epoch " << epoch << " | Loss: " << std::fixed << std::setprecision(5) << loss << std::endl;
    }
    
    model.eval(); // Switch to inference mode
    std::cout << "Training complete. Switching to inference... " << std::endl;

    // Test standalone softmax
    Softmax sm;
    sm.eval();
    Tensor test_logits(1, 10);
    test_logits.fill(0.1f);
    test_logits(0, 0) = 2.0f; // High response on first class
    Tensor probs = sm.forward(test_logits);

    std::cout << "Standalone Softmax prob for class 0 (should be highest): " 
              << probs(0, 0) << std::endl;

    std::cout << "Done! Everything looks good." << std::endl;

    return 0;
}