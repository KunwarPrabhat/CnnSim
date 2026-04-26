#include "include/MetalNet/MetalNet.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace MetalNet;

int main() {
    // Master Timer Start
    auto program_start_time = std::chrono::high_resolution_clock::now();

    std::cout << "====================================================" << std::endl;
    std::cout << "--- MetalNet High-Performance Universal Benchmark ---" << std::endl;
    std::cout << "====================================================" << std::endl;

    // ---------------------------------------------------------
    // 1. DYNAMIC DATA I/O (No more hardcoded sizes!)
    // ---------------------------------------------------------
    std::cout << "[1/7] Scouting and Loading CSV: data/mnist_train.csv..." << std::endl;

    Dataset full_ds;
    try {
        // API: load_csv(path, has_header, num_classes, image_shape, manual_label_col)
        full_ds = Dataset::load_csv("mnist_train.csv", true, 10, {1, 28, 28}, 0);
        full_ds.normalize(); // Explictly map 0-255 pixels to 0-1 for CNN training
    } catch (const std::exception& e) {
        std::cerr << "      FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }
    
    // The engine figured out the dataset size dynamically!
    int total_samples = full_ds.images.shape[0];
    std::cout << "      SUCCESS: Dynamically loaded " << total_samples << " samples." << std::endl;

    // ---------------------------------------------------------
    // 2. ZERO-COPY DATA SPLITTING (50/50)
    // ---------------------------------------------------------
    std::cout << "[2/7] Splitting data into 50% Train / 50% Test..." << std::endl;
    int split_idx = total_samples / 2;
    Dataset train_ds, test_ds;

    train_ds.images = Tensor(split_idx, 1, 28, 28);
    train_ds.labels = Tensor(split_idx, 10);
    test_ds.images  = Tensor(total_samples - split_idx, 1, 28, 28);
    test_ds.labels  = Tensor(total_samples - split_idx, 10);

    std::copy(full_ds.images.data.begin(), full_ds.images.data.begin() + train_ds.images.size(), train_ds.images.data.begin());
    std::copy(full_ds.labels.data.begin(), full_ds.labels.data.begin() + train_ds.labels.size(), train_ds.labels.data.begin());
    std::copy(full_ds.images.data.begin() + train_ds.images.size(), full_ds.images.data.end(), test_ds.images.data.begin());
    std::copy(full_ds.labels.data.begin() + train_ds.labels.size(), full_ds.labels.data.end(), test_ds.labels.data.begin());

    std::cout << "      Split Complete: " << split_idx << " samples per set." << std::endl;

    // ---------------------------------------------------------
    // 3. DEEP ARCHITECTURE DEFINITION 
    // ---------------------------------------------------------
    std::cout << "[3/7] Defining Deep Model Architecture (VGG-Style)..." << std::endl;
    Model model;
    
    // STAGE 1: 64 Filters
    model << conv2d(1, 64, 3, 1, 1)  << batchnorm2d(64) << leaky_relu(0.01f)
          << conv2d(64, 64, 3, 1, 1) << batchnorm2d(64) << leaky_relu(0.01f)
          << maxpool2d(2, 2) // 28x28 shrinks to 14x14

    // STAGE 2: 128 Filters
          << conv2d(64, 128, 3, 1, 1)  << batchnorm2d(128) << leaky_relu(0.01f)
          << conv2d(128, 128, 3, 1, 1) << batchnorm2d(128) << leaky_relu(0.01f)
          << maxpool2d(2, 2) // 14x14 shrinks to 7x7

    // CLASSIFIER HEAD
          << flatten() 
          << dense(128 * 7 * 7, 10); // Flattened size: 6272 -> 10

    // ---------------------------------------------------------
    // 4. ENGINE COMPILATION & HYPERPARAMETERS
    // ---------------------------------------------------------
    const int batch_size = 128; 
    const int num_epochs = 2; 
    std::cout << "[4/7] Compiling Graph (Pre-allocating Zero-Allocation Buffers)..." << std::endl;
    model.compile({batch_size, 1, 28, 28});

    // DataLoader: (dataset, batch_size, shuffle, augment). Augment is FALSE for digits!
    DataLoader train_loader(&train_ds, batch_size, true, false); 
    
    float current_lr = 0.001f;
    Adam optimizer(current_lr, 0.9f, 0.999f, 1e-8f, 0.0001f); 
    CrossEntropyLoss criterion;

    // ---------------------------------------------------------
    // 5. CORE TRAINING LOOP & LR SCHEDULER
    // ---------------------------------------------------------
    std::cout << "[5/7] Starting Training Loop (" << num_epochs << " Epochs)..." << std::endl;
    auto train_start_time = std::chrono::high_resolution_clock::now();

    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        
        if (epoch == 40 || epoch == 75) {
            current_lr *= 0.1f;
            optimizer.lr = current_lr; 
            std::cout << "\n   >>> [SCHEDULER] Learning Rate dropped to: " << current_lr << " <<<\n" << std::endl;
        }

        train_loader.reset();
        model.train();
        float epoch_loss = 0;
        int batches = 0;

        while (train_loader.has_next()) {
            auto [bx, by] = train_loader.next_batch();
            
            Tensor& preds = model.forward(bx);
            epoch_loss += criterion.forward(preds, by);
            model.backward(criterion.backward(preds, by));
            
            const float clip_val = 1.0f;
            for (auto& layer : model.layers) {
                for (Tensor* p : layer->get_parameters()) {
                    for (int i = 0; i < p->size(); ++i) {
                        float g = p->grad_ptr()[i];
                        if (std::isnan(g) || std::isinf(g)) { g = 0.0f; }
                        p->grad_ptr()[i] = std::clamp(g, -clip_val, clip_val);
                    }
                }
            }
            
            optimizer.step(model.layers);
            batches++;
        }
        std::cout << "      Epoch " << std::setw(3) << epoch << "/" << num_epochs 
                  << " | Avg Loss: " << std::fixed << std::setprecision(4) << (epoch_loss / batches) << std::endl;
    }

    auto train_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> training_duration = train_end_time - train_start_time;
    double train_minutes = training_duration.count() / 60.0;
    std::cout << "      Training Finished in " << std::fixed << std::setprecision(2) << train_minutes << " minutes." << std::endl;

    // ---------------------------------------------------------
    // 6. INFERENCE & EVALUATION
    // ---------------------------------------------------------
    std::cout << "[6/7] Running Inference on Unseen Test Split..." << std::endl;
    model.eval();
    DataLoader test_loader(&test_ds, batch_size, false, false);
    int correct = 0, total = 0;

    while (test_loader.has_next()) {
        auto [tx, ty] = test_loader.next_batch();
        Tensor& out = model.forward(tx);
        
        for (int b = 0; b < tx.shape[0]; ++b) {
            int p_idx = 0, t_idx = 0;
            float p_max = -1e9, t_max = -1e9;
            for (int i = 0; i < 10; ++i) {
                if (out(b, i) > p_max) { p_max = out(b, i); p_idx = i; }
                if (ty(b, i) > t_max)  { t_max = ty(b, i);  t_idx = i; }
            }
            if (p_idx == t_idx) correct++;
            total++; // Used internal loop to count safely
        }
    }

    auto program_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_program_duration = program_end_time - program_start_time;
    double total_minutes = total_program_duration.count() / 60.0;

    // ---------------------------------------------------------
    // 7. FINAL REPORTING
    // ---------------------------------------------------------
    float final_acc = (float)correct / (total_samples - split_idx) * 100.0f;
    std::cout << "\n[7/7] --- FINAL PERFORMANCE REPORT ---" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Total Samples Managed : " << total_samples << std::endl;
    std::cout << "Epochs Completed      : " << num_epochs << std::endl;
    std::cout << "Isolated Training Time: " << std::fixed << std::setprecision(2) << train_minutes << " minutes" << std::endl;
    std::cout << "Total Execution Time  : " << std::fixed << std::setprecision(2) << total_minutes << " minutes" << std::endl;
    std::cout << "Throughput            : " << (split_idx * num_epochs) / training_duration.count() << " images/sec" << std::endl;
    std::cout << "Final Test Accuracy   : " << std::setprecision(2) << final_acc << "%" << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    return 0;
}