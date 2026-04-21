#pragma once
#include <string>
#include <vector>
#include "../core/Tensor.h"

class Dataset {
public:
    Tensor images; // (N, C=1, H, W)
    Tensor labels; // (N, num_classes)

    // Load a directory of images or just a few explicit paths for testing
    // To keep it simple, we supply static utilities.
    
    // Reads a P6 (Binary) or P3 (ASCII) PPM file, converts to Grayscale (1, 1, H, W)
    static Tensor load_ppm_grayscale(const std::string& filepath);

    // Reads a PPM file, returns RGB (1, 3, H, W)
    static Tensor load_ppm_rgb(const std::string& filepath);

    void normalize(); // scales 0-255 images to 0.0-1.0
};
