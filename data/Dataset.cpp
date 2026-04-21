#include "Dataset.h"
#include <iostream>
#include <fstream>
//#include <sstream>
#include <stdexcept>

// Helper to skip comments in PPM headers
static void skipComments(std::ifstream& file) {
    char c;
    file >> c;
    while (c == '#') {
        file.ignore(256, '\n');
        file >> c;
    }
    file.unget();
}

Tensor Dataset::load_ppm_grayscale(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::string magic;
    file >> magic;

    if (magic != "P6" && magic != "P3") {
        throw std::runtime_error("Unsupported PPM format (Must be P3 or P6): " + magic);
    }

    skipComments(file);
    int width, height, max_color;
    file >> width >> height >> max_color;
    file.get(); // consume newline

    Tensor img(1, 1, height, width);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float r, g, b;
            if (magic == "P6") {
                unsigned char rgb[3];
                file.read(reinterpret_cast<char*>(rgb), 3);
                r = rgb[0]; g = rgb[1]; b = rgb[2];
            } else {
                int r_i, g_i, b_i;
                file >> r_i >> g_i >> b_i;
                r = r_i; g = g_i; b = b_i;
            }

            // Convert to grayscale: 0.299 R + 0.587 G + 0.114 B
            float gray = 0.299f * r + 0.587f * g + 0.114f * b;
            img(0, 0, y, x) = gray;
        }
    }

    return img;
}

Tensor Dataset::load_ppm_rgb(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::string magic;
    file >> magic;

    if (magic != "P6" && magic != "P3") {
        throw std::runtime_error("Unsupported PPM format (Must be P3 or P6): " + magic);
    }

    skipComments(file);
    int width, height, max_color;
    file >> width >> height >> max_color;
    file.get(); // consume newline

    Tensor img(1, 3, height, width);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float r, g, b;
            if (magic == "P6") {
                unsigned char rgb[3];
                file.read(reinterpret_cast<char*>(rgb), 3);
                r = rgb[0]; g = rgb[1]; b = rgb[2];
            } else {
                int r_i, g_i, b_i;
                file >> r_i >> g_i >> b_i;
                r = r_i; g = g_i; b = b_i;
            }

            img(0, 0, y, x) = r;
            img(0, 1, y, x) = g;
            img(0, 2, y, x) = b;
        }
    }

    return img;
}

void Dataset::normalize() {
    for (int i = 0; i < images.size(); ++i) {
        images.data[i] /= 255.0f; // Scale 0-255 down to 0-1
    }
}
