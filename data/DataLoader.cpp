#include "DataLoader.h"
#include <algorithm>
#include <random>

DataLoader::DataLoader(Dataset* ds, int b_size, bool shuf) 
    : dataset(ds), batch_size(b_size), shuffle(shuf), current_idx(0) {
    if (dataset && dataset->images.dims() > 0) {
        int n_samples = dataset->images.shape[0];
        indices.resize(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            indices[i] = i;
        }
        reset();
    }
}

bool DataLoader::has_next() const {
    if (!dataset || dataset->images.dims() == 0) return false;
    return current_idx < dataset->images.shape[0];
}

void DataLoader::reset() {
    current_idx = 0;
    if (shuffle && !indices.empty()) {
        std::mt19937 g(42); // Fixed seed for reproducibility for now
        std::shuffle(indices.begin(), indices.end(), g);
    }
}

std::pair<Tensor, Tensor> DataLoader::next_batch() {
    int n_samples = dataset->images.shape[0];
    int end_idx = std::min(current_idx + batch_size, n_samples);
    int current_batch_size = end_idx - current_idx;

    int c = dataset->images.shape[1];
    int h = dataset->images.shape[2];
    int w = dataset->images.shape[3];
    int num_classes = dataset->labels.shape[1];

    Tensor batch_x(current_batch_size, c, h, w);
    Tensor batch_y(current_batch_size, num_classes);

    for (int b = 0; b < current_batch_size; ++b) {
        int real_idx = indices[current_idx + b];
        
        // Copy image data
        for (int ch = 0; ch < c; ++ch) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    batch_x(b, ch, y, x) = dataset->images(real_idx, ch, y, x);
                }
            }
        }
        
        // Copy label data
        for (int cls = 0; cls < num_classes; ++cls) {
            batch_y(b, cls) = dataset->labels(real_idx, cls);
        }
    }

    current_idx = end_idx;
    return {batch_x, batch_y};
}
