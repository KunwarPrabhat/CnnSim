#pragma once
#include "Dataset.h"
#include <vector>
#include <utility>

class DataLoader {
public:
    Dataset* dataset;
    int batch_size;
    bool shuffle;
    std::vector<int> indices;
    int current_idx;

    DataLoader(Dataset* ds, int batch_size, bool shuffle = true);

    bool has_next() const;
    std::pair<Tensor, Tensor> next_batch();
    void reset();
};
