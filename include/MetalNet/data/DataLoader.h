#pragma once
#include "Dataset.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <random>

namespace MetalNet {

class DataLoader {
public:
    Dataset*         dataset;
    int              batch_size;
    bool             shuffle;
    bool             drop_last;
    std::vector<int> indices;
    int              current_idx;
    std::mt19937     rng;
    Tensor           batch_x;
    Tensor           batch_y;

    inline DataLoader(Dataset* ds, int bs, bool sh=true, bool drop_last_=true)
        : dataset(ds), batch_size(bs), shuffle(sh), drop_last(drop_last_), current_idx(0), rng(std::random_device{}()) {
        if (dataset && dataset->images.dims()>0) {
            int n=dataset->images.shape[0];
            indices.resize(n);
            for (int i=0;i<n;++i) indices[i]=i;
            
            // Pre-allocate buffers exactly once.
            int C=dataset->images.shape[1], H=dataset->images.shape[2], W=dataset->images.shape[3];
            int NC=dataset->labels.shape[1];
            batch_x = Tensor(batch_size, C, H, W);
            batch_y = Tensor(batch_size, NC);

            reset();
        }
    }

    inline bool has_next() const {
        if (!dataset || dataset->images.dims()==0) return false;
        if (drop_last) return current_idx + batch_size <= dataset->images.shape[0];
        return current_idx < dataset->images.shape[0];
    }

    inline void reset() {
        current_idx=0;
        if (shuffle && !indices.empty()) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }
    }

    inline std::pair<Tensor&, Tensor&> next_batch() {
        const int ns=dataset->images.shape[0];
        const int ei=std::min(current_idx+batch_size, ns);
        const int bs=ei-current_idx;
        const int C=dataset->images.shape[1], H=dataset->images.shape[2], W=dataset->images.shape[3];
        const int NC=dataset->labels.shape[1];
        
        batch_x.shape[0] = bs;
        batch_y.shape[0] = bs;
        
        const float* src_x=dataset->images.data.data();
        const float* src_y=dataset->labels.data.data();
        float* dx=batch_x.data.data(); float* dy=batch_y.data.data();
        for (int b=0;b<bs;++b) {
            int idx=indices[current_idx+b];
            std::span<const float> sx(src_x+idx*(C*H*W), C*H*W);
            std::span<float>       tx(dx   +b  *(C*H*W), C*H*W);
            std::copy(sx.begin(), sx.end(), tx.begin());
            std::span<const float> sy(src_y+idx*NC, NC);
            std::span<float>       ty(dy   +b  *NC, NC);
            std::copy(sy.begin(), sy.end(), ty.begin());
        }
        current_idx=ei;
        return std::pair<Tensor&, Tensor&>(batch_x, batch_y);
    }
};

} // namespace MetalNet
