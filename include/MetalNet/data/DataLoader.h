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
    std::vector<bool> flips;
    std::vector<int>  shift_ys;
    std::vector<int>  shift_xs;
    bool             augment;

    inline DataLoader(Dataset* ds, int bs, bool sh=true, bool drop_last_=true, bool aug=false)
        : dataset(ds), batch_size(bs), shuffle(sh), drop_last(drop_last_), augment(aug), current_idx(0), rng(std::random_device{}()) {
        if (dataset && dataset->images.dims()>0) {
            int n=dataset->images.shape[0];
            indices.resize(n);
            for (int i=0;i<n;++i) indices[i]=i;
            
            // Pre-allocate buffers exactly once.
            int C=dataset->images.shape[1], H=dataset->images.shape[2], W=dataset->images.shape[3];
            int NC=dataset->labels.shape[1];
            batch_x = Tensor(batch_size, C, H, W);
            batch_y = Tensor(batch_size, NC);

            flips.resize(batch_size);
            shift_ys.resize(batch_size);
            shift_xs.resize(batch_size);

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
        
        const float* src_x = dataset->images.data.data();
        const float* src_y = dataset->labels.data.data();
        float* dx = batch_x.data.data(); 
        float* dy = batch_y.data.data();

        if (augment) {
            std::uniform_real_distribution<float> rand_flip(0.0f, 1.0f);
            std::uniform_int_distribution<int> rand_crop(-4, 4);
            for (int b=0; b<bs; ++b) {
                flips[b] = rand_flip(rng) > 0.5f;
                shift_ys[b] = rand_crop(rng);
                shift_xs[b] = rand_crop(rng);
            }
        }

        #pragma omp parallel for schedule(static)
        for (int b=0; b<bs; ++b) {
            int idx = indices[current_idx+b];
            const float* sx = src_x + idx*(C*H*W);
            float*       tx = dx    + b  *(C*H*W);
            
            if (augment) {
                bool flip = flips[b];
                int dy_shift = shift_ys[b];
                int dx_shift = shift_xs[b];
                
                for (int c=0; c<C; ++c) {
                    for (int y=0; y<H; ++y) {
                        int sy = y - dy_shift;
                        for (int x=0; x<W; ++x) {
                            int sx_idx = x - dx_shift;
                            if (flip) sx_idx = (W - 1) - sx_idx;
                            
                            float val = 0.0f;
                            if (sy >= 0 && sy < H && sx_idx >= 0 && sx_idx < W) {
                                val = sx[c * H * W + sy * W + sx_idx];
                            }
                            tx[c * H * W + y * W + x] = val;
                        }
                    }
                }
            } else {
                #pragma omp simd
                for (int i=0; i<C*H*W; ++i) tx[i] = sx[i];
            }

            const float* sy_ptr = src_y + idx*NC;
            float*       ty_ptr = dy    + b  *NC;
            #pragma omp simd
            for (int i=0; i<NC; ++i) ty_ptr[i] = sy_ptr[i];
        }
        
        current_idx = ei;
        return std::pair<Tensor&, Tensor&>(batch_x, batch_y);
    }
};

} // namespace MetalNet
