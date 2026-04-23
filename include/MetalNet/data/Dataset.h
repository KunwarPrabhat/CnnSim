#pragma once
#include <string>
#include <fstream>
#include <stdexcept>
#include <charconv>
#include "../core/Tensor.h"

namespace MetalNet {

class Dataset {
public:
    Tensor images, labels;

    static inline void skipComments(std::ifstream& f) {
        char c; f>>c;
        while (c=='#') { f.ignore(256,'\n'); f>>c; }
        f.unget();
    }

    static inline Tensor load_ppm_grayscale(const std::string& fp) {
        std::ifstream f(fp, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open: "+fp);
        std::string m; f>>m;
        if (m!="P6"&&m!="P3") throw std::runtime_error("Bad PPM: "+m);
        skipComments(f);
        int W,H,mc; f>>W>>H>>mc; f.get();
        Tensor img(1,1,H,W);
        float* d=img.data.data();
        for (int y=0;y<H;++y) for (int x=0;x<W;++x) {
            float r,g,b;
            if (m=="P6"){unsigned char px[3];f.read((char*)px,3);r=px[0];g=px[1];b=px[2];}
            else{int ri,gi,bi;f>>ri>>gi>>bi;r=ri;g=gi;b=bi;}
            d[y*W+x]=0.299f*r+0.587f*g+0.114f*b;
        }
        return img;
    }

    static inline Tensor load_ppm_rgb(const std::string& fp) {
        std::ifstream f(fp, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open: "+fp);
        std::string m; f>>m;
        if (m!="P6"&&m!="P3") throw std::runtime_error("Bad PPM: "+m);
        skipComments(f);
        int W,H,mc; f>>W>>H>>mc; f.get();
        Tensor img(1,3,H,W);
        float* r_=img.data.data();
        float* g_=r_+H*W;
        float* b_=g_+H*W;
        for (int y=0;y<H;++y) for (int x=0;x<W;++x) {
            float r,g,b;
            if (m=="P6"){unsigned char px[3];f.read((char*)px,3);r=px[0];g=px[1];b=px[2];}
            else{int ri,gi,bi;f>>ri>>gi>>bi;r=ri;g=gi;b=bi;}
            r_[y*W+x]=r; g_[y*W+x]=g; b_[y*W+x]=b;
        }
        return img;
    }

    inline void normalize() {
        float* d=images.data.data();
        #pragma omp simd
        for (int i=0;i<images.size();++i) d[i]/=255.0f;
    }

    static inline Dataset load_csv(const std::string& path, int num_samples, int num_pixels, bool has_header = false, int num_classes = 10) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open CSV file: " + path);
        }

        Dataset ds;
        int h = 1, w = num_pixels;
        if (num_pixels == 784) { h = 28; w = 28; } 

        ds.images = Tensor(num_samples, 1, h, w);
        ds.labels = Tensor(num_samples, num_classes);
        ds.labels.fill(0.0f);

        std::string line;
        line.reserve(8192);

        if (has_header) {
            std::getline(file, line);
        }

        float* img_data = ds.images.data.data();
        float* lbl_data = ds.labels.data.data();
        int img_offset = 0;
        int lbl_offset = 0;

        for (int i = 0; i < num_samples; ++i) {
            if (!std::getline(file, line)) break;
            
            size_t start = 0;
            size_t end = line.find(',');
            if (end == std::string::npos) continue;

            int label = 0;
            std::from_chars(line.data() + start, line.data() + end, label);
            if (label >= 0 && label < num_classes) {
                lbl_data[lbl_offset + label] = 1.0f;
            }
            lbl_offset += num_classes;
            
            start = end + 1;
            
            for (int p = 0; p < num_pixels; ++p) {
                end = line.find(',', start);
                if (end == std::string::npos) end = line.length();
                
                int val = 0;
                std::from_chars(line.data() + start, line.data() + end, val);
                img_data[img_offset++] = val / 255.0f;
                
                start = end + 1;
            }
        }
        return ds;
    }
};

} // namespace MetalNet
