#pragma once
// MetalNet SIMD backend - x86_64 AVX2 + FMA (256-bit, 8 lanes).
#include <immintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

namespace MetalNet {
namespace simd {

constexpr int VLEN = 8;
using vfloat = __m256;

inline vfloat vload (const float* p)            { return _mm256_loadu_ps(p); }
inline void   vstore(float* p, vfloat v)        { _mm256_storeu_ps(p, v); }
inline vfloat vset1 (float x)                   { return _mm256_set1_ps(x); }
inline vfloat vzero ()                          { return _mm256_setzero_ps(); }
inline vfloat vadd  (vfloat a, vfloat b)        { return _mm256_add_ps(a, b); }
inline vfloat vmul  (vfloat a, vfloat b)        { return _mm256_mul_ps(a, b); }
inline vfloat vfma  (vfloat a, vfloat b, vfloat c) { return _mm256_fmadd_ps(a, b, c); }

inline float vsum(vfloat v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehl_ps(lo, lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_shuffle_ps(lo, lo, 0x1);
    lo = _mm_add_ss(lo, sh);
    return _mm_cvtss_f32(lo);
}

inline void enable_ftz() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

} // namespace simd
} // namespace MetalNet
