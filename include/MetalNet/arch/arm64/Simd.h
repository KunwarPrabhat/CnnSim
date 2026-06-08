#pragma once
// MetalNet SIMD backend - AArch64 NEON (128-bit, 4 lanes).
#include <arm_neon.h>
#include <cstdint>

namespace MetalNet {
namespace simd {

constexpr int VLEN = 4;
using vfloat = float32x4_t;

inline vfloat vload (const float* p)            { return vld1q_f32(p); }
inline void   vstore(float* p, vfloat v)        { vst1q_f32(p, v); }
inline vfloat vset1 (float x)                   { return vdupq_n_f32(x); }
inline vfloat vzero ()                          { return vdupq_n_f32(0.0f); }
inline vfloat vadd  (vfloat a, vfloat b)        { return vaddq_f32(a, b); }
inline vfloat vmul  (vfloat a, vfloat b)        { return vmulq_f32(a, b); }
// NEON fma argument order is (acc, a, b) -> acc + a*b; expose as a*b + c.
inline vfloat vfma  (vfloat a, vfloat b, vfloat c) { return vfmaq_f32(c, a, b); }

inline float vsum(vfloat v) { return vaddvq_f32(v); }   // AArch64 horizontal add

inline void enable_ftz() {
    uint64_t fpcr;
    __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
    fpcr |= (1ULL << 24);   // FZ: flush-to-zero
    __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));
}

} // namespace simd
} // namespace MetalNet
