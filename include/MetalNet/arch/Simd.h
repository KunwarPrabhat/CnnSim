#pragma once
// =====================================================================
//  MetalNet SIMD abstraction
// ---------------------------------------------------------------------
//  Selects an architecture backend at compile time. All kernels in
//  Layers/ are written ONCE against the MetalNet::simd contract below
//  and compiled with native intrinsics on each target:
//
//      vfloat                 - native vector register
//      VLEN                   - lanes per register (8 AVX2 / 4 NEON)
//      vload / vstore         - unaligned load / store
//      vset1                  - broadcast scalar
//      vzero                  - zeroed register
//      vadd / vmul            - lane-wise add / multiply
//      vfma(a,b,c)            - fused multiply-add: a*b + c
//      vsum                   - horizontal reduce to scalar
//      enable_ftz             - flush denormals to zero (MXCSR / FPCR)
//
//  Universal kernels block by 4*VLEN -> VLEN -> scalar tail, so the same
//  source adapts to either register width automatically.
// =====================================================================
#if defined(__AVX2__)
  #include "x86_64/Simd.h"
#elif defined(__aarch64__) || defined(__ARM_NEON)
  #include "arm64/Simd.h"
#else
  #error "MetalNet: no SIMD backend for this architecture. Build on x86_64 with -mavx2 -mfma, or on arm64 (NEON)."
#endif
