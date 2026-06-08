cmake_minimum_required(VERSION 3.14)
project(MetalNet)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Architecture detection -----------------------------------------
# Compute kernels are written once against MetalNet::simd and compiled
# with a native backend: AVX2 on x86_64, NEON on arm64. Any other arch
# has no backend -> engine targets are skipped (profiler core still builds).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686|i386")
    set(MN_X86 TRUE)
    set(MN_SIMD TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(MN_ARM TRUE)
    set(MN_SIMD TRUE)
else()
    set(MN_SIMD FALSE)
    message(STATUS "MetalNet: no SIMD backend for ${CMAKE_SYSTEM_PROCESSOR} - engine targets disabled, profiler core still builds.")
endif()

if(MN_X86)
    if(MSVC)
        add_compile_options(/arch:AVX2)
    else()
        add_compile_options(-mavx2 -mfma)
    endif()
elseif(MN_ARM)
    # NEON is baseline on AArch64; no extra flag required.
endif()

include_directories(${CMAKE_SOURCE_DIR}/include/MetalNet)
include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(OpenMP)

# Apple clang doesn't ship libomp; wire Homebrew's manually so OpenMP::OpenMP_CXX
# works for every target (find_package can't locate it on macOS).
if(NOT OpenMP_CXX_FOUND AND APPLE)
    execute_process(COMMAND brew --prefix libomp
                    OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(LIBOMP_PREFIX AND EXISTS "${LIBOMP_PREFIX}/include/omp.h")
        message(STATUS "MetalNet: using Homebrew libomp at ${LIBOMP_PREFIX}")
        add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
        target_compile_options(OpenMP::OpenMP_CXX INTERFACE -Xclang -fopenmp)
        target_include_directories(OpenMP::OpenMP_CXX INTERFACE "${LIBOMP_PREFIX}/include")
        target_link_libraries(OpenMP::OpenMP_CXX INTERFACE "${LIBOMP_PREFIX}/lib/libomp.dylib")
        set(OpenMP_CXX_FOUND TRUE)
    else()
        message(WARNING "MetalNet: libomp not found. Install with: brew install libomp")
    endif()
endif()

# --- Core engine benchmark (x86_64 AVX2 / arm64 NEON) ---------------
if(MN_SIMD)
    add_executable(MetalNet main.cpp)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(MetalNet PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(EXISTS ${CMAKE_SOURCE_DIR}/tests.cpp)
        add_executable(MetalNetTests tests.cpp)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(MetalNetTests PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()
endif()

# --- Profiler targets ------------------------------------------------
add_subdirectory(profiler)
