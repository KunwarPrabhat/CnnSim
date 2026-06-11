#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Change directory to the root of the project (one level up from the scripts directory)
cd "$(dirname "$0")/.."

echo "Compiling MetalNet with maximum optimization flags..."
g++ -std=c++20 -O3 -mavx2 -mfma -fopenmp -march=native -ffast-math -I./include main.cpp -o benchmark

echo "Compilation successful! Running benchmark..."
./benchmark
