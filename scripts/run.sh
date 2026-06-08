#!/usr/bin/env bash
# =====================================================================
#  MetalNet - build & run main.cpp benchmark with one command
#  Usage:  ./scripts/run.sh            (build + run)
#          ./scripts/run.sh --run-only (skip rebuild if binary exists)
#  Auto-detects: arch (x86_64 AVX2 / arm64 NEON) + OpenMP flags (mac/linux).
# =====================================================================
set -euo pipefail

# --- locate repo root (script lives in scripts/) ---------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"   # main.cpp uses include paths relative to repo root

BIN="build/metalnet"
SRC="main.cpp"
mkdir -p build

# --- pick compiler (validate it actually runs; skip broken shims) -----
CXX_ENV="${CXX:-}"
CXX=""
for cand in "$CXX_ENV" c++ clang++ g++ g++-14; do
    [ -n "$cand" ] || continue
    if "$cand" --version >/dev/null 2>&1; then CXX="$cand"; break; fi
done
if [ -z "$CXX" ]; then echo "ERROR: no working C++ compiler found."; exit 1; fi

# --- base flags ------------------------------------------------------
FLAGS=(-std=c++20 -O3 -funroll-loops -Iinclude -I.)

# --- architecture SIMD flags ----------------------------------------
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|amd64)
        FLAGS+=(-mavx2 -mfma -march=native)
        ;;
    arm64|aarch64)
        # NEON is baseline on AArch64; tune for the host core.
        FLAGS+=(-mcpu=native) 2>/dev/null || true
        ;;
    *)
        echo "WARN: unknown arch '$ARCH' - no SIMD backend, build may fail."
        ;;
esac

# --- OpenMP flags (multithreading) ----------------------------------
OS="$(uname -s)"
OMP=()
if [ "$OS" = "Darwin" ]; then
    # Apple clang needs libomp (brew). g++ from brew supports -fopenmp directly.
    if "$CXX" --version 2>/dev/null | grep -qiE 'clang|Apple'; then
        LIBOMP="$(brew --prefix libomp 2>/dev/null || true)"
        if [ -n "$LIBOMP" ] && [ -f "$LIBOMP/include/omp.h" ]; then
            OMP=(-Xpreprocessor -fopenmp -I"$LIBOMP/include" -L"$LIBOMP/lib" -lomp)
        else
            echo "ERROR: OpenMP (libomp) not installed - the engine needs it."
            echo "       Install:  brew install libomp"
            exit 1
        fi
    else
        OMP=(-fopenmp)
    fi
else
    OMP=(-fopenmp)   # Linux g++/clang
fi

# --- build -----------------------------------------------------------
if [ "${1:-}" != "--run-only" ] || [ ! -x "$BIN" ]; then
    echo ">> $CXX ${FLAGS[*]} ${OMP[*]} $SRC -o $BIN"
    "$CXX" "${FLAGS[@]}" "${OMP[@]}" "$SRC" -o "$BIN"
    echo ">> build OK"
fi

# --- run -------------------------------------------------------------
echo ">> running $BIN (arch=$ARCH)"
echo "----------------------------------------------------------------"
exec "./$BIN"
