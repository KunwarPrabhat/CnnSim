#!/usr/bin/env bash
# =====================================================================
#  MetalNet Profiler GUI - configure, build & launch with one command
#  Usage:  ./scripts/run_gui.sh            (build if needed, then run)
#          ./scripts/run_gui.sh --rebuild  (force reconfigure + rebuild)
#  First run clones imgui+implot via CMake FetchContent (needs network).
# =====================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

BUILD_DIR="build"
BIN="$BUILD_DIR/profiler/metalnet_profiler_gui"

# --- macOS prerequisite check (glfw + libomp via Homebrew) -----------
if [ "$(uname -s)" = "Darwin" ]; then
    MISSING=""
    command -v brew >/dev/null 2>&1 || { echo "ERROR: Homebrew required. https://brew.sh"; exit 1; }
    brew --prefix glfw   >/dev/null 2>&1 && [ -d "$(brew --prefix glfw)" ]   || MISSING="$MISSING glfw"
    [ -f "$(brew --prefix libomp 2>/dev/null)/include/omp.h" ]               || MISSING="$MISSING libomp"
    if [ -n "$MISSING" ]; then
        echo "ERROR: missing dependencies:$MISSING"
        echo "       Install:  brew install$MISSING"
        exit 1
    fi
fi

# --- configure (FetchContent clones imgui+implot on first run) -------
if [ "${1:-}" = "--rebuild" ]; then rm -rf "$BUILD_DIR"; fi
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    echo ">> configuring (first run clones imgui+implot, ~25s)..."
    cmake -S . -B "$BUILD_DIR" -DMETALNET_BUILD_GUI=ON
fi

# --- build -----------------------------------------------------------
echo ">> building GUI..."
cmake --build "$BUILD_DIR" --target metalnet_profiler_gui

# --- run -------------------------------------------------------------
echo ">> launching $BIN"
exec "./$BIN"
