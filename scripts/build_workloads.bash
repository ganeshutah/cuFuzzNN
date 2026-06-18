#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# build_workloads.bash - Download and build workloads for cuFuzz evaluation
#
# This script downloads the required library dependencies (nvJPEG2000, nvTIFF)
# and HeCBench benchmark suite, then builds the fuzz target binaries needed
# to reproduce the paper's evaluation.
#
# Prerequisites:
#   - CUDA 12.9 toolkit installed (provides nvcc and nvjpeg)
#   - cuFuzz built (AFL++, sanitizer wrappers, NVBit coverage tool)
#   - clang++ compiler installed
#   - wget, tar, git installed
#
# Usage:
#   ./build_workloads.bash [target]
#
# Arguments:
#   target - Optional. Build specific target:
#            "hecbench"         - Build all HeCBench workloads
#            "hecbench:<app>"   - Build specific HeCBench workload (e.g., hecbench:attention)
#            "nvjpeg"           - Build nvJPEG target
#            "nvjpeg2k"         - Build nvJPEG2000 target
#            "nvtiff"           - Build nvTIFF target
#            "all"              - Build everything (default)
#
# Available HeCBench workloads:
#   attention, blas-gemm, boxfilter, convolution3D, crs, dxtc2,
#   lud, medianfilter, recursiveGaussian, seam-carving, urng
#
# Environment Variables:
#   CUFUZZ_PATH - Path to cuFuzz installation (required)
#   EVAL_PATH   - Path where workloads will be built (required)
#   CUDA_PATH   - Path to CUDA installation (default: /usr/local/cuda)
#   GPU_ARCH    - GPU architecture number without sm_ prefix (default: 86 for A40/A100)
#
# Example:
#   export CUFUZZ_PATH=/root/cufuzz
#   export EVAL_PATH=/root/EVAL
#   export GPU_ARCH=86
#   ./build_workloads.bash all
#
# Source Files:
#   Harness files are located in $CUFUZZ_PATH/data/harnesses/
#   Seeds are located in $CUFUZZ_PATH/data/seeds/
#   Dictionaries are located in $CUFUZZ_PATH/data/dictionaries/
#
# Output (after running):
#   $EVAL_PATH/
#   ├── cudnn/              # cuDNN library (for convolution3D)
#   ├── HeCBench/           # HeCBench benchmark suite (cloned from GitHub)
#   │   ├── build_vanilla_release_static/  # Binaries for Compute Sanitizer
#   │   │   ├── attention
#   │   │   ├── attention_persistent
#   │   │   ├── blas-gemm
#   │   │   └── ... (other workloads)
#   │   ├── build_asan_release_static/     # Binaries for AddressSanitizer
#   │   └── build_afl_release_static/      # Binaries for AFL++ fuzzing
#   ├── nvjpeg/
#   │   ├── nvjpeg_harness.vanilla.<arch>
#   │   ├── nvjpeg_harness.afl.<arch>
#   │   ├── nvjpeg_harness.asan.<arch>
#   │   └── nvjpeg_harness_persistent.afl.<arch>  (persistent mode)
#   ├── nvjpeg2k/
#   │   ├── libnvjpeg_2k-linux-x86_64-*/  # Downloaded library
#   │   └── (similar binaries)
#   └── nvtiff/
#       ├── libnvtiff-linux-x86_64-*/     # Downloaded library
#       └── (similar binaries)
#

set -e

# =============================================================================
# Configuration
# =============================================================================

# Default GPU architecture (86 = sm_86 for Ampere GPUs like A40, A100, RTX 3090)
DEFAULT_ARCH=86
ARCH="${GPU_ARCH:-$DEFAULT_ARCH}"

# Paths
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
CUFUZZ_PATH="${CUFUZZ_PATH:-}"
EVAL_PATH="${EVAL_PATH:-}"

# Library versions (as used in the paper)
HECBENCH_COMMIT="9232a31691d51e52a9359d949242de81acc5fa88"
NVJPEG2K_VERSION="0.8.0.38"
NVTIFF_VERSION="0.4.0.62"
CUDNN_VERSION="9.12.0.46"

# Download URLs
NVJPEG2K_URL="https://developer.download.nvidia.com/compute/nvjpeg2000/redist/libnvjpeg_2k/linux-x86_64/libnvjpeg_2k-linux-x86_64-${NVJPEG2K_VERSION}-archive.tar.xz"
NVTIFF_URL="https://developer.download.nvidia.com/compute/nvtiff/redist/libnvtiff/linux-x86_64/libnvtiff-linux-x86_64-${NVTIFF_VERSION}_cuda12-archive.tar.xz"
CUDNN_URL="https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz"

# HeCBench workloads to build
HECBENCH_APPS="attention blas-gemm boxfilter convolution3D crs dxtc2 lud medianfilter recursiveGaussian seam-carving urng"

# Source directories (relative to CUFUZZ_PATH)
HARNESS_DIR="data/harnesses"
SEEDS_DIR="data/seeds"
DICT_DIR="data/dictionaries"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
}

print_step() {
    echo "[STEP] $1"
}

check_requirements() {
    print_header "Checking Requirements"
    
    if [ -z "$CUFUZZ_PATH" ]; then
        echo "Error: CUFUZZ_PATH environment variable is not set"
        echo "Please set it to your cuFuzz installation directory"
        exit 1
    fi
    
    if [ -z "$EVAL_PATH" ]; then
        echo "Error: EVAL_PATH environment variable is not set"
        echo "Please set it to the directory where workloads will be built"
        exit 1
    fi
    
    if [ ! -d "$CUFUZZ_PATH" ]; then
        echo "Error: CUFUZZ_PATH directory does not exist: $CUFUZZ_PATH"
        exit 1
    fi
    
    if [ ! -f "$CUFUZZ_PATH/Tools/AFLplusplus/afl-clang-fast++" ]; then
        echo "Error: AFL++ not found at $CUFUZZ_PATH/Tools/AFLplusplus/"
        echo "Please build cuFuzz first using build.sh"
        exit 1
    fi
    
    if [ ! -d "$CUDA_PATH" ]; then
        echo "Error: CUDA_PATH directory does not exist: $CUDA_PATH"
        exit 1
    fi
    
    if [ ! -d "$CUFUZZ_PATH/$HARNESS_DIR" ]; then
        echo "Error: Harness directory not found: $CUFUZZ_PATH/$HARNESS_DIR"
        exit 1
    fi
    
    echo "CUFUZZ_PATH: $CUFUZZ_PATH"
    echo "EVAL_PATH:   $EVAL_PATH"
    echo "CUDA_PATH:   $CUDA_PATH"
    echo "GPU_ARCH:    sm_${ARCH}"
    
    # Create EVAL_PATH if it doesn't exist
    mkdir -p "$EVAL_PATH"
}

# =============================================================================
# HeCBench Build
# =============================================================================

download_cudnn() {
    print_header "Downloading cuDNN (required for convolution3D)"
    
    cd "$EVAL_PATH"
    
    local CUDNN_ARCHIVE="cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive.tar.xz"
    local CUDNN_DIR="cudnn-linux-x86_64-${CUDNN_VERSION}_cuda12-archive"
    
    if [ -d "cudnn" ]; then
        print_step "cuDNN directory already exists, skipping download"
        return
    fi
    
    if [ ! -f "$CUDNN_ARCHIVE" ]; then
        print_step "Downloading cuDNN ${CUDNN_VERSION}..."
        wget -q --show-progress "$CUDNN_URL" -O "$CUDNN_ARCHIVE"
    fi
    
    print_step "Extracting cuDNN..."
    tar -xf "$CUDNN_ARCHIVE"
    mv "$CUDNN_DIR" cudnn
    
    print_step "cuDNN ready at $EVAL_PATH/cudnn"
}

# Setup HeCBench repository (clone, checkout, apply patch)
setup_hecbench() {
    local HECBENCH_DATA="$CUFUZZ_PATH/data/hecbench"
    
    cd "$EVAL_PATH"
    
    # Download cuDNN first (needed for convolution3D)
    download_cudnn
    
    # Clone or update HeCBench repository
    # Note: Skip Git LFS files - we don't need the large data files for benchmarks we're not using
    if [ -d "HeCBench" ]; then
        print_step "HeCBench directory already exists, skipping clone"
    else
        print_step "Cloning HeCBench repository (skipping LFS files)..."
        GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/zjin-lcf/HeCBench.git
    fi
    
    cd HeCBench
    
    print_step "Checking out specific commit: $HECBENCH_COMMIT"
    git checkout -f "$HECBENCH_COMMIT"
    
    # Reset any local changes to ensure clean state for patching
    print_step "Resetting repository to clean state..."
    git reset --hard HEAD
    git clean -fd src/ 2>/dev/null || true
    
    # Copy configuration files to HeCBench root (Makefiles use ../../make.config.*)
    print_step "Copying build configuration files..."
    cp "$HECBENCH_DATA/make.config.vanilla" .
    cp "$HECBENCH_DATA/make.config.asan" .
    cp "$HECBENCH_DATA/make.config.afl" .
    
    # Apply the cuFuzz patch
    print_step "Applying cuFuzz HeCBench patch..."
    if [ -f "$HECBENCH_DATA/hecbench.patch" ]; then
        if git apply "$HECBENCH_DATA/hecbench.patch"; then
            print_step "Patch applied successfully"
        else
            echo ""
            echo "Patch failed. Showing detailed error:"
            git apply --check "$HECBENCH_DATA/hecbench.patch" 2>&1 || true
            echo ""
            echo "Error: Patch failed to apply."
            echo "Try deleting $EVAL_PATH/HeCBench and re-running."
            exit 1
        fi
    else
        echo "Error: hecbench.patch not found at $HECBENCH_DATA/hecbench.patch"
        exit 1
    fi
    
    # Create output directories
    mkdir -p build_vanilla_release_static
    mkdir -p build_asan_release_static
    mkdir -p build_afl_release_static
    mkdir -p fuzz logs
    
    # Export variables for Makefiles
    export CUFUZZ_PATH
    export EVAL_PATH
    export CUDA_PATH
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export LD=/usr/bin/clang++
    export NVCC_CCBIN=/usr/bin/clang++
    export PATH="${CUDA_PATH}/bin:$PATH"
    export CUDACXX="${CUDA_PATH}/bin/nvcc"
}

# Build a single HeCBench workload
# Arguments: $1 = app name (e.g., "attention", "blas-gemm")
build_hecbench_app() {
    local app="$1"
    local HARNESS_SRC="$CUFUZZ_PATH/$HARNESS_DIR"
    local FULL_ARCH="sm_${ARCH}"
    
    # Validate app name
    if ! echo "$HECBENCH_APPS" | grep -qw "$app"; then
        echo "Error: Unknown HeCBench workload '$app'"
        echo "Available workloads: $HECBENCH_APPS"
        exit 1
    fi
    
    print_header "Building $app"
    
    # Copy persistent harness file if it exists
    local persistent_file=""
    local dest_file="main_persistent.cu"
    case "$app" in
        "lud")
            persistent_file="lud_persistent.cu"
            dest_file="lud_persistent.cu"  # lud Makefile expects lud_persistent.cu
            ;;
        *)
            persistent_file="${app}_main_persistent.cu"
            ;;
    esac
    
    if [ -f "$HARNESS_SRC/$persistent_file" ]; then
        print_step "Copying persistent harness: $persistent_file -> $dest_file"
        cp "$HARNESS_SRC/$persistent_file" "src/${app}-cuda/$dest_file"
    else
        echo "Warning: Persistent harness not found: $HARNESS_SRC/$persistent_file"
    fi
    
    # Extract compressed assets if needed (e.g., seam-carving has stb headers in image.tar.gz)
    if [ "$app" = "seam-carving" ] && [ -f "src/${app}-cuda/image.tar.gz" ]; then
        print_step "Extracting image.tar.gz for seam-carving..."
        cd "src/${app}-cuda"
        tar -xzf image.tar.gz
        cd ../..
    fi
    
    # Build vanilla (for Compute Sanitizer)
    print_step "Building $app - vanilla (for Compute Sanitizer)..."
    cd "src/${app}-cuda"
    make clean 2>/dev/null || true
    make CONFIG=vanilla main main_persistent STATIC=yes ARCH="$FULL_ARCH" || \
        make CONFIG=vanilla main STATIC=yes ARCH="$FULL_ARCH"
    cp main "../../build_vanilla_release_static/${app}" 2>/dev/null || true
    cp main_persistent "../../build_vanilla_release_static/${app}_persistent" 2>/dev/null || true
    cd ../..
    
    # Build ASAN (for AddressSanitizer)
    print_step "Building $app - asan (for AddressSanitizer)..."
    cd "src/${app}-cuda"
    make clean 2>/dev/null || true
    make CONFIG=asan main main_persistent STATIC=yes ARCH="$FULL_ARCH" || \
        make CONFIG=asan main STATIC=yes ARCH="$FULL_ARCH"
    cp main "../../build_asan_release_static/${app}" 2>/dev/null || true
    cp main_persistent "../../build_asan_release_static/${app}_persistent" 2>/dev/null || true
    cd ../..
    
    # Build AFL (for coverage-guided fuzzing)
    print_step "Building $app - afl (for AFL++ fuzzing)..."
    cd "src/${app}-cuda"
    make clean 2>/dev/null || true
    make CONFIG=afl main main_persistent STATIC=yes ARCH="$FULL_ARCH" || \
        make CONFIG=afl main STATIC=yes ARCH="$FULL_ARCH"
    cp main "../../build_afl_release_static/${app}" 2>/dev/null || true
    cp main_persistent "../../build_afl_release_static/${app}_persistent" 2>/dev/null || true
    cd ../..
    
    print_step "$app build complete!"
}

# Build all HeCBench workloads
build_hecbench() {
    print_header "Building All HeCBench Workloads"
    
    setup_hecbench
    
    # Build each workload
    for app in $HECBENCH_APPS; do
        build_hecbench_app "$app"
    done
    
    print_header "HeCBench Build Summary"
    echo "Output directories:"
    echo "  - build_vanilla_release_static/  (for Compute Sanitizer)"
    echo "  - build_asan_release_static/     (for AddressSanitizer)"
    echo "  - build_afl_release_static/      (for AFL++ fuzzing)"
    echo ""
    echo "Built workloads:"
    ls -la build_vanilla_release_static/
    echo ""
    echo "HeCBench build complete!"
}

# Build a single HeCBench workload (with setup)
build_hecbench_single() {
    local app="$1"
    print_header "Building HeCBench Workload: $app"
    
    setup_hecbench
    build_hecbench_app "$app"
    
    echo ""
    echo "Built binaries for $app:"
    ls -la "build_vanilla_release_static/${app}"* 2>/dev/null || true
    ls -la "build_asan_release_static/${app}"* 2>/dev/null || true
    ls -la "build_afl_release_static/${app}"* 2>/dev/null || true
}

# =============================================================================
# nvJPEG Build (uses CUDA's bundled libnvjpeg_static)
# =============================================================================

build_nvjpeg() {
    print_header "Building nvJPEG Fuzz Targets"
    
    local TARGET_DIR="$EVAL_PATH/nvjpeg"
    local HARNESS_SRC="$CUFUZZ_PATH/$HARNESS_DIR"
    
    mkdir -p "$TARGET_DIR/harness"
    cd "$TARGET_DIR"
    
    print_step "nvJPEG library is included with CUDA 12.9 toolkit (static)"
    echo "Using: $CUDA_PATH/lib64/libnvjpeg_static.a"
    
    # Copy harness files
    print_step "Copying harness files..."
    cp "$HARNESS_SRC/nvjpeg_harness.cpp" harness/
    cp "$HARNESS_SRC/nvjpeg_harness_persistent.cpp" harness/
    cp "$HARNESS_SRC/nvjpeg_utils.cpp" harness/
    cp "$HARNESS_SRC/nvjpeg_utils.hxx" harness/
    
    # Build vanilla executable (for Compute Sanitizer)
    print_step "Building vanilla executable..."
    NVCC_CCBIN=clang++ "$CUDA_PATH/bin/nvcc" \
        -arch=sm_${ARCH} \
        harness/nvjpeg_harness.cpp \
        harness/nvjpeg_utils.cpp \
        -std=c++11 \
        -I/usr/include/ \
        -I./harness/ \
        -L"$CUDA_PATH/lib64" \
        -lnvjpeg_static \
        -lcudart_static \
        --compiler-bindir clang++ \
        -o "nvjpeg_harness.vanilla.${ARCH}"
    
    # Build ASAN executable (for host-side error detection)
    print_step "Building ASAN executable..."
    NVCC_CCBIN=clang++ "$CUDA_PATH/bin/nvcc" \
        -Xcompiler "-fsanitize=address" \
        -arch=sm_${ARCH} \
        harness/nvjpeg_harness.cpp \
        harness/nvjpeg_utils.cpp \
        -std=c++11 \
        -I/usr/include/ \
        -I./harness/ \
        -L"$CUDA_PATH/lib64" \
        -lnvjpeg_static \
        -lcudart_static \
        --compiler-bindir clang++ \
        -o "nvjpeg_harness.asan.${ARCH}"
    
    # Build AFL executable (for coverage-guided fuzzing)
    print_step "Building AFL executable..."
    NVCC_CCBIN="$CUFUZZ_PATH/Tools/AFLplusplus/afl-clang-fast++" "$CUDA_PATH/bin/nvcc" \
        -arch=sm_${ARCH} \
        harness/nvjpeg_harness.cpp \
        harness/nvjpeg_utils.cpp \
        -std=c++11 \
        -I/usr/include/ \
        -I./harness/ \
        -L"$CUDA_PATH/lib64" \
        -lnvjpeg_static \
        -lcudart_static \
        --compiler-bindir "$CUFUZZ_PATH/Tools/AFLplusplus/afl-clang-fast++" \
        -o "nvjpeg_harness.afl.${ARCH}"
    
    # Build persistent AFL executable
    print_step "Building persistent AFL executable..."
    NVCC_CCBIN="$CUFUZZ_PATH/Tools/AFLplusplus/afl-clang-fast++" "$CUDA_PATH/bin/nvcc" \
        -x cu \
        -arch=sm_${ARCH} \
        harness/nvjpeg_harness_persistent.cpp \
        harness/nvjpeg_utils.cpp \
        -std=c++11 \
        -I/usr/include/ \
        -I./harness/ \
        -L"$CUDA_PATH/lib64" \
        -lnvjpeg_static \
        -lcudart_static \
        --compiler-bindir "$CUFUZZ_PATH/Tools/AFLplusplus/afl-clang-fast++" \
        -o "nvjpeg_harness_persistent.afl.${ARCH}"
    
    # Create required directories
    mkdir -p fuzz logs
    
    print_step "nvJPEG build complete!"
    ls -la nvjpeg_harness*.${ARCH}
}

# =============================================================================
# nvJPEG2000 Build
# =============================================================================

build_nvjpeg2k() {
    print_header "Building nvJPEG2000 Fuzz Targets"
    
    local TARGET_DIR="$EVAL_PATH/nvjpeg2k"
    local LIB_ARCHIVE="libnvjpeg_2k-linux-x86_64-${NVJPEG2K_VERSION}-archive"
    local HARNESS_SRC="$CUFUZZ_PATH/$HARNESS_DIR"
    
    mkdir -p "$TARGET_DIR/harness"
    cd "$TARGET_DIR"
    
    # Download library if not present
    if [ ! -d "$LIB_ARCHIVE" ]; then
        print_step "Downloading nvJPEG2000 library v${NVJPEG2K_VERSION}..."
        wget -q --show-progress "$NVJPEG2K_URL" -O "${LIB_ARCHIVE}.tar.xz"
        
        print_step "Extracting library..."
        tar -xf "${LIB_ARCHIVE}.tar.xz"
        rm "${LIB_ARCHIVE}.tar.xz"
    else
        print_step "nvJPEG2000 library already downloaded"
    fi
    
    # Copy harness files
    print_step "Copying harness files..."
    cp "$HARNESS_SRC/nvjpeg2k_harness.cpp" harness/
    cp "$HARNESS_SRC/nvjpeg2k_harness_persistent.cpp" harness/
    cp "$HARNESS_SRC/nvjpeg2k_utils.cpp" harness/
    cp "$HARNESS_SRC/nvjpeg2k_utils.hxx" harness/
    
    # Build vanilla executable (for Compute Sanitizer)
    print_step "Building vanilla executable..."
    NVCC_CCBIN=clang++ "$CUDA_PATH/bin/nvcc" \
        -arch=sm_${ARCH} \
        harness/nvjpeg2k_harness.cpp \
        harness/nvjpeg2k_utils.cpp \
        -std=c++11 \
        -I/usr/include/ \
        -I./harness/ \
        -I"./${LIB_ARCHIVE}/include/" \
        -L"./${LIB_ARCHIVE}/lib/12" \
        -L"$CUDA_PATH/lib64" \
        -lnvjpeg2k_static \
        -lcudart_static \
        --compiler-bindir clang++ \
        -o "nvjpeg2k_harness.vanilla.${ARCH}"
    
    # Build ASAN executable (for host-side error detection)
    print_step "Building ASAN executable..."
    NVCC_CCBIN=clang++ "$CUDA_PATH/bin/nvcc" \
        -Xcompiler "-fsanitize=address" \
        -arch=sm_${ARCH} \
        harness/nvjpeg2k_harness.cpp \
        harness/nvjpeg2k_utils.cpp \
        -std=c++11 \
        -I/usr/include/ \
        -I./harness/ \
        -I"./${LIB_ARCHIVE}/include/" \
        -L"./${LIB_ARCHIVE}/lib/12" \
        -L"$CUDA_PATH/lib64" \
        -lnvjpeg2k_static \
        -lcudart_static \
        --compiler-bindir clang++ \
        -o "nvjpeg2k_harness.asan.${ARCH}"
    
    # Build AFL executable (for coverage-guided fuzzing)
    print_step "Building AFL executable..."
    NVCC_CCBIN="$CUFUZZ_PATH/Tools/AFLplusplus/afl-clang-fast++" "$CUDA_PATH/bin/nvcc" \
        -arch=sm_${ARCH} \
        harness/nvjpeg2k_harness.cpp \
        harness/nvjpeg2k_utils.cpp \
        -std=c++11 \
        -I/usr/include/ \
        -I./harness/ \
        -I"./${LIB_ARCHIVE}/include/" \
        -L"./${LIB_ARCHIVE}/lib/12" \
        -L"$CUDA_PATH/lib64" \
        -lnvjpeg2k_static \
        -lcudart_static \
        --compiler-bindir "$CUFUZZ_PATH/Tools/AFLplusplus/afl-clang-fast++" \
        -o "nvjpeg2k_harness.afl.${ARCH}"
    
    # Build persistent AFL executable
    print_step "Building persistent AFL executable..."
    NVCC_CCBIN="$CUFUZZ_PATH/Tools/AFLplusplus/afl-clang-fast++" "$CUDA_PATH/bin/nvcc" \
        -x cu \
        -arch=sm_${ARCH} \
        harness/nvjpeg2k_harness_persistent.cpp \
        harness/nvjpeg2k_utils.cpp \
        -std=c++11 \
        -I/usr/include/ \
        -I./harness/ \
        -I"./${LIB_ARCHIVE}/include/" \
        -L"./${LIB_ARCHIVE}/lib/12" \
        -L"$CUDA_PATH/lib64" \
        -lnvjpeg2k_static \
        -lcudart_static \
        --compiler-bindir "$CUFUZZ_PATH/Tools/AFLplusplus/afl-clang-fast++" \
        -o "nvjpeg2k_harness_persistent.afl.${ARCH}"
    
    # Create required directories
    mkdir -p fuzz logs
    
    print_step "nvJPEG2000 build complete!"
    ls -la nvjpeg2k_harness*.${ARCH}
}

# =============================================================================
# nvTIFF Build
# =============================================================================

build_nvtiff() {
    print_header "Building nvTIFF Fuzz Targets"
    
    local TARGET_DIR="$EVAL_PATH/nvtiff"
    local LIB_ARCHIVE="libnvtiff-linux-x86_64-${NVTIFF_VERSION}_cuda12-archive"
    local HARNESS_SRC="$CUFUZZ_PATH/$HARNESS_DIR"
    
    mkdir -p "$TARGET_DIR/harness"
    cd "$TARGET_DIR"
    
    # Download library if not present
    if [ ! -d "$LIB_ARCHIVE" ]; then
        print_step "Downloading nvTIFF library v${NVTIFF_VERSION}..."
        wget -q --show-progress "$NVTIFF_URL" -O "${LIB_ARCHIVE}.tar.xz"
        
        print_step "Extracting library..."
        tar -xf "${LIB_ARCHIVE}.tar.xz"
        rm "${LIB_ARCHIVE}.tar.xz"
    else
        print_step "nvTIFF library already downloaded"
    fi
    
    # Copy harness files
    print_step "Copying harness files..."
    cp "$HARNESS_SRC/nvtiff_harness.cu" harness/
    cp "$HARNESS_SRC/nvtiff_harness_persistent.cu" harness/
    cp "$HARNESS_SRC/nvTiff_utils.cpp" harness/
    cp "$HARNESS_SRC/nvTiff_utils.h" harness/
    
    # Build vanilla executable (for Compute Sanitizer)
    print_step "Building vanilla executable..."
    NVCC_CCBIN=clang++ "$CUDA_PATH/bin/nvcc" \
        -arch=sm_${ARCH} \
        harness/nvtiff_harness.cu \
        harness/nvTiff_utils.cpp \
        -std=c++17 \
        -I/usr/include/ \
        -I./harness/ \
        -I"./${LIB_ARCHIVE}/include/" \
        -L"./${LIB_ARCHIVE}/lib" \
        -L"$CUDA_PATH/lib64" \
        -lnvtiff_static \
        -lcudart_static \
        --compiler-bindir clang++ \
        -o "nvtiff_harness.vanilla.${ARCH}"
    
    # Build ASAN executable (for host-side error detection)
    print_step "Building ASAN executable..."
    NVCC_CCBIN=clang++ "$CUDA_PATH/bin/nvcc" \
        -Xcompiler "-fsanitize=address" \
        -arch=sm_${ARCH} \
        harness/nvtiff_harness.cu \
        harness/nvTiff_utils.cpp \
        -std=c++17 \
        -I/usr/include/ \
        -I./harness/ \
        -I"./${LIB_ARCHIVE}/include/" \
        -L"./${LIB_ARCHIVE}/lib" \
        -L"$CUDA_PATH/lib64" \
        -lnvtiff_static \
        -lcudart_static \
        --compiler-bindir clang++ \
        -o "nvtiff_harness.asan.${ARCH}"
    
    # Build AFL executable (for coverage-guided fuzzing)
    print_step "Building AFL executable..."
    NVCC_CCBIN="$CUFUZZ_PATH/Tools/AFLplusplus/afl-clang-fast++" "$CUDA_PATH/bin/nvcc" \
        -arch=sm_${ARCH} \
        harness/nvtiff_harness.cu \
        harness/nvTiff_utils.cpp \
        -std=c++17 \
        -I/usr/include/ \
        -I./harness/ \
        -I"./${LIB_ARCHIVE}/include/" \
        -L"./${LIB_ARCHIVE}/lib" \
        -L"$CUDA_PATH/lib64" \
        -lnvtiff_static \
        -lcudart_static \
        --compiler-bindir "$CUFUZZ_PATH/Tools/AFLplusplus/afl-clang-fast++" \
        -o "nvtiff_harness.afl.${ARCH}"
    
    # Build persistent AFL executable
    print_step "Building persistent AFL executable..."
    NVCC_CCBIN="$CUFUZZ_PATH/Tools/AFLplusplus/afl-clang-fast++" "$CUDA_PATH/bin/nvcc" \
        -arch=sm_${ARCH} \
        harness/nvtiff_harness_persistent.cu \
        harness/nvTiff_utils.cpp \
        -std=c++17 \
        -I/usr/include/ \
        -I./harness/ \
        -I"./${LIB_ARCHIVE}/include/" \
        -L"./${LIB_ARCHIVE}/lib" \
        -L"$CUDA_PATH/lib64" \
        -lnvtiff_static \
        -lcudart_static \
        --compiler-bindir "$CUFUZZ_PATH/Tools/AFLplusplus/afl-clang-fast++" \
        -o "nvtiff_harness_persistent.afl.${ARCH}"
    
    # Create required directories
    mkdir -p fuzz logs
    
    print_step "nvTIFF build complete!"
    ls -la nvtiff_harness*.${ARCH}
}

# =============================================================================
# Main
# =============================================================================

main() {
    local TARGET="${1:-all}"
    
    print_header "cuFuzz Workload Build Script"
    echo "Target: $TARGET"
    
    check_requirements
    
    case "$TARGET" in
        hecbench)
            build_hecbench
            ;;
        hecbench:*)
            # Extract app name after "hecbench:"
            local APP_NAME="${TARGET#hecbench:}"
            build_hecbench_single "$APP_NAME"
            ;;
        nvjpeg)
            build_nvjpeg
            ;;
        nvjpeg2k)
            build_nvjpeg2k
            ;;
        nvtiff)
            build_nvtiff
            ;;
        all)
            build_hecbench
            build_nvjpeg
            build_nvjpeg2k
            build_nvtiff
            ;;
        *)
            echo "Error: Unknown target '$TARGET'"
            echo "Valid targets: hecbench, hecbench:<app>, nvjpeg, nvjpeg2k, nvtiff, all"
            echo ""
            echo "Available HeCBench workloads: $HECBENCH_APPS"
            exit 1
            ;;
    esac
    
    print_header "Build Complete!"
    echo ""
    echo "Workloads are ready at: $EVAL_PATH"
    echo ""
    echo "To run fuzzing experiments, use the scripts in scripts/ directory:"
    echo "  - scripts/run_nvjpeg.bash"
    echo "  - scripts/run_nvjpeg2k.bash"
    echo "  - scripts/run_nvtiff.bash"
    echo "  - scripts/run_hecbench.bash"
}

main "$@"
