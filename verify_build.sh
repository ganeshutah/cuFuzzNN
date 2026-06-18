#!/bin/sh
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

# cuFuzz Build Verification Script
# This script runs a short fuzzing test to verify the build is correct

set -e  # Exit on error

echo "=== cuFuzz Build Verification ==="
echo "Running a sample test (fuzz for 200 executions) to verify build is correct!"

# Configuration
GPU_ARCH=${GPU_ARCH:-sm_86}  # Default GPU architecture, can be overridden
CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}
echo "GPU Architecture: $GPU_ARCH"
echo "CUDA Path: $CUDA_PATH"

cd targets/sampleApp/

export PATH=${CUDA_PATH}/bin/:$PATH

# Clean previous outputs
rm -rf out/

echo "=== Building sample application ==="

# Build instrumented version for fuzzing
nvcc sampleApp.cu -I${CUDA_PATH}/include/ -O2 --ptxas-options "-v" \
    --gpu-architecture=$GPU_ARCH \
    --compiler-bindir ../../Tools/AFLplusplus/afl-clang-fast++ \
    -o sampleApp.out

# Build vanilla version for sanitizer
nvcc sampleApp.cu -I${CUDA_PATH}/include/ -O2 --ptxas-options "-v" \
    --gpu-architecture=$GPU_ARCH \
    -o sampleApp-vanilla.out

echo "=== Running cuFuzz for 200 executions ==="

ORIGINAL_APP=./sampleApp-vanilla.out \
SANITIZER_PATH=${CUDA_PATH}/bin/compute-sanitizer \
SANITIZER_ARG="--tool=memcheck --error-exitcode 99" \
CUFUZZ_MAP_SIZE=65536 \
AFL_SKIP_CPUFREQ=1 \
AFL_PRELOAD=../../src/cufuzz_cov_nvbit/cufuzz_cov.so \
./../../Tools/AFLplusplus/afl-fuzz -x sample.dict -i in/ -o out/ \
    -w ./../../src/cufuzz_sand/wrapper_memcheck.out \
    -t 1000000 -E 200 ./sampleApp.out @@

echo ""
echo "=== Verification Complete ==="
echo "Check 'targets/sampleApp/out/' for fuzzing results"


