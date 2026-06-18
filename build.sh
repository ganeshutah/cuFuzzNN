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

# cuFuzz Build Script
# This script builds all cuFuzz components: AFL++, sanitizer wrappers, and NVBit coverage tool

set -e  # Exit on error

# Configuration
GPU_ARCH=${GPU_ARCH:-sm_86}  # Default GPU architecture, can be overridden

echo "=== Building cuFuzz ==="
echo "GPU Architecture: $GPU_ARCH"

# Building the patched AFL++ fuzzer 
echo "=== Step 1: Building AFL++ ==="
cd Tools/AFLplusplus 
patch -N -p1 < ../AFLplusplus.patch || true  # Ignore if already patched
make clean 
make -j8 

# Building our cuFuzz wrapper for compute sanitizer integration
echo "=== Step 2: Building Sanitizer Wrappers ==="
cd ../../src/cufuzz_sand

# Non-persistent wrappers (for regular mode)
AFL_SAN_NO_INST=1 ./../../Tools/AFLplusplus/afl-clang-fast -O2 wrapper_san.c -o wrapper_memcheck.out 
AFL_SAN_NO_INST=1 ./../../Tools/AFLplusplus/afl-clang-fast -DSAN_MODE_INIT -O2 wrapper_san.c -o wrapper_initcheck.out 
AFL_SAN_NO_INST=1 ./../../Tools/AFLplusplus/afl-clang-fast -DSAN_MODE_RACE -O2 wrapper_san.c -o wrapper_racecheck.out
AFL_SAN_NO_INST=1 ./../../Tools/AFLplusplus/afl-clang-fast -DSAN_MODE_ASAN -O2 wrapper_san.c -o wrapper_asan.out

# Persistent wrappers (for persistent fuzzing mode)
AFL_SAN_NO_INST=1 ./../../Tools/AFLplusplus/afl-clang-fast -O2 wrapper_persistent_san.c -o wrapper_persistent_memcheck.out 
AFL_SAN_NO_INST=1 ./../../Tools/AFLplusplus/afl-clang-fast -DSAN_MODE_INIT -O2 wrapper_persistent_san.c -o wrapper_persistent_initcheck.out 
AFL_SAN_NO_INST=1 ./../../Tools/AFLplusplus/afl-clang-fast -DSAN_MODE_RACE -O2 wrapper_persistent_san.c -o wrapper_persistent_racecheck.out
AFL_SAN_NO_INST=1 ./../../Tools/AFLplusplus/afl-clang-fast -DSAN_MODE_ASAN -O2 wrapper_persistent_san.c -o wrapper_persistent_asan.out 

# Download and setup NVBit
echo "=== Step 3: Setting up NVBit ==="
cd ../../
rm -rf Tools/NVBit
mkdir -p Tools/NVBit
wget -q https://github.com/NVlabs/NVBit/releases/download/v1.7.7.3/nvbit-Linux-x86_64-1.7.7.3.tar.bz2
tar -xvf nvbit-Linux-x86_64-1.7.7.3.tar.bz2
mv nvbit_release_x86_64/* Tools/NVBit/
rm -rf nvbit_release_x86_64 nvbit-Linux-x86_64-1.7.7.3.tar.bz2

# Building our NVBit coverage tool
echo "=== Step 4: Building NVBit Coverage Tool ==="
# Use CUDA_PATH if set, otherwise try common locations
CUDA_PATH=${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}
export PATH=${CUDA_PATH}/bin/:$PATH
cd src/cufuzz_cov_nvbit/
make clean 
ARCH=$GPU_ARCH make 

echo "=== Build Complete ==="
echo "cuFuzz has been successfully built!" 


