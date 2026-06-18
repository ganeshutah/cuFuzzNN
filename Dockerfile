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

# Use NVIDIA CUDA 12.9 development image with Ubuntu 22.04
# This provides CUDA toolkit, nvcc, and all development headers pre-installed
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# Set the maintainer label
LABEL maintainer="cufuzz-authors"
LABEL description="Docker image for the cuFuzz toolchain"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root
COPY . ./cufuzz/

# Update the package list and install AFL++ and NVBit dependencies
RUN apt-get update && apt-get install -y build-essential python3-dev automake cmake git flex \
    bison libglib2.0-dev libpixman-1-dev python3-setuptools cargo libgtk-3-dev lld llvm llvm-dev \
    clang ninja-build cpio libcapstone-dev wget curl python3-pip vim less libxxhash-dev bc zlib1g-dev git git-lfs libomp-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev hyperfine
RUN apt-get update && apt-get install -y \
    gcc-$(gcc --version|head -n1|sed 's/\..*//'|sed 's/.* //')-plugin-dev \
    libstdc++-$(gcc --version|head -n1|sed 's/\..*//'|sed 's/.* //')-dev

# Set environment variables for compilers
# CUDA is already in PATH from the base image at /usr/local/cuda
ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++

# GPU architecture (default: sm_86 for Ampere GPUs like A40, A100, RTX 3090)
# Override with: docker build --build-arg GPU_ARCH=sm_90 ...
ARG GPU_ARCH=sm_86
ENV GPU_ARCH=${GPU_ARCH}

# Change to the cufuzz directory
WORKDIR ./cufuzz/

RUN chmod +x build.sh verify_build.sh

RUN ./build.sh
