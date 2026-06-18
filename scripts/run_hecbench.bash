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
# run_hecbench.bash - Run cuFuzz experiments on HeCBench applications
#
# This script launches fuzzing campaigns for HeCBench applications using
# different configurations evaluated in the cuFuzz paper. Each configuration
# corresponds to a specific mode (AFL++, cuFuzz variants, persistent mode, etc.)
#
# Usage:
#   ./run_hecbench.bash <MACHINE_ID> <APP_NAME> <RUN_ID> <TIME_SECONDS> <CONFIG>
#
# Arguments:
#   MACHINE_ID   - Identifier for the machine (used in log filenames)
#   APP_NAME     - Name of the HeCBench application to fuzz
#   RUN_ID       - Run identifier for distinguishing multiple runs
#   TIME_SECONDS - Fuzzing duration in seconds (e.g., 86400 for 24 hours)
#   CONFIG       - Configuration name (see below)
#
# Configurations (as referenced in the paper):
#   nosanafl           - AFL++ baseline (no device coverage, no sanitizers)
#   nosanhd            - cuFuzz-noSanitizer (device coverage, no sanitizers)
#   sansimtrace4hd     - cuFuzz default (simplify-trace sanitization strategy)
#   sansimtrace4afl    - cuFuzz-noDeviceCoverage (host coverage only)
#   perssansimtrace4hd - cuFuzz persistent mode
#   sanall4hd          - cuFuzz all-trace strategy
#   sanuniqtrace4hd    - cuFuzz unique-trace strategy
#   sancovinc4hd       - cuFuzz coverage-increase strategy
#
# Environment:
#   CUFUZZ_PATH   - Path to cuFuzz installation (required)
#   HECBENCH_PATH - Path to HeCBench directory (required)
#
# Example:
#   export CUFUZZ_PATH=/path/to/cufuzz
#   export HECBENCH_PATH=/path/to/HeCBench
#   ./run_hecbench.bash node1 bfs 1 86400 sansimtrace4hd
#

set -e

export MACHINE=$1
export APP=$2
export RUN=$3
export TIME=$4 
export CONFIG=$5

# Validate required environment variables
if [ -z "$CUFUZZ_PATH" ]; then
    echo "Error: CUFUZZ_PATH environment variable is not set"
    exit 1
fi

if [ -z "$HECBENCH_PATH" ]; then
    echo "Error: HECBENCH_PATH environment variable is not set"
    exit 1
fi

export LD_LIBRARY_PATH=${CUDA_PATH:-/usr/local/cuda}/lib64/:$LD_LIBRARY_PATH 
cd "$HECBENCH_PATH"

export DSTN=exp_hecbench

# Create output directories
mkdir -p fuzz/${DSTN}/${APP}
mkdir -p logs

$CUFUZZ_PATH/scripts/start_monitor.sh logs/my_custom_log-${MACHINE}.log 10 300 

# nosanhd is the "cufuzz-noSanitizer" mode
if [ "$CONFIG" = "nosanhd" ]; then
    AFL_NO_AFFINITY=1 CUFUZZ_MAP_SIZE=65536 AFL_MAP_SIZE=65536 AFL_SKIP_CPUFREQ=1 EDGE_COV=1 COUNT_WARP_LEVEL=1 COV_CTRL=3 AFL_PRELOAD=$CUFUZZ_PATH/src/cufuzz_cov.so ${CUFUZZ_PATH}/Tools/AFLplusplus/afl-fuzz -i $CUFUZZ_PATH/data/seeds/${APP}/ -o fuzz/${DSTN}/${APP}/out_seeds_nosan_hd_${RUN} -V ${TIME} -t 120000   -Z ./build_afl_release_static/${APP} @@
fi

# nosanafl is the "AFL++" mode
if [ "$CONFIG" = "nosanafl" ]; then
    AFL_NO_AFFINITY=1 CUFUZZ_MAP_SIZE=65536 AFL_MAP_SIZE=65536 AFL_SKIP_CPUFREQ=1 ${CUFUZZ_PATH}/Tools/AFLplusplus/afl-fuzz -i $CUFUZZ_PATH/data/seeds/${APP}/ -o fuzz/${DSTN}/${APP}/out_seeds_nosan_afl_${RUN} -t 120000 -V ${TIME} -Z ./build_afl_release_static/${APP} @@ 
fi

# sancovinc4hd is the "cufuzz-coverageIncrease" mode
if [ "$CONFIG" = "sancovinc4hd" ]; then
    AFL_NO_AFFINITY=1 AFL_SAN_ABSTRACTION=coverage_increase ASAN_APP="./build_asan_release_static/${APP}" ORIGINAL_APP="./build_vanilla_release_static/${APP}" SANITIZER_PATH=${CUDA_PATH:-/usr/local/cuda}/bin/compute-sanitizer SANITIZER_ARG="--tool=memcheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_RACE="--tool=racecheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_INIT="--tool=initcheck --report-api-errors=no --error-exitcode 99" CUFUZZ_MAP_SIZE=65536 AFL_MAP_SIZE=65536 AFL_SKIP_CPUFREQ=1 EDGE_COV=1 COUNT_WARP_LEVEL=1 COV_CTRL=3 AFL_PRELOAD=$CUFUZZ_PATH/src/cufuzz_cov.so ${CUFUZZ_PATH}/Tools/AFLplusplus/afl-fuzz -i $CUFUZZ_PATH/data/seeds/${APP}/ -o fuzz/${DSTN}/${APP}/out_seeds_sancovinc_MRIA_hd_${RUN} -t 120000 -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_memcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_racecheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_initcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_asan.out -V ${TIME} -Z ./build_afl_release_static/${APP} @@
fi

# sanuniqtrace4hd is the "cufuzz-uniqueTrace" mode
if [ "$CONFIG" = "sanuniqtrace4hd" ]; then
    AFL_NO_AFFINITY=1 AFL_SAN_ABSTRACTION=unique_trace ASAN_APP="./build_asan_release_static/${APP}" ORIGINAL_APP="./build_vanilla_release_static/${APP}" SANITIZER_PATH=${CUDA_PATH:-/usr/local/cuda}/bin/compute-sanitizer SANITIZER_ARG="--tool=memcheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_RACE="--tool=racecheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_INIT="--tool=initcheck --report-api-errors=no --error-exitcode 99" CUFUZZ_MAP_SIZE=65536 AFL_MAP_SIZE=65536 AFL_SKIP_CPUFREQ=1 EDGE_COV=1 COUNT_WARP_LEVEL=1 COV_CTRL=3 AFL_PRELOAD=$CUFUZZ_PATH/src/cufuzz_cov.so ${CUFUZZ_PATH}/Tools/AFLplusplus/afl-fuzz -i $CUFUZZ_PATH/data/seeds/${APP}/ -o fuzz/${DSTN}/${APP}/out_seeds_sanuniqtrace_MRIA_hd_${RUN} -t 120000 -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_memcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_racecheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_initcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_asan.out -V ${TIME} -Z ./build_afl_release_static/${APP} @@
fi

# sansimtrace4hd is the "cufuzz" mode which is also the default mode in our experiments. It is the "simplify trace" mode
if [ "$CONFIG" = "sansimtrace4hd" ]; then
    AFL_NO_AFFINITY=1 AFL_SAN_ABSTRACTION=simplify_trace ASAN_APP="./build_asan_release_static/${APP}" ORIGINAL_APP="./build_vanilla_release_static/${APP}" SANITIZER_PATH=${CUDA_PATH:-/usr/local/cuda}/bin/compute-sanitizer SANITIZER_ARG="--tool=memcheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_RACE="--tool=racecheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_INIT="--tool=initcheck --report-api-errors=no --error-exitcode 99" CUFUZZ_MAP_SIZE=65536 AFL_MAP_SIZE=65536 AFL_SKIP_CPUFREQ=1 EDGE_COV=1 COUNT_WARP_LEVEL=1 COV_CTRL=3 AFL_PRELOAD=$CUFUZZ_PATH/src/cufuzz_cov.so ${CUFUZZ_PATH}/Tools/AFLplusplus/afl-fuzz -i $CUFUZZ_PATH/data/seeds/${APP}/ -o fuzz/${DSTN}/${APP}/out_seeds_sansimtrace_MRIA_hd_${RUN} -t 120000 -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_memcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_racecheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_initcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_asan.out -V ${TIME} -Z ./build_afl_release_static/${APP} @@
fi

# sansimtrace4afl is the "cufuzz-noDeviceCoverage" mode
if [ "$CONFIG" = "sansimtrace4afl" ]; then
    AFL_NO_AFFINITY=1 AFL_SAN_ABSTRACTION=simplify_trace ASAN_APP="./build_asan_release_static/${APP}" ORIGINAL_APP="./build_vanilla_release_static/${APP}" SANITIZER_PATH=${CUDA_PATH:-/usr/local/cuda}/bin/compute-sanitizer SANITIZER_ARG="--tool=memcheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_RACE="--tool=racecheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_INIT="--tool=initcheck --report-api-errors=no --error-exitcode 99" CUFUZZ_MAP_SIZE=65536 AFL_MAP_SIZE=65536 AFL_SKIP_CPUFREQ=1 ${CUFUZZ_PATH}/Tools/AFLplusplus/afl-fuzz -i $CUFUZZ_PATH/data/seeds/${APP}/ -o fuzz/${DSTN}/${APP}/out_seeds_sansimtrace_MRIA_afl_${RUN} -t 120000 -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_memcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_racecheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_initcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_asan.out -V ${TIME} -Z ./build_afl_release_static/${APP} @@ 
fi

# perssansimtrace4hd is the "persistent" mode
if [ "$CONFIG" = "perssansimtrace4hd" ]; then
    AFL_NO_AFFINITY=1 COV_PERSISTENT=1 AFL_SAN_ABSTRACTION=simplify_trace ASAN_APP="./build_asan_release_static/${APP}" ORIGINAL_APP="./build_vanilla_release_static/${APP}" SANITIZER_PATH=${CUDA_PATH:-/usr/local/cuda}/bin/compute-sanitizer SANITIZER_ARG="--tool=memcheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_RACE="--tool=racecheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_INIT="--tool=initcheck --report-api-errors=no --error-exitcode 99" CUFUZZ_MAP_SIZE=65536 AFL_MAP_SIZE=65536 AFL_SKIP_CPUFREQ=1 EDGE_COV=1 COUNT_WARP_LEVEL=1 COV_CTRL=3 AFL_PRELOAD=$CUFUZZ_PATH/src/cufuzz_cov.so ${CUFUZZ_PATH}/Tools/AFLplusplus/afl-fuzz -i $CUFUZZ_PATH/data/seeds/${APP}/ -o fuzz/${DSTN}/${APP}/out_seeds_pers_sansimtrace_MRIA_hd_${RUN} -t 120000 -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_persistent_memcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_persistent_racecheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_persistent_initcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_persistent_asan_A40.out  -V ${TIME} -Z ./build_afl_release_static/${APP}_persistent @@ 
fi

# sanall4hd is the "cufuzz-allTrace" mode
if [ "$CONFIG" = "sanall4hd" ]; then
    AFL_NO_AFFINITY=1 AFL_SAN_ABSTRACTION=all_trace ASAN_APP="./build_asan_release_static/${APP}" ORIGINAL_APP="./build_vanilla_release_static/${APP}" SANITIZER_PATH=${CUDA_PATH:-/usr/local/cuda}/bin/compute-sanitizer SANITIZER_ARG="--tool=memcheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_RACE="--tool=racecheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_INIT="--tool=initcheck --report-api-errors=no --error-exitcode 99" CUFUZZ_MAP_SIZE=65536 AFL_MAP_SIZE=65536 AFL_SKIP_CPUFREQ=1 EDGE_COV=1 COUNT_WARP_LEVEL=1 COV_CTRL=3 AFL_PRELOAD=$CUFUZZ_PATH/src/cufuzz_cov.so ${CUFUZZ_PATH}/Tools/AFLplusplus/afl-fuzz -i $CUFUZZ_PATH/data/seeds/${APP}/ -o fuzz/${DSTN}/${APP}/out_seeds_sanall_MRIA_hd_${RUN} -t 120000 -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_memcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_racecheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_initcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_asan.out -V ${TIME} -Z ./build_afl_release_static/${APP} @@
fi
