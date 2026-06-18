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
# run_nvjpeg.bash - Run cuFuzz experiments on nvJPEG library
#
# This script launches fuzzing campaigns for the NVIDIA nvJPEG library using
# different configurations evaluated in the cuFuzz paper. Each configuration
# corresponds to a specific mode (AFL++, cuFuzz variants, persistent mode, etc.)
#
# Usage:
#   ./run_nvjpeg.bash <MACHINE_ID> <APP_NAME> <RUN_ID> <TIME_SECONDS> <CONFIG>
#
# Arguments:
#   MACHINE_ID   - Identifier for the machine (used in log filenames)
#   APP_NAME     - Application name (typically "nvjpeg")
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
#   CUFUZZ_PATH - Path to cuFuzz installation (required)
#   EVAL_PATH   - Path to evaluation targets directory (required)
#
# Example:
#   export CUFUZZ_PATH=/path/to/cufuzz
#   export EVAL_PATH=/path/to/EVAL
#   ./run_nvjpeg.bash node1 nvjpeg 1 86400 sansimtrace4hd
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

if [ -z "$EVAL_PATH" ]; then
    echo "Error: EVAL_PATH environment variable is not set"
    exit 1
fi

cd "$EVAL_PATH/${APP}"

export DSTN=exp_nvjpeg

# Create output directories
mkdir -p fuzz/${DSTN}
mkdir -p logs

$CUFUZZ_PATH/scripts/start_monitor.sh logs/my_custom_log-${MACHINE}.log 10 300 

# nosanhd is the "cufuzz-noSanitizer" mode
if [ "$CONFIG" = "nosanhd" ]; then
    AFL_NO_AFFINITY=1 CUFUZZ_MAP_SIZE=65536 AFL_MAP_SIZE=65536 AFL_SKIP_CPUFREQ=1 EDGE_COV=1 COUNT_WARP_LEVEL=1 COV_CTRL=3 AFL_PRELOAD=$CUFUZZ_PATH/src/cufuzz_cov.so ${CUFUZZ_PATH}/Tools/AFLplusplus/afl-fuzz -i $CUFUZZ_PATH/data/seeds/nvjpeg/ -o fuzz/${DSTN}/out_seeds_nosan_hd_${RUN} -V ${TIME} -x $CUFUZZ_PATH/data/dictionaries/jpeg.dict -t 120000   -Z ./nvjpeg_harness.afl.86 @@ rgb 
fi

# nosanafl is the "AFL++" mode
if [ "$CONFIG" = "nosanafl" ]; then
    AFL_NO_AFFINITY=1 CUFUZZ_MAP_SIZE=65536 AFL_MAP_SIZE=65536 AFL_SKIP_CPUFREQ=1 ${CUFUZZ_PATH}/Tools/AFLplusplus/afl-fuzz -i $CUFUZZ_PATH/data/seeds/nvjpeg/ -o fuzz/${DSTN}/out_seeds_nosan_afl_${RUN} -t 120000 -V ${TIME} -Z -x $CUFUZZ_PATH/data/dictionaries/jpeg.dict ./nvjpeg_harness.afl.86 @@ rgb 
fi

# sancovinc4hd is the "cufuzz-coverageIncrease" mode
if [ "$CONFIG" = "sancovinc4hd" ]; then
    AFL_NO_AFFINITY=1 AFL_SAN_ABSTRACTION=coverage_increase ASAN_APP="./nvjpeg_harness.asan.86" ORIGINAL_APP="./nvjpeg_harness.vanilla.86" SANITIZER_PATH=${CUDA_PATH:-/usr/local/cuda}/bin/compute-sanitizer SANITIZER_ARG="--tool=memcheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_RACE="--tool=racecheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_INIT="--tool=initcheck --report-api-errors=no --error-exitcode 99" CUFUZZ_MAP_SIZE=65536 AFL_MAP_SIZE=65536 AFL_SKIP_CPUFREQ=1 EDGE_COV=1 COUNT_WARP_LEVEL=1 COV_CTRL=3 AFL_PRELOAD=$CUFUZZ_PATH/src/cufuzz_cov.so ${CUFUZZ_PATH}/Tools/AFLplusplus/afl-fuzz -i $CUFUZZ_PATH/data/seeds/nvjpeg/ -o fuzz/${DSTN}/out_seeds_sancovinc_MRIA_hd_${RUN} -t 120000 -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_memcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_racecheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_initcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_asan.out -V ${TIME} -Z -x $CUFUZZ_PATH/data/dictionaries/jpeg.dict ./nvjpeg_harness.afl.86 @@ rgb
fi

# sanuniqtrace4hd is the "cufuzz-uniqueTrace" mode
if [ "$CONFIG" = "sanuniqtrace4hd" ]; then
    AFL_NO_AFFINITY=1 AFL_SAN_ABSTRACTION=unique_trace ASAN_APP="./nvjpeg_harness.asan.86" ORIGINAL_APP="./nvjpeg_harness.vanilla.86" SANITIZER_PATH=${CUDA_PATH:-/usr/local/cuda}/bin/compute-sanitizer SANITIZER_ARG="--tool=memcheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_RACE="--tool=racecheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_INIT="--tool=initcheck --report-api-errors=no --error-exitcode 99" CUFUZZ_MAP_SIZE=65536 AFL_MAP_SIZE=65536 AFL_SKIP_CPUFREQ=1 EDGE_COV=1 COUNT_WARP_LEVEL=1 COV_CTRL=3 AFL_PRELOAD=$CUFUZZ_PATH/src/cufuzz_cov.so ${CUFUZZ_PATH}/Tools/AFLplusplus/afl-fuzz -i $CUFUZZ_PATH/data/seeds/nvjpeg/ -o fuzz/${DSTN}/out_seeds_sanuniqtrace_MRIA_hd_${RUN} -t 120000 -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_memcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_racecheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_initcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_asan.out -V ${TIME} -Z -x $CUFUZZ_PATH/data/dictionaries/jpeg.dict ./nvjpeg_harness.afl.86 @@ rgb
fi

# sansimtrace4hd is the "cufuzz" mode which is also the default mode in our experiments. It is the "simplify trace" mode
if [ "$CONFIG" = "sansimtrace4hd" ]; then
    AFL_NO_AFFINITY=1 AFL_SAN_ABSTRACTION=simplify_trace ASAN_APP="./nvjpeg_harness.asan.86" ORIGINAL_APP="./nvjpeg_harness.vanilla.86" SANITIZER_PATH=${CUDA_PATH:-/usr/local/cuda}/bin/compute-sanitizer SANITIZER_ARG="--tool=memcheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_RACE="--tool=racecheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_INIT="--tool=initcheck --report-api-errors=no --error-exitcode 99" CUFUZZ_MAP_SIZE=65536 AFL_MAP_SIZE=65536 AFL_SKIP_CPUFREQ=1 EDGE_COV=1 COUNT_WARP_LEVEL=1 COV_CTRL=3 AFL_PRELOAD=$CUFUZZ_PATH/src/cufuzz_cov.so ${CUFUZZ_PATH}/Tools/AFLplusplus/afl-fuzz -i $CUFUZZ_PATH/data/seeds/nvjpeg/ -o fuzz/${DSTN}/out_seeds_sansimtrace_MRIA_hd_${RUN} -t 120000 -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_memcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_racecheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_initcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_asan.out -V ${TIME} -Z -x $CUFUZZ_PATH/data/dictionaries/jpeg.dict ./nvjpeg_harness.afl.86 @@ rgb
fi

# sansimtrace4afl is the "cufuzz-noDeviceCoverage" mode
if [ "$CONFIG" = "sansimtrace4afl" ]; then
    AFL_NO_AFFINITY=1 AFL_SAN_ABSTRACTION=simplify_trace ASAN_APP="./nvjpeg_harness.asan.86" ORIGINAL_APP="./nvjpeg_harness.vanilla.86" SANITIZER_PATH=${CUDA_PATH:-/usr/local/cuda}/bin/compute-sanitizer SANITIZER_ARG="--tool=memcheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_RACE="--tool=racecheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_INIT="--tool=initcheck --report-api-errors=no --error-exitcode 99" CUFUZZ_MAP_SIZE=65536 AFL_MAP_SIZE=65536 AFL_SKIP_CPUFREQ=1 ${CUFUZZ_PATH}/Tools/AFLplusplus/afl-fuzz -i $CUFUZZ_PATH/data/seeds/nvjpeg/ -o fuzz/${DSTN}/out_seeds_sansimtrace_MRIA_afl_${RUN} -t 120000 -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_memcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_racecheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_initcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_asan.out -V ${TIME} -Z -x $CUFUZZ_PATH/data/dictionaries/jpeg.dict ./nvjpeg_harness.afl.86 @@ rgb 
fi

# perssansimtrace4hd is the "persistent" mode
if [ "$CONFIG" = "perssansimtrace4hd" ]; then
    AFL_NO_AFFINITY=1 COV_PERSISTENT=1 AFL_SAN_ABSTRACTION=simplify_trace ASAN_APP="./${APP}_harness.asan.86" ORIGINAL_APP="./${APP}_harness.vanilla.86" SANITIZER_PATH=${CUDA_PATH:-/usr/local/cuda}/bin/compute-sanitizer SANITIZER_ARG="--tool=memcheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_RACE="--tool=racecheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_INIT="--tool=initcheck --report-api-errors=no --error-exitcode 99" CUFUZZ_MAP_SIZE=65536 AFL_MAP_SIZE=65536 AFL_SKIP_CPUFREQ=1 EDGE_COV=1 COUNT_WARP_LEVEL=1 COV_CTRL=3 AFL_PRELOAD=$CUFUZZ_PATH/src/cufuzz_cov.so ${CUFUZZ_PATH}/Tools/AFLplusplus/afl-fuzz -i $CUFUZZ_PATH/data/seeds/nvjpeg/ -o fuzz/${DSTN}/out_seeds_pers_sansimtrace_MRIA_hd_${RUN} -t 120000 -x $CUFUZZ_PATH/data/dictionaries/jpeg.dict -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_persistent_memcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_persistent_racecheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_persistent_initcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_persistent_asan_A40.out  -V ${TIME} -Z ./${APP}_harness_persistent.afl.86 @@ rgb 
fi

# sanall4hd is the "cufuzz-allTrace" mode
if [ "$CONFIG" = "sanall4hd" ]; then
    AFL_NO_AFFINITY=1 AFL_SAN_ABSTRACTION=all_trace ASAN_APP="./nvjpeg_harness.asan.86" ORIGINAL_APP="./nvjpeg_harness.vanilla.86" SANITIZER_PATH=${CUDA_PATH:-/usr/local/cuda}/bin/compute-sanitizer SANITIZER_ARG="--tool=memcheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_RACE="--tool=racecheck --report-api-errors=no --error-exitcode 99" SANITIZER_ARG_INIT="--tool=initcheck --report-api-errors=no --error-exitcode 99" CUFUZZ_MAP_SIZE=65536 AFL_MAP_SIZE=65536 AFL_SKIP_CPUFREQ=1 EDGE_COV=1 COUNT_WARP_LEVEL=1 COV_CTRL=3 AFL_PRELOAD=$CUFUZZ_PATH/src/cufuzz_cov.so ${CUFUZZ_PATH}/Tools/AFLplusplus/afl-fuzz -i $CUFUZZ_PATH/data/seeds/nvjpeg/ -o fuzz/${DSTN}/out_seeds_sanall_MRIA_hd_${RUN} -t 120000 -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_memcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_racecheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_initcheck.out -w ${CUFUZZ_PATH}/src/cufuzz_sand/wrapper_asan.out -V ${TIME} -Z -x $CUFUZZ_PATH/data/dictionaries/jpeg.dict ./nvjpeg_harness.afl.86 @@ rgb
fi
