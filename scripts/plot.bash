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
# plot.bash - Generate coverage plots for cuFuzz paper figures
#
# This script generates coverage plots (edges and inputs over time) for both
# HeCBench applications and NVIDIA library targets. The plots compare different
# cuFuzz configurations as shown in the paper.
#
# Usage:
#   ./plot.bash <TYPE> <FOLDER> <TARGET> <TIME_HOURS> <RUN_ID>
#
# Arguments:
#   TYPE       - Target type: "app" for HeCBench, "lib" for NVIDIA libraries
#   FOLDER     - Output folder name for generated plots
#   TARGET     - Target application/library name
#   TIME_HOURS - X-axis limit in hours
#   RUN_ID     - Run identifier to select which run's data to plot
#
# Environment:
#   HECBENCH_PATH - Path to HeCBench fuzzing results (required for type=app)
#   EVAL_PATH     - Path to library fuzzing results (required for type=lib)
#   OUTPUT_PATH   - Path to store generated plots (required)
#
# Output:
#   - edges_combo_edges_<TARGET>.jpg  - Edge coverage over time
#   - edges_combo_inputs_<TARGET>.jpg - Unique inputs over time
#   - edges_sand_edges_<TARGET>.jpg   - Sanitization strategy comparison (edges)
#   - edges_sand_inputs_<TARGET>.jpg  - Sanitization strategy comparison (inputs)
#   - (Pers folder) - Persistent mode comparison plots
#
# Example:
#   export HECBENCH_PATH=/path/to/HeCBench/fuzz
#   export OUTPUT_PATH=/path/to/output
#   ./plot.bash app plotsMain bfs 24 1
#

set -e

export type=$1
export FOLDER=$2
export TARGET=$3
export TIME=$4
export RUN=$5

# Validate required environment variables based on type
if [ -z "$OUTPUT_PATH" ]; then
    echo "Error: OUTPUT_PATH environment variable is not set"
    exit 1
fi

if [ ${type} == "app" ]; then
    if [ -z "$HECBENCH_PATH" ]; then
        echo "Error: HECBENCH_PATH environment variable is not set"
        exit 1
    fi

    #paper plot 1 sansimtrace (edges) --> Figure 8
    export CPATH=${HECBENCH_PATH}/exp_hecbench/${TARGET}/out_seeds_ && python3 afl-plot-edges.py -f jpg --width 6 -a combo -x ${TIME} -p total --no-legend -c "cuFuzz:blue:${CPATH}sansimtrace_MRIA_hd_${RUN}/default/"  -c "cuFuzz-noSan:yellow:${CPATH}nosan_hd_${RUN}/default" -c "cuFuzz-noDcov:orange:${CPATH}sansimtrace_MRIA_afl_${RUN}/default/" -c "AFL++:red:${CPATH}nosan_afl_${RUN}/default" ${OUTPUT_PATH}/${FOLDER}
    mv ${OUTPUT_PATH}/${FOLDER}/edges_combo.jpg ${OUTPUT_PATH}/${FOLDER}/edges_combo_edges_${TARGET}.jpg

    #paper plot 1 sansimtrace (inputs) --> Figure 9
    export CPATH=${HECBENCH_PATH}/exp_hecbench/${TARGET}/out_seeds_ && python3 afl-plot-edges.py -f jpg --width 6 -a combo -x ${TIME} -p unique --no-legend -c "cuFuzz:blue:${CPATH}sansimtrace_MRIA_hd_${RUN}/default/"  -c "cuFuzz-noSan:yellow:${CPATH}nosan_hd_${RUN}/default" -c "cuFuzz-noDcov:orange:${CPATH}sansimtrace_MRIA_afl_${RUN}/default/" -c "AFL++:red:${CPATH}nosan_afl_${RUN}/default" ${OUTPUT_PATH}/${FOLDER}
    mv ${OUTPUT_PATH}/${FOLDER}/edges_combo.jpg ${OUTPUT_PATH}/${FOLDER}/edges_combo_inputs_${TARGET}.jpg

    #paper plot 2 sansimtrace (persistent) (edges) --> Figure 12
    export CPATH=${HECBENCH_PATH}/exp_hecbench/${TARGET}/out_seeds_ && python3 afl-plot-edges.py -f jpg --width 6 -a combo -x ${TIME} -p total --no-legend -c "cuFuzz:blue:${CPATH}sansimtrace_MRIA_hd_${RUN}/default/" -c "cuFuzz-persistent:black:${CPATH}pers_sansimtrace_MRIA_hd_${RUN}/default/" ${OUTPUT_PATH}/${FOLDER}Pers
    mv ${OUTPUT_PATH}/${FOLDER}Pers/edges_combo.jpg ${OUTPUT_PATH}/${FOLDER}Pers/edges_combo_edges_${TARGET}.jpg

    # Sanitization Strategy Comparison (edges) --> Figure 13
    export CPATH=${HECBENCH_PATH}/exp_hecbench/${TARGET}/out_seeds_ && python3 afl-plot-edges.py -f jpg --width 6 -a combo -x ${TIME} -p total --no-legend -c "cuFuzz-SimpleTrace:blue:${CPATH}sansimtrace_MRIA_hd_${RUN}/default/"  -c "cuFuzz-UniqueTrace:green:${CPATH}sanuniqtrace_MRIA_hd_${RUN}/default" -c "cuFuzz-CoverageIncrease:gray:${CPATH}sancovinc_MRIA_hd_${RUN}/default/" -c "cuFuzz-AllTrace:lightgreen:${CPATH}sanall_MRIA_hd_${RUN}/default" ${OUTPUT_PATH}/${FOLDER}
    mv ${OUTPUT_PATH}/${FOLDER}/edges_combo.jpg ${OUTPUT_PATH}/${FOLDER}/edges_sand_edges_${TARGET}.jpg

    # Sanitization Strategy Comparison (inputs) --> not shown in paper
    export CPATH=${HECBENCH_PATH}/exp_hecbench/${TARGET}/out_seeds_ && python3 afl-plot-edges.py -f jpg --width 6 -a combo -x ${TIME} -p unique --no-legend -c "cuFuzz-SimpleTrace:blue:${CPATH}sansimtrace_MRIA_hd_${RUN}/default/"  -c "cuFuzz-UniqueTrace:green:${CPATH}sanuniqtrace_MRIA_hd_${RUN}/default" -c "cuFuzz-CoverageIncrease:gray:${CPATH}sancovinc_MRIA_hd_${RUN}/default/" -c "cuFuzz-AllTrace:lightgreen:${CPATH}sanall_MRIA_hd_${RUN}/default" ${OUTPUT_PATH}/${FOLDER}
    mv ${OUTPUT_PATH}/${FOLDER}/edges_combo.jpg ${OUTPUT_PATH}/${FOLDER}/edges_sand_inputs_${TARGET}.jpg

elif [ ${type} == "lib" ]; then
    if [ -z "$EVAL_PATH" ]; then
        echo "Error: EVAL_PATH environment variable is not set"
        exit 1
    fi

    #paper plot sansimtrace (edges) --> Figure 8
    export CPATH=${EVAL_PATH}/${TARGET}/fuzz/exp_lib/out_seeds_ && python3 afl-plot-edges.py -f jpg --width 6 -a combo -x ${TIME} -p total --no-legend -c "cuFuzz:blue:${CPATH}sansimtrace_MRIA_hd_${RUN}/default/" -c "cuFuzz-noSan:yellow:${CPATH}nosan_hd_${RUN}/default" -c "cuFuzz-noDcov:orange:${CPATH}sansimtrace_MRIA_afl_${RUN}/default/" -c "AFL++:red:${CPATH}nosan_afl_${RUN}/default" ${OUTPUT_PATH}/${FOLDER}
    mv ${OUTPUT_PATH}/${FOLDER}/edges_combo.jpg ${OUTPUT_PATH}/${FOLDER}/edges_combo_edges_${TARGET}.jpg

    #paper plot sansimtrace (inputs) --> Figure 9
    export CPATH=${EVAL_PATH}/${TARGET}/fuzz/exp_lib/out_seeds_ && python3 afl-plot-edges.py -f jpg --width 6 -a combo -x ${TIME} -p unique --no-legend -c "cuFuzz:blue:${CPATH}sansimtrace_MRIA_hd_${RUN}/default/" -c "cuFuzz-noSan:yellow:${CPATH}nosan_hd_${RUN}/default" -c "cuFuzz-noDcov:orange:${CPATH}sansimtrace_MRIA_afl_${RUN}/default/" -c "AFL++:red:${CPATH}nosan_afl_${RUN}/default" ${OUTPUT_PATH}/${FOLDER}
    mv ${OUTPUT_PATH}/${FOLDER}/edges_combo.jpg ${OUTPUT_PATH}/${FOLDER}/edges_combo_inputs_${TARGET}.jpg

    #paper plot sansimtrace (persistent) (edges) --> Figure 12
    export CPATH=${EVAL_PATH}/${TARGET}/fuzz/exp_lib/out_seeds_ && python3 afl-plot-edges.py -f jpg --width 6 -a combo -x ${TIME} -p total --no-legend -c "cuFuzz:blue:${CPATH}sansimtrace_MRIA_hd_${RUN}/default/" -c "cuFuzz-persistent:black:${CPATH}pers_sansimtrace_MRIA_hd_${RUN}/default/" ${OUTPUT_PATH}/${FOLDER}Pers
    mv ${OUTPUT_PATH}/${FOLDER}Pers/edges_combo.jpg ${OUTPUT_PATH}/${FOLDER}Pers/edges_combo_edges_${TARGET}.jpg

    # Sanitization Strategy Comparison (edges) --> Figure 13
    export CPATH=${EVAL_PATH}/${TARGET}/fuzz/exp_lib/out_seeds_ && python3 afl-plot-edges.py -f jpg --width 6 -a combo -x ${TIME} -p total --no-legend -c "cuFuzz-SimpleTrace:blue:${CPATH}sansimtrace_MRIA_hd_${RUN}/default/"  -c "cuFuzz-UniqueTrace:green:${CPATH}sanuniqtrace_MRIA_hd_${RUN}/default" -c "cuFuzz-CoverageIncrease:gray:${CPATH}sancovinc_MRIA_hd_${RUN}/default/" -c "cuFuzz-AllTrace:lightgreen:${CPATH}sanall_MRIA_hd_${RUN}/default" ${OUTPUT_PATH}/${FOLDER}
    mv ${OUTPUT_PATH}/${FOLDER}/edges_combo.jpg ${OUTPUT_PATH}/${FOLDER}/edges_sand_edges_${TARGET}.jpg

    # Sanitization Strategy Comparison (inputs) --> not shown in paper
    export CPATH=${EVAL_PATH}/${TARGET}/fuzz/exp_lib/out_seeds_ && python3 afl-plot-edges.py -f jpg --width 6 -a combo -x ${TIME} -p unique --no-legend -c "cuFuzz-SimpleTrace:blue:${CPATH}sansimtrace_MRIA_hd_${RUN}/default/"  -c "cuFuzz-UniqueTrace:green:${CPATH}sanuniqtrace_MRIA_hd_${RUN}/default" -c "cuFuzz-CoverageIncrease:gray:${CPATH}sancovinc_MRIA_hd_${RUN}/default/" -c "cuFuzz-AllTrace:lightgreen:${CPATH}sanall_MRIA_hd_${RUN}/default" ${OUTPUT_PATH}/${FOLDER}
    mv ${OUTPUT_PATH}/${FOLDER}/edges_combo.jpg ${OUTPUT_PATH}/${FOLDER}/edges_sand_inputs_${TARGET}.jpg
fi





