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
# create-edge-csv.bash - Generate edge coverage CSV from AFL queue
#
# This script runs the edge coverage analysis tool on an AFL output queue
# to produce a CSV file with coverage progression data. The CSV can be used
# for plotting coverage over time.
#
# Usage:
#   ./create-edge-csv.bash <PATH> <LOG_NUMBER> <BINARY_NAME>
#
# Arguments:
#   PATH        - Path to the AFL output directory containing the queue
#   LOG_NUMBER  - Numeric identifier for log files (e.g., 1, 2, 3)
#   BINARY_NAME - Name of the target binary (e.g., nvtiff_example.afl.86)
#
# Environment:
#   CUFUZZ_PATH - Path to cuFuzz installation (required)
#
# Output:
#   Creates edge.csv in the specified PATH directory with columns:
#   - testcase, time, execs, new_findings, total_edges, host_edges, device_edges
#
# Example:
#   export CUFUZZ_PATH=/path/to/cufuzz
#   ./create-edge-csv.bash fuzz/out_seeds_cufuzz_1/default 1 nvtiff_example.afl.86
#

# Check if correct number of arguments provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <path> <log_number> <binary_name>"
    echo "Example: $0 fuzz/exp_fpnotify_A40/out_seeds_sanall_csan_hd_1/default 1 nvtiff_example_persistent_base_external.afl.86"
    exit 1
fi

# Assign parameters
MY_PATH="$1"
LOG_NUM="$2"
BINARY_NAME="$3"

# Define log file names
T_LOG="t${LOG_NUM}.log"
X_LOG="x${LOG_NUM}.log"

# Check if edge.csv already exists and warn user
if [ -f "$MY_PATH/edge.csv" ]; then
    echo "WARNING: File '$MY_PATH/edge.csv' already exists and will be overwritten!"
    echo -n "Do you want to continue? (y/N): "
    read -r response
    case "$response" in
        [yY][eE][sS]|[yY])
            echo "Continuing with execution..."
            ;;
        *)
            echo "Aborting script execution."
            exit 1
            ;;
    esac
fi

echo "Running create-edge-info.py with:"
echo "  Path: $MY_PATH"
echo "  Log number: $LOG_NUM"
echo "  Binary: $BINARY_NAME"
echo "  Output logs: $T_LOG, $X_LOG"

# Export the path and run the command
export MY_PATH="$MY_PATH"
python3 $CUFUZZ_PATH/scripts/create-edge-info.py --edge-data "$MY_PATH/edge.csv" --output "$X_LOG" --input-folder "$MY_PATH/queue" --cov-ctrl 3 "$BINARY_NAME" XYZ &> "$T_LOG" &

# Get the process ID of the background job
PID=$!

echo "Process started with PID: $PID"
echo "Waiting for completion..."

# Wait for the background process to complete
wait $PID
EXIT_CODE=$?

echo "Process completed with exit code: $EXIT_CODE"

# Handle success vs failure cases
if [ $EXIT_CODE -eq 0 ]; then
    echo "Success! Cleaning up intermediate log files..."
    # Clean up intermediate log files on success
    if [ -f "$T_LOG" ]; then
        rm "$T_LOG"
        echo "Deleted $T_LOG"
    fi
    
    if [ -f "$X_LOG" ]; then
        rm "$X_LOG"
        echo "Deleted $X_LOG"
    fi
    
    echo "Script completed successfully!"
else
    echo "ERROR: Process failed with exit code $EXIT_CODE"
    echo "Preserving log files for debugging:"
    
    if [ -f "$T_LOG" ]; then
        echo "Error log contents ($T_LOG):"
        echo "========================="
        cat "$T_LOG"
        echo "========================="
        echo "Log file preserved at: $T_LOG"
    fi
    
    if [ -f "$X_LOG" ]; then
        echo "Output log preserved at: $X_LOG"
    fi
    
    echo "Please check the error logs above for debugging information."
fi

exit $EXIT_CODE
