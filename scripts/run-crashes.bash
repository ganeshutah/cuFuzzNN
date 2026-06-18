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

# Default folder to search in
FOLDER_TYPE=${1:-crashes}

# Check if we have enough arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 [folder_type] <program_path> <program_args> <log_file_path> <afl_output_folder1> [afl_output_folder2 ...]"
    echo "Example: $0 crashes ./my_program '--input XYZ --output out.txt' ./crash_analysis.log ./afl_output1 ./afl_output2"
    echo "Note: Use XYZ as a placeholder in program_args where the crash file should be inserted"
    echo "      folder_type defaults to 'crashes' if not specified"
    exit 1
fi

# Get the program path and arguments
PROGRAM="$2"
PROGRAM_ARGS="$3"
LOG_FILE="$4"
shift 4

# Initialize the log file with a header
echo "Crash Analysis Log - $(date)" > "$LOG_FILE"
echo "Program: $PROGRAM" >> "$LOG_FILE"
echo "Arguments template: $PROGRAM_ARGS" >> "$LOG_FILE"
echo "Searching in folder: $FOLDER_TYPE" >> "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

# Process each AFL output folder
for folder in "$@"; do
    if [ ! -d "$folder" ]; then
        echo "Warning: $folder is not a directory, skipping..."
        continue
    fi

    # Find all files in the specified folder
    if [ "$FOLDER_TYPE" = "queue" ]; then
        # For queue folder, exclude .state/ subdirectory and its contents
        find "$folder" -path "*/${FOLDER_TYPE}/*" -type f ! -path "*/.state/*"
    else
        # For other folder types, include all files
        find "$folder" -path "*/${FOLDER_TYPE}/*" -type f
    fi | while read -r crash_file; do
        echo "Processing file: $crash_file" >> "$LOG_FILE"
        echo "------------------------------------------" >> "$LOG_FILE"
        
        # Replace XYZ with the actual file path
        current_args="${PROGRAM_ARGS//XYZ/$crash_file}"
        
        # Create temporary files for stdout and stderr
        stdout_temp=$(mktemp)
        stderr_temp=$(mktemp)
        
        # Run the program and capture stdout and stderr separately
        "$PROGRAM" $current_args > "$stdout_temp" 2> "$stderr_temp"
        exit_code=$?
        
        # Log stdout
        echo "STDOUT:" >> "$LOG_FILE"
        echo "------------------------------------------" >> "$LOG_FILE"
        cat "$stdout_temp" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
        
        # Log stderr
        echo "STDERR:" >> "$LOG_FILE"
        echo "------------------------------------------" >> "$LOG_FILE"
        cat "$stderr_temp" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
        
        # Log exit code
        echo "Exit Code: $exit_code" >> "$LOG_FILE"
        
        # Clean up temporary files
        rm -f "$stdout_temp" "$stderr_temp"
        
        # Add a separator between different files
        echo -e "\n==========================================\n" >> "$LOG_FILE"
    done
done

echo "Analysis complete. Results written to $LOG_FILE" 