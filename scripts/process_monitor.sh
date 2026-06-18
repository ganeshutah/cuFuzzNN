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
# process_monitor.sh - Process Monitor for SAND Mode
#
# Monitors and kills zombie/hanging processes that exceed the timeout threshold.
# This prevents long-running processes from corrupting the GPU driver state
# during fuzzing campaigns with Compute Sanitizer or ASan.
# Usage: ./process_monitor.sh [log_file_name] [max_runtime_minutes] [check_interval_seconds]

# Show usage if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [log_file_name] [max_runtime_minutes] [check_interval_seconds]"
    echo "  log_file_name: Path to log file (default: afl_process_monitor.log)"
    echo "  max_runtime_minutes: Maximum runtime before killing process (default: 10)"
    echo "  check_interval_seconds: How often to check for long-running processes (default: 600)"
    echo ""
    echo "Environment variables (override command line args):"
    echo "  MAX_RUNTIME_MINUTES: Maximum runtime in minutes"
    echo "  CHECK_INTERVAL_SECONDS: Check interval in seconds"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use all defaults"
    echo "  $0 my_monitor.log                     # Custom log file"
    echo "  $0 my_monitor.log 15 300             # 15min max runtime, check every 5min"
    echo "  MAX_RUNTIME_MINUTES=20 $0            # Use environment variable"
    exit 0
fi

# Configuration - can be overridden by command line args or environment variables
LOG_FILE="${1:-afl_process_monitor.log}"  # Use first argument or default
MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-${2:-10}}"  # Env var, then arg, then default
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-${3:-600}}"  # Env var, then arg, then default
SANITIZER_PATTERNS=("example_persistent_base_external.asan.86" "example_persistent_base_external.vanilla.86" "build_vanilla_release_static" "build_vanilla_debug_static" "build_asan_debug_static" "compute-sanitizer" "build_vanilla_debug" "build_vanilla_release" "debug" "release" "vanilla" "asan")
WHITELIST_PATTERNS=("run-crashes.bash" "process_monitor.sh" "cleanup_cores.sh" "example_persistent_external.afl.86" "build_afl_release_static" "build_afl_debug" "build_afl_release" ".afl" "afl-fuzz" "afl-showmap" "afl-cmin" "afl-tmin" "afl-gotcpu" "afl-whatsup" "afl-plot" "afl-stat")

# Validate numeric inputs
if ! [[ "$MAX_RUNTIME_MINUTES" =~ ^[0-9]+$ ]]; then
    echo "Error: MAX_RUNTIME_MINUTES must be a positive integer, got: $MAX_RUNTIME_MINUTES"
    exit 1
fi

if ! [[ "$CHECK_INTERVAL_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "Error: CHECK_INTERVAL_SECONDS must be a positive integer, got: $CHECK_INTERVAL_SECONDS"
    exit 1
fi

# Function to log messages with timestamp
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOG_FILE"
}

# Function to parse time string from ps aux output
parse_time() {
    local time_str="$1"
    
    # Handle different time formats
    case "$time_str" in
        *-*:*)
            # Format: DD-HH:MM (days-hours:minutes)
            local days=$(echo "$time_str" | cut -d'-' -f1)
            local time_part=$(echo "$time_str" | cut -d'-' -f2)
            local hours=$(echo "$time_part" | cut -d':' -f1)
            local minutes=$(echo "$time_part" | cut -d':' -f2)
            echo $((days * 1440 + hours * 60 + minutes))
            ;;
        *:*)
            # Format: HH:MM (hours:minutes)
            local hours=$(echo "$time_str" | cut -d':' -f1)
            local minutes=$(echo "$time_str" | cut -d':' -f2)
            echo $((hours * 60 + minutes))
            ;;
        *)
            # Format: MM (minutes only)
            echo "$time_str"
            ;;
    esac
}

# Function to check if process matches our patterns
is_target_process() {
    local cmd="$1"
    
    # First check if process is in whitelist - if so, leave it alone
    for pattern in "${WHITELIST_PATTERNS[@]}"; do
        if [[ "$cmd" == *"$pattern"* ]]; then
            return 1  # Not a target process (whitelisted)
        fi
    done
    
    # Then check if it matches our sanitizer patterns
    for pattern in "${SANITIZER_PATTERNS[@]}"; do
        if [[ "$cmd" == *"$pattern"* ]]; then
            return 0  # Is a target process
        fi
    done
    return 1  # Not a target process
}

# Function to kill process and log
kill_process() {
    local pid="$1"
    local cmd="$2"
    local runtime="$3"
    
    log_message "KILLING: PID=$pid, CMD='$cmd', RUNTIME=${runtime}min"
    
    # Try SIGTERM first
    kill -TERM "$pid" 2>/dev/null
    sleep 2
    
    # Check if still alive, then SIGKILL
    if kill -0 "$pid" 2>/dev/null; then
        log_message "Process $pid still alive after SIGTERM, sending SIGKILL"
        kill -KILL "$pid" 2>/dev/null
        sleep 1
        
        # Final check
        if kill -0 "$pid" 2>/dev/null; then
            log_message "WARNING: Process $pid still alive after SIGKILL"
        else
            log_message "SUCCESS: Process $pid killed"
        fi
    else
        log_message "SUCCESS: Process $pid killed"
    fi
}

# Main monitoring loop
main() {
    log_message "Process monitor started - checking every ${CHECK_INTERVAL_SECONDS}s for processes running >${MAX_RUNTIME_MINUTES}min"
    
    while true; do
        log_message "Checking for long-running processes..."
        
        # Get processes with time info, skip header
        ps aux | tail -n +2 | while read -r line; do
            # Extract fields
            local pid=$(echo "$line" | awk '{print $2}')
            local time_col=$(echo "$line" | awk '{print $10}')
            local cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i}')
            
            # Skip if no PID or invalid
            if [[ -z "$pid" ]] || ! [[ "$pid" =~ ^[0-9]+$ ]]; then
                continue
            fi
            
            # Parse runtime
            local runtime_minutes=$(parse_time "$time_col")
            
            # Check if process has been running too long
            if [[ "$runtime_minutes" -gt "$MAX_RUNTIME_MINUTES" ]]; then
                # Check if it's a target process
                if is_target_process "$cmd"; then
                    log_message "FOUND LONG-RUNNING PROCESS: PID=$pid, CMD='$cmd', RUNTIME=${runtime_minutes}min"
                    kill_process "$pid" "$cmd" "$runtime_minutes"
                fi
            fi
        done
        
        log_message "Sleeping for ${CHECK_INTERVAL_SECONDS} seconds..."
        sleep "$CHECK_INTERVAL_SECONDS"
    done
}

# Handle script termination
cleanup() {
    log_message "Process monitor stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start monitoring
main 