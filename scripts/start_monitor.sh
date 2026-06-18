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
# start_monitor.sh - Start Process Monitor Daemon
#
# Starts the process monitor in the background. The monitor kills zombie/hanging
# processes that exceed the timeout threshold, preventing them from corrupting
# the GPU driver state during fuzzing campaigns.
# Usage: ./start_monitor.sh [log_file_name] [max_runtime_minutes] [check_interval_seconds]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONITOR_SCRIPT="$SCRIPT_DIR/process_monitor.sh"

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

echo "Starting AFL++ Process Monitor..."
echo "Log file: $LOG_FILE"
echo "Max runtime: ${MAX_RUNTIME_MINUTES} minutes"
echo "Check interval: ${CHECK_INTERVAL_SECONDS} seconds"
echo "Monitor script: $MONITOR_SCRIPT"

# Check if monitor is already running
if pgrep -f "process_monitor.sh" > /dev/null; then
    echo "Process monitor is already running, skipping start."
    echo "Running processes:"
    pgrep -f "process_monitor.sh" -a
    exit 0  # Exit successfully - monitor is already active
fi

# Start the monitor in background with all parameters
nohup "$MONITOR_SCRIPT" "$LOG_FILE" "$MAX_RUNTIME_MINUTES" "$CHECK_INTERVAL_SECONDS" > /dev/null 2>&1 &

# Get the PID
MONITOR_PID=$!
echo "Process monitor started with PID: $MONITOR_PID"

# Save PID to file for easy management
echo "$MONITOR_PID" > /tmp/afl_monitor.pid

echo "Monitor is now running in background."
echo "To stop it, run: kill \$(cat /tmp/afl_monitor.pid)"
echo "To view logs: tail -f $LOG_FILE" 