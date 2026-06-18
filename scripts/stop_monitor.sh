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
# stop_monitor.sh - Stop Process Monitor Daemon
# Usage: ./stop_monitor.sh [log_file_name]

LOG_FILE="${1:-afl_process_monitor.log}"  # Use first argument or default
PID_FILE="/tmp/afl_monitor.pid"

echo "Stopping AFL++ Process Monitor..."
echo "Log file: $LOG_FILE"

# Check if PID file exists
if [[ ! -f "$PID_FILE" ]]; then
    echo "PID file not found: $PID_FILE"
    echo "Trying to find and kill monitor processes..."
    pkill -f "process_monitor.sh"
    exit 0
fi

# Read PID from file
MONITOR_PID=$(cat "$PID_FILE" 2>/dev/null)

if [[ -z "$MONITOR_PID" ]]; then
    echo "Invalid PID in file: $PID_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

# Check if process is still running
if ! kill -0 "$MONITOR_PID" 2>/dev/null; then
    echo "Process $MONITOR_PID is not running"
    rm -f "$PID_FILE"
    exit 0
fi

# Kill the process
echo "Killing process $MONITOR_PID..."
kill "$MONITOR_PID"

# Wait a bit and check if it's still alive
sleep 2
if kill -0 "$MONITOR_PID" 2>/dev/null; then
    echo "Process still alive, sending SIGKILL..."
    kill -KILL "$MONITOR_PID"
fi

# Clean up PID file
rm -f "$PID_FILE"

echo "Process monitor stopped." 