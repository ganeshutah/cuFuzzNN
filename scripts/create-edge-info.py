#!/usr/bin/env python3
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

"""
create-edge-info.py - Compute edge coverage progression from AFL queues

This script analyzes AFL fuzzing queue directories to compute edge coverage
statistics over time. It uses AFL's afl-showmap to measure coverage for each
queue entry and tracks both host-side and device-side (GPU) edge coverage.

Usage:
    python3 create-edge-info.py <BINARY> [options] [-- BINARY_ARGS]

Arguments:
    BINARY  - Path to the instrumented target binary

Options:
    --input-folder PATH  - AFL queue folder to process
    --edge-data FILE     - Output CSV file path (default: edge_data.csv)
    --output FILE        - Temporary map output file (default: tmap.log)
    --afl-half-map N     - Threshold for host/device ID separation (default: 32768)
    --cov-ctrl N         - Coverage control mode (default: 3)

Environment:
    CUFUZZ_PATH - Path to cuFuzz installation (required)

Output CSV columns:
    testcase, time, execs, new_findings, total_edges, host_edges, device_edges,
    abs_total_edges, abs_host_edges, abs_device_edges

Example:
    export CUFUZZ_PATH=/path/to/cufuzz
    python3 create-edge-info.py ./target_binary \\
        --input-folder fuzz/out/queue \\
        --edge-data coverage.csv \\
        --cov-ctrl 3 \\
        -- @@ output_arg
"""

import os
import argparse
import subprocess
from typing import List, Dict, Tuple
import glob
import csv
from datetime import datetime

# Constants
AFL_MAP_SIZE = "65536"
CUFUZZ_MAP_SIZE = "65536"
AFL_SKIP_CPUFREQ = "1"
EDGE_COV = "1"
COUNT_WARP_LEVEL = "1"
COV_CTRL = "3"
TIMEOUT = "120000"  # Default timeout in milliseconds
TEMP_OUT = "tmap.log"  # Default output file
XYZ_PLACEHOLDER = "XYZ"  # Placeholder to be replaced with input files
EDGE_DATA_FILE = "edge_data.csv"  # CSV output file
AFL_HALF_MAP = "32768"  # Default threshold for host/device separation
TEST_COV_CTRL = "3" 
DEV_START = "32768"  # Default threshold for host/device separation


# Global golden map to store all unique byte IDs
golden_map: Dict[str, int] = {}

def count_host_device_ids(map_dict: Dict[str, int], half_map: int) -> Tuple[int, int]:
    """Count the number of host and device IDs in the map."""
    host_count = 0
    device_count = 0
    
    for byte_id in map_dict.keys():
        try:
            id_value = int(byte_id)
            if id_value < half_map:
                host_count += 1
            else:
                device_count += 1
        except ValueError:
            continue
    
    return host_count, device_count

def parse_testcase_info(testcase: str) -> Tuple[int, int]:
    """Parse testcase name to extract time and execs values."""
    time_val = 0
    execs_val = 0
    
    # Split by commas to get individual components
    parts = testcase.split(',')
    for part in parts:
        if part.startswith('time:'):
            try:
                time_val = int(part.split(':')[1])
            except (ValueError, IndexError):
                pass
        elif part.startswith('execs:'):
            try:
                execs_val = int(part.split(':')[1])
            except (ValueError, IndexError):
                pass
    
    return time_val, execs_val

def run_afl_showmap(app: str, app_inputs: List[str], timeout: str = TIMEOUT, temp_out: str = TEMP_OUT,
                   afl_map_size: str = AFL_MAP_SIZE, cufuzz_map_size: str = CUFUZZ_MAP_SIZE, afl_skip_cpufreq: str = AFL_SKIP_CPUFREQ,
                   edge_cov: str = EDGE_COV, count_warp_level: str = COUNT_WARP_LEVEL,
                   cov_ctrl: str = COV_CTRL, dev_start: str = DEV_START, test_cov_ctrl: str = TEST_COV_CTRL):
    # Get CUFUZZ_PATH from environment variable
    cufuzz_path = os.environ.get('CUFUZZ_PATH')
    if not cufuzz_path:
        raise ValueError("CUFUZZ_PATH environment variable is not set")

    # Construct the environment variables
    env = os.environ.copy()
    env.update({
        'AFL_MAP_SIZE': afl_map_size,
        'CUFUZZ_MAP_SIZE': cufuzz_map_size,
        'AFL_SKIP_CPUFREQ': afl_skip_cpufreq,
        'EDGE_COV': edge_cov,
        'COUNT_WARP_LEVEL': count_warp_level,
        'COV_CTRL': cov_ctrl,
        'DEV_START': dev_start,
        'TEST_COV_CTRL': test_cov_ctrl,
        #'AFL_PRELOAD': f"{cufuzz_path}/src/cufuzz_cov_16.so"
        'AFL_PRELOAD': f"{cufuzz_path}/src/cufuzz_cov_8.so"
    })

    # Construct the command
    cmd = [
        f"{cufuzz_path}/Tools/AFLplusplus/afl-showmap",
        "-o", temp_out,
        "-t", timeout,
        app
    ]
    
    # Add all app inputs
    cmd.extend(app_inputs)

    # Run the command
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"Successfully created edge info in {temp_out}")
    except subprocess.CalledProcessError as e:
        print(f"Error running afl-showmap: {e}")
        raise

def parse_tmap_content(content: str) -> Dict[str, int]:
    """Parse tmap content into a dictionary of byte_id:value pairs."""
    temp_map = {}
    for line in content.strip().split('\n'):
        if ':' in line:
            byte_id, value = line.split(':')
            temp_map[byte_id] = int(value)
    return temp_map

def read_tmap_content(temp_out: str) -> str:
    """Read and return the content of the tmap file."""
    try:
        with open(temp_out, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "Error: tmap file not found"
    except Exception as e:
        return f"Error reading tmap file: {str(e)}"

def write_edge_data(testcase: str, new_findings: int, total_edges: int, host_edges: int, device_edges: int, 
                   abs_total_edges: int, abs_host_edges: int, abs_device_edges: int):
    """Write edge data to CSV file."""
    file_exists = os.path.isfile(EDGE_DATA_FILE)
    
    # Parse time and execs from testcase name
    time_val, execs_val = parse_testcase_info(testcase)
    
    with open(EDGE_DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            f.write("#")  # Add # before the header
            writer.writerow(['testcase', 'time', 'execs', 'new_findings', 'total_edges', 'host_edges', 'device_edges', 
                           'abs_total_edges', 'abs_host_edges', 'abs_device_edges'])
        writer.writerow([testcase, time_val, execs_val, new_findings, total_edges, host_edges, device_edges,
                        abs_total_edges, abs_host_edges, abs_device_edges])

def compare_and_update_maps(temp_map: Dict[str, int], input_file: str, half_map: int) -> None:
    """Compare temp_map with golden_map and update golden_map if new entries found."""
    new_entries = {}
    
    # Find new entries
    for byte_id, value in temp_map.items():
        if byte_id not in golden_map:
            new_entries[byte_id] = value
            golden_map[byte_id] = value
    
    # Count host and device IDs for accumulated values
    host_count, device_count = count_host_device_ids(golden_map, half_map)
    
    # Count host and device IDs for absolute values (from current run only)
    abs_host_count, abs_device_count = count_host_device_ids(temp_map, half_map)
    abs_total_count = len(temp_map)
    
    # Report results
    if not new_entries:
        print(f"No new byte IDs found in {input_file}")
        write_edge_data(input_file, 0, len(golden_map), host_count, device_count, 
                       abs_total_count, abs_host_count, abs_device_count)
    else:
        print(f"\nFound {len(new_entries)} new byte IDs in {input_file}:")
        for byte_id, value in new_entries.items():
            print(f"  {byte_id}:{value}")
        print(f"Golden map now contains {len(golden_map)} unique byte IDs: {host_count} on the host and {device_count} on the device")
        write_edge_data(input_file, len(new_entries), len(golden_map), host_count, device_count,
                       abs_total_count, abs_host_count, abs_device_count)

def extract_id_number(filename: str) -> int:
    """Extract the numeric ID from a filename like 'id:000251'."""
    try:
        # Get just the filename without path
        basename = os.path.basename(filename)
        # Extract the number after 'id:'
        id_str = basename.split('id:')[1].split(',')[0]
        return int(id_str)
    except (IndexError, ValueError):
        # If we can't parse the ID, return a large number to put it at the end
        return float('inf')

def process_input_folder(app: str, app_inputs: List[str], input_folder: str, half_map: int, **kwargs):
    """Process all files in the input folder, replacing XYZ with each file."""
    if not os.path.isdir(input_folder):
        raise ValueError(f"Input folder {input_folder} does not exist or is not a directory")

    # Get all files in the input folder and sort them by ID number
    input_files = glob.glob(os.path.join(input_folder, "*"))
    input_files.sort(key=extract_id_number)
    
    if not input_files:
        print(f"Warning: No files found in {input_folder}")
        return

    # Check if XYZ exists in app_inputs
    if XYZ_PLACEHOLDER not in app_inputs:
        print(f"Warning: {XYZ_PLACEHOLDER} placeholder not found in app_inputs. Running with original inputs.")
        run_afl_showmap(app, app_inputs, **kwargs)
        return

    # Process each input file
    for input_file in input_files:
        # Create a copy of app_inputs and replace XYZ with the current input file
        current_inputs = [x.replace(XYZ_PLACEHOLDER, input_file) for x in app_inputs]
        
        print(f"\nProcessing input file: {input_file}")
        
        try:
            run_afl_showmap(app, current_inputs, **kwargs)
            # Read and parse the content of tmap.log
            tmap_content = read_tmap_content(kwargs['temp_out'])
            temp_map = parse_tmap_content(tmap_content)
            # Compare and update maps
            compare_and_update_maps(temp_map, input_file, half_map)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {input_file}: {e}")
            continue

def main():
    global EDGE_DATA_FILE
    
    parser = argparse.ArgumentParser(description='Run AFL showmap with CUDA fuzzing configuration')
    parser.add_argument('app', help='Path to the application to run')
    parser.add_argument('--input-folder', help='Folder containing input files to process (will replace XYZ in app_inputs)')
    parser.add_argument('--timeout', default=TIMEOUT, help=f'Timeout in milliseconds (default: {TIMEOUT})')
    parser.add_argument('--output', default=TEMP_OUT, help=f'Output file path (default: {TEMP_OUT})')
    parser.add_argument('--edge-data', default=EDGE_DATA_FILE, help=f'Edge data CSV output file (default: {EDGE_DATA_FILE})')
    parser.add_argument('--afl-half-map', default=AFL_HALF_MAP, help=f'Threshold for host/device ID separation (default: {AFL_HALF_MAP})')
    
    # Add arguments for environment variables
    parser.add_argument('--afl-map-size', default=AFL_MAP_SIZE,
                       help=f'AFL map size (default: {AFL_MAP_SIZE})')
    parser.add_argument('--cufuzz-map-size', default=CUFUZZ_MAP_SIZE,
                       help=f'CUFUZZ map size (default: {CUFUZZ_MAP_SIZE})')
    parser.add_argument('--afl-skip-cpufreq', default=AFL_SKIP_CPUFREQ,
                       help=f'Skip CPU frequency check (default: {AFL_SKIP_CPUFREQ})')
    parser.add_argument('--edge-cov', default=EDGE_COV,
                       help=f'Enable edge coverage (default: {EDGE_COV})')
    parser.add_argument('--count-warp-level', default=COUNT_WARP_LEVEL,
                       help=f'Count warp level (default: {COUNT_WARP_LEVEL})')
    parser.add_argument('--cov-ctrl', default=COV_CTRL,
                       help=f'Coverage control (default: {COV_CTRL})')
    parser.add_argument('--dev-start', default=DEV_START,
                       help=f'Device-side start threshold (default: {DEV_START})')
    parser.add_argument('--test-cov-ctrl', default=TEST_COV_CTRL,
                       help=f'Test coverage control (default: {TEST_COV_CTRL})')
    
    # Parse known args first to get script arguments
    args, app_inputs = parser.parse_known_args()
    
    # Update edge data file path if specified
    EDGE_DATA_FILE = args.edge_data

    # Convert half_map to integer
    try:
        half_map = int(args.afl_half_map)
    except ValueError:
        print(f"Warning: Invalid AFL_HALF_MAP value '{args.afl_half_map}', using default {AFL_HALF_MAP}")
        half_map = int(AFL_HALF_MAP)

    kwargs = {
        'timeout': args.timeout,
        'temp_out': args.output,
        'afl_map_size': args.afl_map_size,
        'cufuzz_map_size': args.cufuzz_map_size,
        'afl_skip_cpufreq': args.afl_skip_cpufreq,
        'edge_cov': args.edge_cov,
        'count_warp_level': args.count_warp_level,
        'cov_ctrl': args.cov_ctrl,
        'dev_start': args.dev_start,
        'test_cov_ctrl': args.test_cov_ctrl
    }

    if args.input_folder:
        process_input_folder(args.app, app_inputs, args.input_folder, half_map, **kwargs)
    else:
        run_afl_showmap(args.app, app_inputs, **kwargs)

if __name__ == "__main__":
    main() 
