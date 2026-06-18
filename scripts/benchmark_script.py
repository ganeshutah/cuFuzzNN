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
benchmark_script.py - Run hyperfine benchmarks on fuzzing inputs

This script uses hyperfine to measure execution time of a target binary
across multiple input files. It's used to evaluate the performance overhead
of different cuFuzz configurations.

Usage:
    python3 benchmark_script.py <INPUT_FOLDER> <COMMAND> [options]

Arguments:
    INPUT_FOLDER - Directory containing input files to benchmark
    COMMAND      - Command to execute (use $f as placeholder for input file)

Options:
    -o, --output FILE       - Output CSV file for benchmark results
    -t, --tag TAG           - Tag to add as first column in CSV
    -c, --concatenate       - Append to existing CSV instead of overwriting
    -i, --interactive       - Use hyperfine's interactive mode

Output CSV columns:
    [Tag], Filename, Size (bytes), Mean (ms), Std (ms), Min (ms), Max (ms)

Example:
    python3 benchmark_script.py seeds/ './target_binary "$f"' -o results.csv -t vanilla
    python3 benchmark_script.py seeds/ './target_binary "$f"' -o results.csv -t cufuzz -c
"""

import argparse
import subprocess
import sys
import os
import csv
import re
import shlex
from pathlib import Path

# Debug flag from environment variable
DEBUG = os.environ.get('DEBUG_BENCHMARK', '0') == '1'

def debug_print(*args, **kwargs):
    """Print debug messages only if DEBUG_BENCHMARK=1"""
    if DEBUG:
        print(*args, **kwargs)

def parse_hyperfine_output(output_text):
    """
    Parse hyperfine output to extract timing statistics.
    
    Args:
        output_text (str): Raw output from hyperfine command
        
    Returns:
        dict: Dictionary with timing statistics for each command
    """
    results = {}
    
    debug_print("DEBUG: Parsing hyperfine output...")
    debug_print("DEBUG: Output text length:", len(output_text))
    
    # Split output into lines and process each line
    lines = output_text.split('\n')
    debug_print("DEBUG: Found", len(lines), "lines")
    
    current_command = None
    timing_stats = {}
    
    def parse_time_value(time_str, unit):
        """Parse time value and convert to milliseconds"""
        try:
            value = float(time_str)
            if unit == 's':
                return value * 1000  # Convert seconds to milliseconds
            elif unit == 'ms':
                return value
            else:
                debug_print(f"DEBUG: Unknown time unit: {unit}")
                return value
        except ValueError:
            debug_print(f"DEBUG: Could not parse time value: {time_str}")
            return None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        debug_print(f"DEBUG: Processing line: {line}")
        
        # Check if this is a new benchmark
        if line.startswith('Benchmark '):
            # Save previous benchmark results if we have them
            if current_command and timing_stats:
                results[current_command] = timing_stats.copy()
                debug_print(f"DEBUG: Added results for command: {current_command}")
            
            # Start new benchmark
            cmd_match = re.search(r': (.+)$', line)
            if cmd_match:
                current_command = cmd_match.group(1)
                timing_stats = {}
                debug_print(f"DEBUG: New benchmark command: {current_command}")
            continue
        
        # Skip summary section
        if line.startswith('Summary'):
            # Save the last benchmark results
            if current_command and timing_stats:
                results[current_command] = timing_stats.copy()
                debug_print(f"DEBUG: Added final results for command: {current_command}")
            break
        
        # Parse timing statistics for current benchmark
        if current_command:
            if 'Time (mean ± σ):' in line:
                # Extract mean and std - handle both ms and s units
                # Pattern: Time (mean ± σ):     1.806 s ±  0.019 s
                match = re.search(r'Time \(mean ± σ\):\s+([\d.]+)\s+(\w+)\s+±\s+([\d.]+)\s+(\w+)', line)
                if match:
                    mean_val = parse_time_value(match.group(1), match.group(2))
                    std_val = parse_time_value(match.group(3), match.group(4))
                    if mean_val is not None and std_val is not None:
                        timing_stats['mean'] = mean_val
                        timing_stats['std'] = std_val
                        debug_print(f"DEBUG: Found mean={timing_stats['mean']}ms, std={timing_stats['std']}ms")
                    else:
                        debug_print(f"DEBUG: Could not parse mean/std values from line: {line}")
                else:
                    debug_print(f"DEBUG: No match for mean/std in line: {line}")
            elif 'Range (min … max):' in line:
                # Extract min and max - handle both ms and s units
                # Pattern: Range (min … max):    1.774 s …  1.834 s
                match = re.search(r'Range \(min … max\):\s+([\d.]+)\s+(\w+)\s+…\s+([\d.]+)\s+(\w+)', line)
                if match:
                    min_val = parse_time_value(match.group(1), match.group(2))
                    max_val = parse_time_value(match.group(3), match.group(4))
                    if min_val is not None and max_val is not None:
                        timing_stats['min'] = min_val
                        timing_stats['max'] = max_val
                        debug_print(f"DEBUG: Found min={timing_stats['min']}ms, max={timing_stats['max']}ms")
                    else:
                        debug_print(f"DEBUG: Could not parse min/max values from line: {line}")
                else:
                    debug_print(f"DEBUG: No match for min/max in line: {line}")
    
    debug_print(f"DEBUG: Final results count: {len(results)}")
    return results

def run_benchmark(input_folder, execution_command, output_file=None, tag=None, concatenate=False, interactive=False):
    """
    Run hyperfine benchmark on all files in the input folder using the provided execution command.
    
    Args:
        input_folder (str): Path to the folder containing input files
        execution_command (str): The command to execute for each file
        output_file (str, optional): Path to CSV output file
        tag (str, optional): Tag to add as first column in CSV
        concatenate (bool): If True, append to existing CSV file instead of overwriting
        interactive (bool): If True, add -i flag to hyperfine for interactive mode
    """
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)
    
    # Get all files in the input folder
    input_path = Path(input_folder)
    files = sorted([p for p in input_path.iterdir() if p.is_file()], key=lambda p: p.name)[:100]
    
    if not files:
        print(f"Error: No files found in '{input_folder}'.")
        sys.exit(1)
    
    # Build commands array and track file info
    commands = []
    file_info = {}  # command -> (filename, size)
    
    for file_path in files:
        if file_path.is_file():
            # Parse the execution command to get the base command and arguments
            if "$f" in execution_command:
                # Replace the placeholder with the properly quoted file path
                quoted_file_path = shlex.quote(str(file_path))
                command = execution_command.replace("$f", quoted_file_path)
                commands.append(command)
            else:
                # If no placeholder, just use the command as is
                commands.append(execution_command)
            
            # Store file information using the command as key
            file_info[commands[-1]] = (file_path.name, file_path.stat().st_size)
    
    if not commands:
        print(f"Error: No valid files found in '{input_folder}'.")
        sys.exit(1)
    
    # Prepare hyperfine command
    hyperfine_cmd = ["hyperfine", "--warmup", "2", "-r", "10"]
    if interactive:
        hyperfine_cmd.append("-i")
    hyperfine_cmd.extend(commands)
    
    print(f"Running benchmark on {len(commands)} files:")
    for cmd in commands:
        print(f"  {cmd}")
    print()
    
    # Execute hyperfine
    try:
        result = subprocess.run(hyperfine_cmd, check=True, capture_output=True, text=True)
        output_text = result.stdout
        print(output_text)
        
        # Parse results if output file is specified
        if output_file:
            timing_results = parse_hyperfine_output(output_text)
            
            # Determine fieldnames based on tag presence
            fieldnames = ['Tag', 'Filename', 'Size (bytes)', 'Mean (ms)', 'Std (ms)', 'Min (ms)', 'Max (ms)'] if tag else ['Filename', 'Size (bytes)', 'Mean (ms)', 'Std (ms)', 'Min (ms)', 'Max (ms)']
            
            # Check if file exists and we're concatenating
            file_exists = os.path.exists(output_file)
            
            if concatenate and file_exists:
                # Read existing file to get fieldnames
                with open(output_file, 'r', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    existing_fieldnames = reader.fieldnames
                    
                    # Ensure fieldnames match
                    if existing_fieldnames != fieldnames:
                        print(f"Warning: Existing CSV has different columns: {existing_fieldnames}")
                        print(f"Expected columns: {fieldnames}")
                        print("Results may not align correctly.")
                
                # Append to existing file
                with open(output_file, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    for command, timing_stats in timing_results.items():
                        if command in file_info:
                            filename, size = file_info[command]
                            row_data = {
                                'Filename': filename,
                                'Size (bytes)': size,
                                'Mean (ms)': timing_stats.get('mean', ''),
                                'Std (ms)': timing_stats.get('std', ''),
                                'Min (ms)': timing_stats.get('min', ''),
                                'Max (ms)': timing_stats.get('max', '')
                            }
                            
                            # Add tag as first column if provided
                            if tag:
                                row_data['Tag'] = tag
                            
                            writer.writerow(row_data)
                
                print(f"\nResults appended to: {output_file}")
            else:
                # Write new file or overwrite existing
                with open(output_file, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for command, timing_stats in timing_results.items():
                        if command in file_info:
                            filename, size = file_info[command]
                            row_data = {
                                'Filename': filename,
                                'Size (bytes)': size,
                                'Mean (ms)': timing_stats.get('mean', ''),
                                'Std (ms)': timing_stats.get('std', ''),
                                'Min (ms)': timing_stats.get('min', ''),
                                'Max (ms)': timing_stats.get('max', '')
                            }
                            
                            # Add tag as first column if provided
                            if tag:
                                row_data['Tag'] = tag
                            
                            writer.writerow(row_data)
                
                print(f"\nResults written to: {output_file}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running hyperfine: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'hyperfine' command not found. Please install hyperfine first.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Run hyperfine benchmark on files in a folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_script.py nvtiff/seeds './build_vanilla_release/example/nvtiff_example_persistent_base_static "$f"'
  python benchmark_script.py ../nvtiff_seeds './build_vanilla_release/example/nvtiff_example_persistent_base_static "$f"'
  python benchmark_script.py nvtiff/seeds './build_vanilla_release/example/nvtiff_example_persistent_base_static "$f"' -o results.csv
  python benchmark_script.py nvtiff/seeds './build_vanilla_release/example/nvtiff_example_persistent_base_static "$f"' -o results.csv -t vanilla
  python benchmark_script.py nvtiff/seeds './build_vanilla_release/example/nvtiff_example_persistent_base_static "$f"' -o results.csv -t optimized -c
  python benchmark_script.py nvtiff/seeds './build_vanilla_release/example/nvtiff_example_persistent_base_static "$f"' -i
        """
    )
    
    parser.add_argument(
        "input_folder",
        help="Path to the folder containing input files"
    )
    
    parser.add_argument(
        "execution_command",
        help="Command to execute for each file. Use '$f' as placeholder for the file path"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output CSV file to write benchmark results"
    )
    
    parser.add_argument(
        "-t", "--tag",
        help="Tag to add as first column in CSV output"
    )
    
    parser.add_argument(
        "-c", "--concatenate",
        action="store_true",
        help="Concatenate results to existing CSV file instead of overwriting"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Add -i flag to hyperfine for interactive mode"
    )
    
    args = parser.parse_args()
    
    run_benchmark(args.input_folder, args.execution_command, args.output, args.tag, args.concatenate, args.interactive)

if __name__ == "__main__":
    main() 