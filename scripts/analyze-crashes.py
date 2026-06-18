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
analyze-crashes.py - Analyze and categorize AFL crash logs

This script parses crash analysis logs from fuzzing campaigns and extracts
unique crash signatures. It identifies different error types including:
- AddressSanitizer errors (heap-buffer-overflow, SEGV, etc.)
- CUDA Compute Sanitizer errors (invalid memory access, uninitialized memory)
- Race condition errors from racecheck

Usage:
    python3 analyze-crashes.py <LOG_FILE> [options]

Arguments:
    LOG_FILE - Path to the crash analysis log file

Options:
    --unique     - Show only unique crash signatures with their input files
    -o, --output - Output CSV file path for crash data

Output CSV columns:
    signature, filename, execs, time, clean_time

Example:
    python3 analyze-crashes.py crashes.log --unique -o unique_crashes.csv
"""

import sys
import re
import os
import csv
from pathlib import Path
from typing import List, Dict, Any
import argparse
from collections import defaultdict

def dbg(msg):
    if os.environ.get('ENABLE_DBG'):
        print(f'[DEBUG] {msg}')

def parse_log_file(log_file: str) -> List[Dict[str, Any]]:
    """
    Parse the crash analysis log file and extract relevant information.
    Returns a list of dictionaries containing crash information.
    """
    crashes = []
    current_crash = {}
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        
        # Skip empty lines only
        if not line:
            continue
        
        # Start of a new crash entry
        if line.startswith('Processing crash file:') or line.startswith('Processing file:'):
            dbg(f'New crash entry: {line}')
            if current_crash:
                crashes.append(current_crash)
            current_crash = {
                'file': line.replace('Processing crash file:', '').replace('Processing file:', '').strip(),
                'output': []
            }
        # Program output
        elif current_crash and not line.startswith('---'):
            current_crash['output'].append(line)
            dbg(f'Appending line to output: {line}')
    
    # Add the last crash if exists
    if current_crash:
        crashes.append(current_crash)
    
    dbg(f'Total crashes parsed: {len(crashes)}')
    return crashes

def analyze_crashes(crashes: List[Dict[str, Any]]) -> None:
    """
    Analyze the crashes and print relevant information.
    This is a template function - modify it based on your specific analysis needs.
    """
    print(f"Total number of crashes analyzed: {len(crashes)}")
    
    # Example analysis - count crashes per folder
    folder_counts = {}
    for crash in crashes:
        folder = str(Path(crash['file']).parent.parent)
        folder_counts[folder] = folder_counts.get(folder, 0) + 1
    
    print("\nCrashes per folder:")
    for folder, count in folder_counts.items():
        print(f"{folder}: {count} crashes")

def extract_signature(output_lines):
    """
    Extracts a crash signature from the output lines.
    Handles two types of errors:
    1. '========= Invalid' errors with location in next line
    2. AddressSanitizer errors (SEGV, heap-buffer-overflow, etc) with location in same line
    Returns a string that uniquely identifies the crash type/location.
    """
    for i, line in enumerate(output_lines):
        # Handle AddressSanitizer errors
        if "SUMMARY: AddressSanitizer:" in line:
            dbg(f'Found AddressSanitizer line: {line}')
            # Try to match the debug info format first (file:line)
            match = re.search(r'SUMMARY: AddressSanitizer: (\S+) ([^:]+):(\d+):', line)
            dbg(f'AddressSanitizer debug regex match: {match.groups() if match else None}')
            if match:
                error_type, file, line_no = match.groups()
                return f"AddressSanitizer: {error_type} in {file}:{line_no}"
            
            # If no debug info, try to match the non-debug format with function names
            match = re.search(r'SUMMARY: AddressSanitizer: (\S+) \(([^)]+)\) \(BuildId: [^)]+\) in ([^(]+)', line)
            dbg(f'AddressSanitizer non-debug regex match: {match.groups() if match else None}')
            if match:
                error_type, address, function = match.groups()
                return f"AddressSanitizer: {error_type} at {address} in {function}"
            
            # Fallback: extract just the error type if no specific location info
            match = re.search(r'SUMMARY: AddressSanitizer: (\S+)', line)
            dbg(f'AddressSanitizer fallback regex match: {match.groups() if match else None}')
            if match:
                error_type = match.group(1)
                return f"AddressSanitizer: {error_type} (no location info)"

        # Handle '========= Invalid' errors
        elif "========= Invalid" in line or "========= Uninitialized" in line or "Race reported" in line:
            invalid_start = min(
                (line.find(marker) for marker in ["========= Invalid", "========= Uninitialized", "Race reported"] if marker in line),
                default=-1
            )
            error_text = line[invalid_start:].strip()
            dbg(f'Found Invalid error line: {error_text}')
            j = 1
            if i + j < len(output_lines):
                next_line = output_lines[i + j]
                dbg(f'Checking next line for location: {next_line}')
                # Match both cases: with and without file:line
                match = re.search(r'(at .*?)(?:\s+in\s+(\S+):(\d+))?$', next_line)
                dbg(f'Location regex match: {match.groups() if match else None}')
                if match:
                    location_text = match.group(1)  # Get the full "at ..." part
                    file_line = match.group(2)  # File name (if present)
                    line_no = match.group(3)    # Line number (if present)
                    
                    # If we have file and line info, append it
                    if file_line and line_no:
                        signature = f"{error_text} {location_text} in {file_line}:{line_no}"
                    else:
                        signature = f"{error_text} {location_text}"
                    dbg(f'Extracted signature: {signature}')
                    return signature
    dbg('No signature found in output_lines')
    return None

def analyze_unique_crashes(crashes):
    """
    Groups input files by unique crash signatures.
    """
    signature_to_files = defaultdict(list)
    for crash in crashes:
        sig = extract_signature(crash['output'])
        if sig:
            signature_to_files[sig].append(crash['file'])

    print("Unique crash signatures and input files:")
    for sig, files in signature_to_files.items():
        print(f"\nSignature: {sig}")
        for f in files:
            print(f"  - {f}")

def parse_filename_metadata(filename):
    """
    Parse execs and time values from the filename.
    Returns a tuple of (execs, time) or (None, None) if not found.
    """
    # Extract the filename part (last component of the path)
    filename_part = Path(filename).name
    
    # Parse execs value
    execs_match = re.search(r'execs:(\d+)', filename_part)
    execs = execs_match.group(1) if execs_match else None
    
    # Parse time value
    time_match = re.search(r'time:(\d+)', filename_part)
    time = time_match.group(1) if time_match else None
    
    return execs, time

def convert_milliseconds_to_hh_mm_ss(milliseconds_str):
    """
    Convert milliseconds string to hh:mm:ss format.
    Returns the formatted string or empty string if conversion fails.
    """
    if not milliseconds_str:
        return ''
    
    try:
        milliseconds = int(milliseconds_str)
        total_seconds = milliseconds // 1000
        
        hours = total_seconds // 3600
        remaining_seconds = total_seconds % 3600
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except (ValueError, TypeError):
        return ''

def write_csv_output(signature_to_files, output_file):
    """
    Write the unique crash signatures and their corresponding files to a CSV file.
    """
    with open(output_file, 'w', newline='') as csvfile:        
        writer = csv.writer(csvfile)
        writer.writerow(['signature', 'filename', 'execs', 'time', 'clean_time'])
        
        for sig, files in signature_to_files.items():
            for file in files:
                execs, time = parse_filename_metadata(file)
                clean_time = convert_milliseconds_to_hh_mm_ss(time)
                writer.writerow([sig, file, execs or '', time or '', clean_time])
    
    print(f"CSV output written to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze AFL crash logs')
    parser.add_argument('log_file', help='Path to the crash analysis log file')
    parser.add_argument('--unique', action='store_true', help='Show only unique crash signatures and input files')
    parser.add_argument('-o', '--output', help='Output CSV file path')
    args = parser.parse_args()
    
    if not Path(args.log_file).exists():
        print(f"Error: Log file {args.log_file} does not exist")
        sys.exit(1)
    
    crashes = parse_log_file(args.log_file)
    
    # Get unique crash signatures
    signature_to_files = defaultdict(list)
    for crash in crashes:
        sig = extract_signature(crash['output'])
        if sig:
            signature_to_files[sig].append(crash['file'])
    
    if args.output:
        write_csv_output(signature_to_files, args.output)
    
    if args.unique:
        analyze_unique_crashes(crashes)
    else:
        analyze_crashes(crashes)
        analyze_unique_crashes(crashes)

if __name__ == '__main__':
    main() 