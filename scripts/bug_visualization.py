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
bug_visualization.py - Generate bug type distribution visualizations

This script reads a markdown table of bugs and creates pie chart visualizations
showing the distribution of bug types. It categorizes bugs into host-side and
device-side errors and generates publication-quality figures.

Bug categories:
    Host-side:
    - Heap Buffer Overflow
    - Segmentation Fault
    - Floating Point Exception

    Device-side:
    - Invalid Global Memory Read/Write
    - Invalid Shared Memory Read/Write
    - Shared Memory Race
    - Uninitialized Global Memory

Usage:
    python3 bug_visualization.py <INPUT_FILE> [options]

Arguments:
    INPUT_FILE - Path to markdown file containing the bug table

Options:
    -o, --output FILE   - Output file base name (default: bug_type_distribution)
    -p, --project NAME  - Filter bugs by project name (nvTIFF, nvJPEG2000, nvJPEG)

Output:
    Creates both .jpg and .svg versions of the visualization

Example:
    python3 bug_visualization.py bugs.md -o figures/bug_dist -p nvTIFF
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
import argparse
import os
import numpy as np

def debug_print(*args, **kwargs):
    """Print debug messages only if DEBUG environment variable is set"""
    if os.environ.get('DEBUG'):
        print(*args, **kwargs)

def extract_bug_type(description):
    # Clean up escaped underscores
    description = description.replace('\\_\\_', '__')
    
    # Extract the main bug type from the description
    if "Invalid __global__" in description:
        if "read" in description.lower():
            return "Device Invalid Global Memory Read"
        elif "write" in description.lower():
            return "Device Invalid Global Memory Write"
        return "Device Invalid Global Memory"
    elif "Invalid __shared__" in description:
        if "read" in description.lower():
            return "Device Invalid Shared Memory Read"
        elif "write" in description.lower():
            return "Device Invalid Shared Memory Write"
        return "Device Invalid Shared Memory"
    elif "heap-buffer-overflow" in description:
        return "Host Heap Buffer Overflow"
    elif "SEGV" in description:
        return "Host Segmentation Fault"
    elif "memory Race" in description:
        return "Device Shared Memory Race"
    elif "Uninitialized" in description:
        return "Device Uninitialized Global Memory"
    elif "Floating point exception" in description:
        return "Host Floating Point Exception"
    else:
        return "Other"

def parse_markdown_table(markdown_content):
    # Split the content into lines
    lines = markdown_content.strip().split('\n')
    
    # Find the table content
    table_start = False
    table_lines = []
    
    for line in lines:
        if line.startswith('| Bug ID'):
            table_start = True
            continue
        if table_start and '|' in line:
            if line.startswith('| :'):  # Skip separator line
                continue
            if line.startswith('| Total'):  # Skip total row
                continue
            table_lines.append(line)
    
    debug_print("\nTable lines found:")
    for line in table_lines[:3]:  # Print first 3 lines for debugging
        debug_print(line)
    
    # Parse the table
    data = []
    for line in table_lines:
        # Split by | and remove empty strings
        cells = [cell.strip() for cell in line.split('|') if cell.strip()]
        debug_print(f"\nProcessing line: {cells}")
        
        if len(cells) >= 6:  # Ensure we have enough columns
            # Extract bug ID from markdown link if present
            bug_id = cells[0]
            if '[' in bug_id and ']' in bug_id:
                bug_id = re.search(r'\[(.*?)\]', bug_id).group(1)
            
            # The description is in the 5th column (index 4)
            description = cells[4]
            
            row_data = {
                'Bug ID': bug_id,
                'Project': cells[1],
                'Version': cells[2],
                'Description': description,
                'Status': cells[5] if len(cells) > 5 else 'Unknown'
            }
            data.append(row_data)
            debug_print(f"Added row: {row_data}")
    
    if not data:
        debug_print("No data rows were parsed!")
        return pd.DataFrame()
    
    return pd.DataFrame(data)

def is_host_bug(description):
    host_indicators = [
        "heap-buffer-overflow",
        "SEGV",
        "Floating point exception"
    ]
    return any(indicator in description for indicator in host_indicators)

def plot_bug_type_distribution(df, output_file, project_name=None):
    if df.empty:
        print("Error: No data to visualize")
        return
    
    # Filter by project if specified and create a copy
    if project_name:
        df = df[df['Project'] == project_name].copy()
        if df.empty:
            print(f"Error: No data found for project '{project_name}'")
            return
        title = f'Distribution of Bug Types - {project_name}'
    else:
        df = df.copy()  # Create a copy even when not filtering
        title = 'Distribution of Bug Types - All Projects'
        
    # Extract bug types from descriptions
    df['Bug Type'] = df['Description'].apply(extract_bug_type)
    df['Is Host'] = df['Description'].apply(is_host_bug)
    
    # Separate host and device bugs
    host_bugs = df[df['Is Host']]
    device_bugs = df[~df['Is Host']]
    
    # Count bug types for each category
    host_counts = host_bugs['Bug Type'].value_counts()
    device_counts = device_bugs['Bug Type'].value_counts()
    
    # Print debug information
    debug_print("\nHost Bug Distribution:")
    debug_print(host_counts)
    debug_print("\nDevice Bug Distribution:")
    debug_print(device_counts)
    debug_print("\nTotal number of bugs:", len(df))
    
    # Create the pie chart
    plt.figure(figsize=(12, 8))
    
    # Combine the counts with host bugs first, then device bugs
    all_counts = pd.concat([host_counts, device_counts])
    
    # Create color schemes
    host_colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(host_counts)))
    device_colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(device_counts)))
    colors = np.concatenate([host_colors, device_colors])
    
    # Create the pie chart
    wedges, texts, autotexts = plt.pie(all_counts, 
                                      labels=all_counts.index,
                                      autopct=lambda pct: f'{int(pct/100.*len(df))}',
                                      colors=colors,
                                      textprops={'fontsize': 20},
                                      labeldistance=1.2,  # Distance of labels from center
                                      pctdistance=0.85)  # Distance of percentage from center
    
    # Adjust the percentage text size
    plt.setp(autotexts, size=30, weight="bold")
    
    #plt.title(title)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Save in both JPG and SVG formats
    base_name = os.path.splitext(output_file)[0]
    plt.savefig(f"{base_name}.jpg", format='jpg', dpi=300, bbox_inches='tight')
    plt.savefig(f"{base_name}.svg", format='svg', bbox_inches='tight')
    plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate bug type distribution visualization from a markdown table.')
    parser.add_argument('input_file', help='Path to the markdown file containing the bug table')
    parser.add_argument('-o', '--output', default='bug_type_distribution',
                      help='Output file base name without extension (default: bug_type_distribution)')
    parser.add_argument('-p', '--project', help='Filter bugs by project name (e.g., nvTIFF, nvJPEG2000, nvJPEG)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return
    
    try:
        # Read the markdown file
        with open(args.input_file, 'r') as f:
            markdown_content = f.read()
        
        # Parse the table
        df = parse_markdown_table(markdown_content)
        
        if df.empty:
            print("Error: Could not parse any data from the table")
            return
        
        # Print debug information about the parsed data
        debug_print("\nParsed Data Summary:")
        debug_print(f"Number of rows: {len(df)}")
        debug_print("\nFirst few rows:")
        debug_print(df[['Bug ID', 'Project', 'Description']].head())
        
        # Create the visualization
        plot_bug_type_distribution(df, args.output, args.project)
        print(f"Visualization saved as '{args.output}.jpg' and '{args.output}.svg'")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main() 