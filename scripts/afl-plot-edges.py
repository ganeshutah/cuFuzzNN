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
#
# american fuzzy lop++ - Edge Coverage Comparison Plotting
# ------------------------------------------------------
#
# This script plots edge coverage from multiple AFL++ fuzzing sessions
# on the same graph for easy comparison.
#
# Usage: afl-plot-edges.py [-c "config name:color:folder1,folder2,..."] [-x max_hours] [-f format] [-a axis] [-p plots] output_dir

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, is_color_like
from matplotlib.ticker import FuncFormatter
import re

def get_abs_path(path):
    """Convert a path to its absolute form."""
    return str(Path(path).resolve())

def sanitize_filename(name):
    """Sanitize a string for use in filenames."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)

def process_plot_data(file_path, x_axis='time', edge_type='total', include_execs=False):
    """Process edge.csv file and return x-axis and y-axis data."""
    try:
        # Read the edge.csv file, skipping comment lines
        df = pd.read_csv(file_path, sep=',', header=None, skipinitialspace=True, 
                        comment='#')
        
        # Print file being processed
        print(f"\n[*] Processing file: {file_path}")
        
        # Verify we have enough columns
        if len(df.columns) < 7:
            raise ValueError(f"Expected at least 7 columns, got {len(df.columns)}")
            
        # Get x-axis data based on selection
        if x_axis == 'time':
            x_data = df[1].astype(float) / 3600000  # Convert milliseconds to hours
            x_label = "time (hours)"
        else:  # executions
            x_data = df[2].astype(float)  # execs column
            x_label = "total executions"
            print(f"[*] Total executions range: {x_data.min()} to {x_data.max()}")
            
        # Get y-axis data based on edge type
        if edge_type == 'total':
            y_data = df[4].astype(float)  # total_edges column
            y_label = "total edges"
        elif edge_type == 'host':
            y_data = df[5].astype(float)  # host_edges column
            y_label = "host edges"
        elif edge_type == 'device':
            y_data = df[6].astype(float)  # device_edges column
            y_label = "device edges"
        elif edge_type == 'unique':
            # Extract unique input count from filename in first column
            filenames = df[0].astype(str)
            unique_counts = []
            for filename in filenames:
                match = re.search(r'id:(\d+)', filename)
                if match:
                    unique_counts.append(int(match.group(1)) + 1)
                else:
                    print(f"[-] Warning: Could not find id:XYZ pattern in filename: {filename}", file=sys.stderr)
                    unique_counts.append(0)
            y_data = pd.Series(unique_counts)
            y_label = "unique inputs"
            
        print(f"[*] {y_label} range: {y_data.min()} to {y_data.max()}")
        
        # Remove any NaN values
        x_data = x_data.dropna()
        y_data = y_data.dropna()
        
        # Sort by x-axis
        sorted_idx = x_data.argsort()
        x_data = x_data.iloc[sorted_idx]
        y_data = y_data.iloc[sorted_idx]
        
        # Get execution data if requested
        exec_data = None
        if include_execs:
            exec_data = df[2].astype(float).iloc[sorted_idx].dropna()
        
        return x_data, y_data, x_label, y_label, exec_data
    except Exception as e:
        print(f"[-] Error processing {file_path}: {str(e)}", file=sys.stderr)
        sys.exit(1)

def format_execution_count(exec_data):
    """Format execution count for display in legend."""
    # Return empty string to remove execution count from legend
    return ""

def format_executions_in_thousands(x, pos):
    """Format execution count in thousands for secondary y-axis."""
    if x >= 1000:
        return f"{x/1000:.0f}K"
    else:
        return f"{x:.0f}"

def parse_config(config_str):
    """Parse a configuration string into name, color, and folders."""
    try:
        # First split by comma to separate folders
        main_parts, *folder_parts = config_str.split(',')
        
        # Split the first part to get name and color
        name_parts = main_parts.split(':')
        if len(name_parts) < 2:
            raise ValueError("Configuration must contain at least name and folders")
        
        config_name = name_parts[0]
        color = name_parts[1] if len(name_parts) > 1 else None
        
        # Get the first folder (it's in the main_parts)
        first_folder = ':'.join(name_parts[2:]) if len(name_parts) > 2 else None
        
        # Combine all folders
        folders = []
        if first_folder:
            folders.append(first_folder)
        folders.extend(folder_parts)
        
        # Validate color if specified
        if color and not is_color_like(color):
            print(f"[-] Warning: Invalid color '{color}' for config '{config_name}', using default color", file=sys.stderr)
            color = None
            
        # Convert folders to absolute paths
        folders = [get_abs_path(f.strip()) for f in folders]
        
        return config_name, color, folders
    except Exception as e:
        print(f"[-] Error parsing configuration '{config_str}': {str(e)}", file=sys.stderr)
        sys.exit(1)

def find_global_x_max(configs, axis, edge_types):
    """Find the global maximum x-value across all configurations and edge types."""
    global_max = 0
    
    # For combo mode, we use time for x-axis calculations
    x_axis_type = 'time' if axis == 'combo' else axis
    
    for config in configs:
        config_name, color, folders = parse_config(config)
        
        for folder in folders:
            for edge_type in edge_types:
                try:
                    x_data, _, _, _, _ = process_plot_data(Path(folder) / "edge.csv", x_axis_type, edge_type)
                    if len(x_data) > 0:
                        folder_max = x_data.max()
                        global_max = max(global_max, folder_max)
                except Exception as e:
                    print(f"[-] Warning: Could not process {folder}/edge.csv for global max calculation: {str(e)}", file=sys.stderr)
                    continue
    
    return global_max

def extend_dataset_to_max(x_data, y_data, target_max):
    """Extend a dataset to reach the target maximum x-value by repeating the last y-value."""
    if len(x_data) == 0 or len(y_data) == 0:
        return x_data, y_data
    
    current_max = x_data.max()
    
    if current_max >= target_max:
        return x_data, y_data
    
    # Add a new point at the target maximum with the last y-value
    extended_x = pd.concat([x_data, pd.Series([target_max])])
    extended_y = pd.concat([y_data, pd.Series([y_data.iloc[-1]])])
    
    print(f"[*] Extended dataset from {current_max:.2f} to {target_max:.2f} (final value: {y_data.iloc[-1]:.0f})")
    
    return extended_x, extended_y

def collect_plot_data(configs, axis, edge_types, global_x_max=None, no_extend=False):
    """Collect all plot data for CSV export."""
    collected_data = {}
    
    # For combo mode, we use time for x-axis calculations
    x_axis_type = 'time' if axis == 'combo' else axis
    
    for config in configs:
        config_name, color, folders = parse_config(config)
        
        # Initialize data structure for this config
        if config_name not in collected_data:
            collected_data[config_name] = {}
        
        for edge_type in edge_types:
            if edge_type not in collected_data[config_name]:
                collected_data[config_name][edge_type] = {'x': [], 'y': [], 'exec': []}
            
            if len(folders) == 1:
                # Single folder - collect single dataset
                x_data, y_data, _, _, exec_data = process_plot_data(
                    Path(folders[0]) / "edge.csv", 
                    x_axis_type, 
                    edge_type, 
                    include_execs=(axis in ['time', 'combo'])
                )
                
                # Extend dataset if needed
                if global_x_max is not None and not no_extend:
                    x_data, y_data = extend_dataset_to_max(x_data, y_data, global_x_max)
                
                collected_data[config_name][edge_type]['x'] = x_data.tolist()
                collected_data[config_name][edge_type]['y'] = y_data.tolist()
                collected_data[config_name][edge_type]['exec'] = exec_data.tolist() if exec_data is not None else []
                
            else:
                # Multiple folders - collect all datasets for interpolation
                all_x_data = []
                all_y_data = []
                all_exec_data = []
                
                for folder in folders:
                    x_data, y_data, _, _, exec_data = process_plot_data(
                        Path(folder) / "edge.csv", 
                        x_axis_type, 
                        edge_type,
                        include_execs=(axis in ['time', 'combo'])
                    )
                    
                    # Extend dataset if needed
                    if global_x_max is not None and not no_extend:
                        x_data, y_data = extend_dataset_to_max(x_data, y_data, global_x_max)
                    
                    all_x_data.append(x_data)
                    all_y_data.append(y_data)
                    all_exec_data.append(exec_data)
                
                # Find the full x-axis range
                min_x = min(x.min() for x in all_x_data)
                max_x = max(x.max() for x in all_x_data)
                
                # Create interpolated x-axis points
                interp_x = np.linspace(min_x, max_x, 1000)
                
                # Interpolate y values for each folder
                interp_y = []
                for x_data, y_data in zip(all_x_data, all_y_data):
                    # Use linear interpolation
                    interp_y.append(np.interp(interp_x, x_data, y_data, left=y_data.iloc[0], right=y_data.iloc[-1]))
                
                # Calculate min and max y values
                min_y = np.min(interp_y, axis=0)
                max_y = np.max(interp_y, axis=0)
                
                # Store interpolated data
                collected_data[config_name][edge_type]['x'] = interp_x.tolist()
                collected_data[config_name][edge_type]['y_min'] = min_y.tolist()
                collected_data[config_name][edge_type]['y_max'] = max_y.tolist()
                
                # Handle execution data for combo mode
                if axis == 'combo':
                    interp_exec = []
                    for x_data, exec_data in zip(all_x_data, all_exec_data):
                        if exec_data is not None and len(exec_data) > 0:
                            # Extend execution data to match x_data if needed
                            if len(exec_data) < len(x_data):
                                last_exec = exec_data.iloc[-1]
                                padding = pd.Series([last_exec] * (len(x_data) - len(exec_data)))
                                extended_exec_data = pd.concat([exec_data, padding])
                            else:
                                extended_exec_data = exec_data.iloc[:len(x_data)]
                            
                            interp_exec.append(np.interp(interp_x, x_data, extended_exec_data, 
                                                       left=extended_exec_data.iloc[0], 
                                                       right=extended_exec_data.iloc[-1]))
                    
                    if interp_exec:
                        # Calculate min and max execution values
                        min_exec = np.min(interp_exec, axis=0)
                        max_exec = np.max(interp_exec, axis=0)
                        
                        collected_data[config_name][edge_type]['exec_min'] = min_exec.tolist()
                        collected_data[config_name][edge_type]['exec_max'] = max_exec.tolist()
    
    return collected_data

def export_to_csv(collected_data, output_file, axis, edge_types):
    """Export collected data to CSV format for Excel import."""
    try:
        # Create a list to store all rows
        rows = []
        
        # Create header row
        header = ['x_axis']
        for config_name in collected_data.keys():
            for edge_type in edge_types:
                if edge_type in collected_data[config_name]:
                    if 'y_min' in collected_data[config_name][edge_type]:
                        # Multiple folders case
                        header.extend([
                            f'{config_name}_{edge_type}_y_min',
                            f'{config_name}_{edge_type}_y_max'
                        ])
                        if axis == 'combo' and 'exec_min' in collected_data[config_name][edge_type]:
                            header.extend([
                                f'{config_name}_{edge_type}_exec_min',
                                f'{config_name}_{edge_type}_exec_max'
                            ])
                    else:
                        # Single folder case
                        header.extend([
                            f'{config_name}_{edge_type}_y'
                        ])
                        if axis == 'combo' and collected_data[config_name][edge_type]['exec']:
                            header.append(f'{config_name}_{edge_type}_exec')
        
        rows.append(header)
        
        # Find the maximum length of any dataset
        max_length = 0
        for config_name in collected_data.keys():
            for edge_type in edge_types:
                if edge_type in collected_data[config_name]:
                    data = collected_data[config_name][edge_type]
                    if 'x' in data:
                        max_length = max(max_length, len(data['x']))
        
        # Create data rows
        for i in range(max_length):
            row = []
            
            # Add x-axis value (use the first available x-axis data)
            x_value = None
            for config_name in collected_data.keys():
                for edge_type in edge_types:
                    if edge_type in collected_data[config_name] and 'x' in collected_data[config_name][edge_type]:
                        if i < len(collected_data[config_name][edge_type]['x']):
                            x_value = collected_data[config_name][edge_type]['x'][i]
                            break
                if x_value is not None:
                    break
            
            row.append(f"{x_value:.6f}" if x_value is not None else "")
            
            # Add data for each config and edge type
            for config_name in collected_data.keys():
                for edge_type in edge_types:
                    if edge_type in collected_data[config_name]:
                        data = collected_data[config_name][edge_type]
                        
                        if 'y_min' in data:
                            # Multiple folders case
                            y_min = data['y_min'][i] if i < len(data['y_min']) else ""
                            y_max = data['y_max'][i] if i < len(data['y_max']) else ""
                            row.extend([f"{y_min:.6f}" if y_min != "" else "", f"{y_max:.6f}" if y_max != "" else ""])
                            
                            if axis == 'combo' and 'exec_min' in data:
                                exec_min = data['exec_min'][i] if i < len(data['exec_min']) else ""
                                exec_max = data['exec_max'][i] if i < len(data['exec_max']) else ""
                                row.extend([f"{exec_min:.0f}" if exec_min != "" else "", f"{exec_max:.0f}" if exec_max != "" else ""])
                        else:
                            # Single folder case
                            y_value = data['y'][i] if i < len(data['y']) else ""
                            row.append(f"{y_value:.6f}" if y_value != "" else "")
                            
                            if axis == 'combo' and data['exec'] and i < len(data['exec']):
                                exec_value = data['exec'][i]
                                row.append(f"{exec_value:.0f}" if exec_value != "" else "")
            
            rows.append(row)
        
        # Write to CSV file
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(rows)
        
        print(f"[+] Data exported to CSV: {output_file}")
        
    except Exception as e:
        print(f"[-] Error exporting data to CSV: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Generate plots comparing different edge coverage metrics from multiple afl-fuzz sessions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    %(prog)s -c "AFL++:red:fuzz1,fuzz2,fuzz3" -c "AFL++ ASAN:blue:fuzz4,fuzz5" -f pdf -a executions -p unique,total output_dir

Configuration format:
    "name:color:folder1,folder2,..."
    - name: Configuration name
    - color: Color name (e.g., red, blue) or hex code (e.g., #ff0000)
    - folders: Comma-separated list of folders

Available plot types:
    - unique: Number of unique inputs
    - total: Total edge coverage
    - host: Host edge coverage
    - device: Device edge coverage

Dataset extension:
    By default, shorter datasets are extended to match the longest dataset's end point
    to facilitate comparison. Use --no-extend to disable this feature.

Execution annotations:
    Execution counts are no longer displayed in the legend labels. The --no-exec-annotations
    option is kept for backward compatibility but has no effect.

Legend:
    Use --no-legend to remove the legend from the plots.

Combo mode:
    When using combo mode (-a combo), the chart shows time on the x-axis with edge coverage
    on the primary y-axis and execution count on the secondary y-axis. Execution lines use
    the same colors as edge coverage lines but with dotted style.

CSV export:
    Use --dump filename.csv to export the plot data to a CSV file for Excel import.
    The CSV will contain columns for each configuration and edge type combination,
    with separate columns for min/max values when multiple folders are used.

The program will create edges_{axis}.{format} in the output directory.
"""
    )
    
    parser.add_argument(
        "-c", "--config",
        action="append",
        help="configuration string (format: 'name:color:folder1,folder2,...')",
        required=True
    )
    parser.add_argument(
        "-x", "--xmax",
        type=float,
        help="maximum value for x-axis"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["png", "jpg", "pdf"],
        default="png",
        help="output format (default: png)"
    )
    parser.add_argument(
        "-a", "--axis",
        choices=["time", "executions", "combo"],
        default="time",
        help="x-axis type (default: time). 'combo' shows time on x-axis with both edge coverage and execution count on dual y-axes"
    )
    parser.add_argument(
        "-p", "--plots",
        default="unique,total,host,device",
        help="comma-separated list of plots to include (default: all plots)"
    )
    parser.add_argument(
        "output_dir",
        help="directory where the resulting plot will be saved"
    )
    parser.add_argument(
        "--no-extend",
        action="store_true",
        help="disable extending shorter datasets to match the longest one"
    )
    parser.add_argument(
        "--no-exec-annotations",
        action="store_true",
        help="kept for backward compatibility but has no effect (execution counts no longer shown in legend)"
    )
    parser.add_argument(
        "--width",
        type=float,
        default=10.0,
        help="width of the output image in inches (default: 10.0)"
    )
    parser.add_argument(
        "--dump",
        type=str,
        help="dump plot data to CSV file for Excel import"
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="remove legend from the plots"
    )
    
    args = parser.parse_args()
    
    # Validate and process plot types
    available_plots = ['unique', 'total', 'host', 'device']
    requested_plots = [p.strip() for p in args.plots.split(',')]
    invalid_plots = [p for p in requested_plots if p not in available_plots]
    
    if invalid_plots:
        print(f"[-] Error: Invalid plot types: {', '.join(invalid_plots)}", file=sys.stderr)
        print(f"[-] Available plot types: {', '.join(available_plots)}", file=sys.stderr)
        sys.exit(1)
    
    if not requested_plots:
        print("[-] Error: At least one plot type must be specified", file=sys.stderr)
        sys.exit(1)
    
    # Find global maximum x-value if extension is enabled
    global_x_max = None
    if not args.no_extend:
        print("[*] Finding global maximum x-value across all configurations...")
        global_x_max = find_global_x_max(args.config, args.axis, requested_plots)
        print(f"[*] Global maximum {args.axis}: {global_x_max}")
    
    # Create output directory
    output_dir = Path(get_abs_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        
        # Create figure with requested number of subplots
        n_plots = len(requested_plots)
        fig, axes = plt.subplots(n_plots, 1, figsize=(args.width, 3 * n_plots + 1))  # Adjust height based on number of plots
        if n_plots == 1:
            axes = [axes]  # Make axes iterable for single plot case
        
        # For combo mode, create secondary y-axes
        if args.axis == 'combo':
            for ax in axes:
                ax2 = ax.twinx()
                # Store the secondary axis for later use
                ax._secondary_axis = ax2
        
        # Default colors if not specified
        default_colors = [
            '#0090ff', '#c00080', '#c000f0', '#00c000',
            '#c0c000', '#c00000', '#00c0c0', '#c000c0'
        ]
        
        # Process each configuration
        for config_idx, config in enumerate(args.config):
            config_name, color, folders = parse_config(config)
            
            # Use default color if none specified
            if color is None:
                color = default_colors[config_idx % len(default_colors)]
            
            print(f"[*] Processing configuration: {config_name}")
            print(f"[*] Using color: {color}")
            print(f"[*] Folders: {folders}")
            
            # Store ranges for debug output
            config_ranges = []
            
            # Plot for each edge type
            for ax, edge_type in zip(axes, requested_plots):
                if len(folders) == 1:
                    # Single folder - plot a line
                    x_data, y_data, _, y_label, exec_data = process_plot_data(
                        Path(folders[0]) / "edge.csv", 
                        'time' if args.axis == 'combo' else args.axis, 
                        edge_type, 
                        include_execs=(args.axis in ['time', 'combo'])
                    )
                    
                    # Extend dataset if needed
                    if global_x_max is not None:
                        x_data, y_data = extend_dataset_to_max(x_data, y_data, global_x_max)
                    
                    # Store range for debug output
                    config_ranges.extend([int(y_data.min()), int(y_data.max())])
                    
                    # Plot edge coverage data
                    ax.plot(x_data, y_data, label=config_name + format_execution_count(exec_data), color=color, linewidth=4)
                    
                    # For combo mode, also plot execution data on secondary y-axis
                    if args.axis == 'combo' and exec_data is not None and len(exec_data) > 0:
                        # Extend execution data to match x_data if needed
                        if len(exec_data) < len(x_data):
                            # Pad execution data with the last value
                            last_exec = exec_data.iloc[-1]
                            padding = pd.Series([last_exec] * (len(x_data) - len(exec_data)))
                            extended_exec_data = pd.concat([exec_data, padding])
                        else:
                            extended_exec_data = exec_data.iloc[:len(x_data)]
                        
                        ax._secondary_axis.plot(x_data, extended_exec_data, color=color, linestyle='--', linewidth=2, alpha=0.7)
                    
                else:
                    # Multiple folders - plot filled region
                    all_x_data = []
                    all_y_data = []
                    all_exec_data = []
                    
                    for folder in folders:
                        x_data, y_data, _, y_label, exec_data = process_plot_data(
                            Path(folder) / "edge.csv", 
                            'time' if args.axis == 'combo' else args.axis, 
                            edge_type,
                            include_execs=(args.axis in ['time', 'combo'])
                        )
                        
                        # Extend dataset if needed
                        if global_x_max is not None:
                            x_data, y_data = extend_dataset_to_max(x_data, y_data, global_x_max)
                        
                        all_x_data.append(x_data)
                        all_y_data.append(y_data)
                        all_exec_data.append(exec_data)
                    
                    # Find the full x-axis range
                    min_x = min(x.min() for x in all_x_data)
                    max_x = max(x.max() for x in all_x_data)
                    
                    # Create interpolated x-axis points
                    interp_x = np.linspace(min_x, max_x, 1000)
                    
                    # Interpolate y values for each folder
                    interp_y = []
                    for x_data, y_data in zip(all_x_data, all_y_data):
                        # Use linear interpolation
                        interp_y.append(np.interp(interp_x, x_data, y_data, left=y_data.iloc[0], right=y_data.iloc[-1]))
                    
                    # Calculate min and max y values
                    min_y = np.min(interp_y, axis=0)
                    max_y = np.max(interp_y, axis=0)
                    
                    # Store range for debug output
                    config_ranges.extend([int(min_y.min()), int(max_y.max())])
                    
                    # Plot filled region
                    # Legend text is just the configuration name (no execution count)
                    legend_exec_text = ""
                    
                    ax.fill_between(interp_x, min_y, max_y,
                                  label=config_name + legend_exec_text, color=color, alpha=0.2)
                    ax.plot(interp_x, min_y, color=color, linewidth=1, alpha=0.5)
                    ax.plot(interp_x, max_y, color=color, linewidth=1, alpha=0.5)
                    
                    # For combo mode, also plot execution data on secondary y-axis
                    if args.axis == 'combo':
                        # Interpolate execution data for each folder
                        interp_exec = []
                        for x_data, exec_data in zip(all_x_data, all_exec_data):
                            if exec_data is not None and len(exec_data) > 0:
                                # Extend execution data to match x_data if needed
                                if len(exec_data) < len(x_data):
                                    last_exec = exec_data.iloc[-1]
                                    padding = pd.Series([last_exec] * (len(x_data) - len(exec_data)))
                                    extended_exec_data = pd.concat([exec_data, padding])
                                else:
                                    extended_exec_data = exec_data.iloc[:len(x_data)]
                                
                                interp_exec.append(np.interp(interp_x, x_data, extended_exec_data, 
                                                           left=extended_exec_data.iloc[0], 
                                                           right=extended_exec_data.iloc[-1]))
                        
                        if interp_exec:
                            # Calculate min and max execution values
                            min_exec = np.min(interp_exec, axis=0)
                            max_exec = np.max(interp_exec, axis=0)
                            
                            # Plot execution data as dotted lines
                            ax._secondary_axis.plot(interp_x, min_exec, color=color, linestyle='--', linewidth=1, alpha=0.5)
                            ax._secondary_axis.plot(interp_x, max_exec, color=color, linestyle='--', linewidth=1, alpha=0.5)
                
                # Configure subplot
                ax.grid(True, linestyle='-', alpha=0.3)
                ax.set_xlabel("total executions" if args.axis == "executions" else "time (hours)", fontsize=16, weight='bold')
                ax.set_ylabel(y_label, fontsize=16, weight='bold')
                
                # For combo mode, configure secondary y-axis
                if args.axis == 'combo':
                    ax._secondary_axis.set_ylabel("total executions", fontsize=16, color='gray', weight='bold')
                    ax._secondary_axis.tick_params(axis='y', which='major', labelsize=14, colors='gray')
                    # Make secondary axis tick labels bold
                    for label in ax._secondary_axis.get_yticklabels():
                        label.set_weight('bold')
                    # Set secondary y-axis color to gray
                    ax._secondary_axis.spines['right'].set_color('gray')
                    ax._secondary_axis.yaxis.label.set_color('gray')
                    # Apply custom formatter to show executions in thousands
                    ax._secondary_axis.yaxis.set_major_formatter(FuncFormatter(format_executions_in_thousands))
                
                # Increase tick label font size and make bold
                ax.tick_params(axis='both', which='major', labelsize=14)
                # Make tick labels bold
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_weight('bold')
                
                if args.xmax:
                    ax.set_xlim(0, args.xmax)
                    if args.axis == 'combo':
                        ax._secondary_axis.set_xlim(0, args.xmax)
                
                # Add legend only to the first subplot (unless --no-legend is specified)
                if ax == axes[0] and not args.no_legend:
                    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, fontsize=14)
            
            # Print configuration summary with ranges
            ranges_str = ", ".join(map(str, config_ranges))
            print(f"{config_name}, {color}, {ranges_str}")
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        output_file = output_dir / f"edges_{args.axis}.{args.format}"
        try:
            if args.format == 'jpg':
                plt.savefig(str(output_file), format=args.format, dpi=300, bbox_inches='tight')
            else:
                plt.savefig(str(output_file), format=args.format, dpi=300, bbox_inches='tight', 
                          metadata={'Creator': 'AFL++ Edge Plotter'}, 
                          encoding='utf-8')
            print(f"[+] Plot generated successfully: {output_file}")
        except Exception as e:
            print(f"[-] Error saving plot to {output_file}: {str(e)}", file=sys.stderr)
            sys.exit(1)
        finally:
            plt.close(fig)  # Ensure the figure is closed properly
        
        # Export data to CSV if requested
        if args.dump:
            print("[*] Collecting data for CSV export...")
            collected_data = collect_plot_data(args.config, args.axis, requested_plots, global_x_max, args.no_extend)
            
            # Determine CSV output path
            if os.path.isabs(args.dump):
                csv_output_file = args.dump
            else:
                csv_output_file = output_dir / args.dump
            
            export_to_csv(collected_data, csv_output_file, args.axis, requested_plots)

if __name__ == "__main__":
    main() 