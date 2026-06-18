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
report-ablation.py - Pivot and analyze performance ablation results

This script processes benchmark CSV data to create pivoted reports comparing
execution times across different configurations. It computes normalized
overhead relative to vanilla (uninstrumented) execution and calculates
geometric means for summary statistics.

Configurations compared:
    - vanilla_1   : Uninstrumented baseline
    - afl_1       : AFL++ instrumentation only
    - cufuzz_1    : cuFuzz with device coverage
    - asan_1      : AddressSanitizer
    - memcheck_1  : CUDA memcheck
    - racecheck_1 : CUDA racecheck
    - initcheck_1 : CUDA initcheck

Usage:
    python3 report-ablation.py --input <CSV> --app <NAME> --output <CSV> [options]

Options:
    --input FILE           - Input CSV with benchmark data
    --app NAME             - Application name for output
    --edge-info FILE       - Optional edge coverage CSV
    --output FILE          - Output pivoted CSV
    --output-throughput F  - Optional throughput CSV output

Output:
    Pivoted CSV with columns for each configuration's timing statistics,
    normalized means, and geometric mean summaries

Example:
    python3 report-ablation.py --input bench.csv --app nvtiff --output report.csv
"""

import argparse
import csv
import os
import math
from typing import Dict, List, Tuple, Optional


ORDERED_TAGS: List[str] = [
    "vanilla_1",
    "afl_1",
    "cufuzz_1",
    "asan_1",
    "memcheck_1",
    "racecheck_1",
    "initcheck_1",
]

INPUT_HEADER_FIELDS: Tuple[str, str, str, str, str, str, str] = (
    "Tag",
    "Filename",
    "Size (bytes)",
    "Mean (ms)",
    "Std (ms)",
    "Min (ms)",
    "Max (ms)",
)

OUTPUT_COMMON_FIELDS: Tuple[str, str] = (
    "Filename",
    "Size (bytes)",
)

METRICS: Tuple[str, str, str, str] = (
    "Mean",
    "Std",
    "Min",
    "Max",
)

EDGE_INFO_COLUMNS: Tuple[str, str, str, str, str, str, str] = (
    "new_findings",
    "total_edges",
    "host_edges",
    "device_edges",
    "abs_total_edges",
    "abs_host_edges",
    "abs_device_edges",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pivot ablation CSV results by Tag into a single CSV per Filename,"
            " ordered by tags: " + ", ".join(ORDERED_TAGS)
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV with columns: " + ", ".join(INPUT_HEADER_FIELDS),
    )
    parser.add_argument(
        "--app",
        required=True,
        help="Application name to include in the output 'App' column",
    )
    parser.add_argument(
        "--edge-info",
        required=False,
        help=(
            "Optional CSV with columns: #testcase,time,execs,"
            "new_findings,total_edges,host_edges,device_edges,"
            "abs_total_edges,abs_host_edges,abs_device_edges"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV to write the pivoted results",
    )
    parser.add_argument(
        "--output-throughput",
        required=False,
        help=(
            "Optional path to output a second CSV with throughput columns "
            "(throughput_<variant> = 1000/Mean, and throughput_<variant>_d only "
            "for rows with abs_device_edges > 0; else 0)."
        ),
    )
    return parser.parse_args()


def validate_input_header(fieldnames: Optional[List[str]]) -> None:
    if fieldnames is None:
        raise ValueError("Input CSV appears to have no header row.")
    missing = [f for f in INPUT_HEADER_FIELDS if f not in fieldnames]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing)
        )


def build_output_header() -> List[str]:
    header: List[str] = ["App"]
    header.extend(list(OUTPUT_COMMON_FIELDS))
    header.extend(list(EDGE_INFO_COLUMNS))
    for tag in ORDERED_TAGS:
        for metric in METRICS:
            header.append(f"{metric}_{tag} (ms)")
    # Normalized mean columns (relative to vanilla)
    for tag in ORDERED_TAGS:
        if tag == "vanilla_1":
            continue
        label = tag.split("_")[0]
        header.append(f"Normalized_Mean_{label}")
    # Normalized device mean columns (only if abs_device_edges > 0, else 0)
    for tag in ORDERED_TAGS:
        if tag == "vanilla_1":
            continue
        label = tag.split("_")[0]
        header.append(f"Normalized_Device_Mean_{label}")
    return header


def build_output_header_throughput() -> List[str]:
    header: List[str] = ["App"]
    header.extend(list(OUTPUT_COMMON_FIELDS))
    header.extend(list(EDGE_INFO_COLUMNS))
    for tag in ORDERED_TAGS:
        for metric in METRICS:
            header.append(f"{metric}_{tag} (ms)")
    # Throughput columns for all variants (including vanilla)
    for tag in ORDERED_TAGS:
        label = tag.split("_")[0]
        header.append(f"throughput_{label}")
    for tag in ORDERED_TAGS:
        label = tag.split("_")[0]
        header.append(f"throughput_{label}_d")
    return header


def read_input_csv(input_path: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, Dict[str, str]]]]:
    """
    Returns:
        filename_to_size: map of Filename -> Size (bytes) as string (first seen)
        filename_to_tag_to_metrics: map of Filename -> Tag -> metric_name -> value (as string)
            where metric_name is one of: "Mean (ms)", "Std (ms)", "Min (ms)", "Max (ms)"
    """
    filename_to_size: Dict[str, str] = {}
    filename_to_tag_to_metrics: Dict[str, Dict[str, Dict[str, str]]] = {}

    with open(input_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        validate_input_header(reader.fieldnames)

        for row in reader:
            tag = row.get("Tag", "").strip()
            filename = row.get("Filename", "").strip()
            size = row.get("Size (bytes)", "").strip()

            if not filename:
                # Skip malformed rows without a filename
                continue

            if filename not in filename_to_size and size:
                filename_to_size[filename] = size

            # Collect metrics as provided (keep as strings to preserve formatting)
            metrics_map = {
                "Mean (ms)": row.get("Mean (ms)", "").strip(),
                "Std (ms)": row.get("Std (ms)", "").strip(),
                "Min (ms)": row.get("Min (ms)", "").strip(),
                "Max (ms)": row.get("Max (ms)", "").strip(),
            }

            if filename not in filename_to_tag_to_metrics:
                filename_to_tag_to_metrics[filename] = {}
            filename_to_tag_to_metrics[filename][tag] = metrics_map

    return filename_to_size, filename_to_tag_to_metrics


def normalize_testcase_name(value: str) -> str:
    v = value.strip()
    # Strip any surrounding quotes already handled by csv, but ensure trimming
    # Remove leading ./ if present
    if v.startswith("./"):
        v = v[2:]
    # If path components exist, take the last component
    if "/" in v:
        v = v.split("/")[-1]
    # If it contains an 'id:' segment, return from that segment
    idx = v.find("id:")
    if idx != -1:
        v = v[idx:]
    return v


def read_edge_info_csv(edge_info_path: str) -> Dict[str, Dict[str, str]]:
    """
    Returns mapping: normalized_testcase -> edge info columns as strings
    """
    mapping: Dict[str, Dict[str, str]] = {}
    with open(edge_info_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        testcase_key = None
        for name in fieldnames:
            if name.strip() in ("#testcase", "testcase"):
                testcase_key = name
                break
        if testcase_key is None:
            raise ValueError("Edge info CSV missing '#testcase' column")

        missing = [c for c in EDGE_INFO_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(
                "Edge info CSV missing required columns: " + ", ".join(missing)
            )

        for row in reader:
            testcase_raw = row.get(testcase_key, "").strip()
            if not testcase_raw:
                continue
            key = normalize_testcase_name(testcase_raw)
            mapping[key] = {c: (row.get(c, "").strip()) for c in EDGE_INFO_COLUMNS}
    return mapping


def write_output_csv(
    output_path: str,
    filename_to_size: Dict[str, str],
    filename_to_tag_to_metrics: Dict[str, Dict[str, Dict[str, str]]],
    app_name: str,
    edge_info_by_testcase: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    header = build_output_header()
    # Deterministic row order by filename
    filenames: List[str] = sorted(filename_to_tag_to_metrics.keys())

    # Collect normalized values per config for geomean
    normalized_values_by_label: Dict[str, List[float]] = {}
    normalized_device_values_by_label: Dict[str, List[float]] = {}
    for tag in ORDERED_TAGS:
        if tag == "vanilla_1":
            continue
        label = tag.split("_")[0]
        normalized_values_by_label[label] = []
        normalized_device_values_by_label[label] = []

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for filename in filenames:
            row_out: Dict[str, str] = {
                "App": app_name,
                "Filename": filename,
                "Size (bytes)": filename_to_size.get(filename, ""),
            }

            # Populate edge info columns if available
            for c in EDGE_INFO_COLUMNS:
                row_out[c] = ""
            if edge_info_by_testcase is not None:
                normalized_name = normalize_testcase_name(filename)
                edge_row = edge_info_by_testcase.get(normalized_name)
                if edge_row is None:
                    # Try exact match without normalization as a fallback
                    edge_row = edge_info_by_testcase.get(filename)
                if edge_row is not None:
                    for c in EDGE_INFO_COLUMNS:
                        row_out[c] = edge_row.get(c, "")

            tag_to_metrics = filename_to_tag_to_metrics.get(filename, {})

            # Determine baseline (vanilla) mean
            baseline_mean_str = tag_to_metrics.get("vanilla_1", {}).get("Mean (ms)", "")
            try:
                baseline_mean = float(baseline_mean_str) if baseline_mean_str != "" else None
            except ValueError:
                baseline_mean = None

            for tag in ORDERED_TAGS:
                metrics_for_tag = tag_to_metrics.get(tag, None)
                if metrics_for_tag is None:
                    # Leave cells empty for missing tags
                    for metric in METRICS:
                        row_out[f"{metric}_{tag} (ms)"] = ""
                else:
                    row_out[f"Mean_{tag} (ms)"] = metrics_for_tag.get("Mean (ms)", "")
                    row_out[f"Std_{tag} (ms)"] = metrics_for_tag.get("Std (ms)", "")
                    row_out[f"Min_{tag} (ms)"] = metrics_for_tag.get("Min (ms)", "")
                    row_out[f"Max_{tag} (ms)"] = metrics_for_tag.get("Max (ms)", "")

            # Compute normalized mean columns relative to vanilla
            for tag in ORDERED_TAGS:
                if tag == "vanilla_1":
                    continue
                label = tag.split("_")[0]
                col_name = f"Normalized_Mean_{label}"
                mean_str = tag_to_metrics.get(tag, {}).get("Mean (ms)", "")
                value_str: str = ""
                if baseline_mean is not None and baseline_mean > 0 and mean_str != "":
                    try:
                        mean_val = float(mean_str)
                        ratio = mean_val / baseline_mean
                        value_str = f"{ratio:.6f}"
                        normalized_values_by_label[label].append(ratio)
                    except ValueError:
                        pass
                row_out[col_name] = value_str

            # Compute normalized device mean columns; if abs_device_edges == 0 then 0
            abs_device_edges_str = row_out.get("abs_device_edges", "")
            abs_device_edges_val = 0
            try:
                abs_device_edges_val = int(float(abs_device_edges_str)) if abs_device_edges_str != "" else 0
            except ValueError:
                abs_device_edges_val = 0

            for tag in ORDERED_TAGS:
                if tag == "vanilla_1":
                    continue
                label = tag.split("_")[0]
                col_name_d = f"Normalized_Device_Mean_{label}"
                if abs_device_edges_val <= 0:
                    row_out[col_name_d] = "0"
                    continue
                mean_str = tag_to_metrics.get(tag, {}).get("Mean (ms)", "")
                value_str_d: str = ""
                if baseline_mean is not None and baseline_mean > 0 and mean_str != "":
                    try:
                        mean_val = float(mean_str)
                        ratio = mean_val / baseline_mean
                        value_str_d = f"{ratio:.6f}"
                        if ratio > 0:
                            normalized_device_values_by_label[label].append(ratio)
                    except ValueError:
                        pass
                row_out[col_name_d] = value_str_d if value_str_d != "" else ""

            writer.writerow(row_out)

    # Compute geomean for each label
    geomean_by_label: Dict[str, float] = {}
    for label, values in normalized_values_by_label.items():
        if not values:
            continue
        # geometric mean = exp(average(log(x)))
        sum_logs = 0.0
        count = 0
        for v in values:
            if v > 0:
                sum_logs += math.log(v)
                count += 1
        if count > 0:
            geomean_by_label[label] = math.exp(sum_logs / count)

    geomean_device_by_label: Dict[str, float] = {}
    for label, values in normalized_device_values_by_label.items():
        if not values:
            continue
        sum_logs = 0.0
        count = 0
        for v in values:
            if v > 0:
                sum_logs += math.log(v)
                count += 1
        if count > 0:
            geomean_device_by_label[label] = math.exp(sum_logs / count)

    return geomean_by_label, geomean_device_by_label


def write_output_csv_throughput(
    output_path: str,
    filename_to_size: Dict[str, str],
    filename_to_tag_to_metrics: Dict[str, Dict[str, Dict[str, str]]],
    app_name: str,
    edge_info_by_testcase: Optional[Dict[str, Dict[str, str]]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    header = build_output_header_throughput()
    filenames: List[str] = sorted(filename_to_tag_to_metrics.keys())

    # Collect throughput values per config for geomean (include vanilla for summary)
    geomean_labels = [t.split("_")[0] for t in ORDERED_TAGS]
    throughput_values_by_label: Dict[str, List[float]] = {l: [] for l in geomean_labels}
    throughput_device_values_by_label: Dict[str, List[float]] = {l: [] for l in geomean_labels}

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for filename in filenames:
            row_out: Dict[str, str] = {
                "App": app_name,
                "Filename": filename,
                "Size (bytes)": filename_to_size.get(filename, ""),
            }

            # Edge info
            for c in EDGE_INFO_COLUMNS:
                row_out[c] = ""
            if edge_info_by_testcase is not None:
                normalized_name = normalize_testcase_name(filename)
                edge_row = edge_info_by_testcase.get(normalized_name) or edge_info_by_testcase.get(filename)
                if edge_row is not None:
                    for c in EDGE_INFO_COLUMNS:
                        row_out[c] = edge_row.get(c, "")

            # Metrics per tag
            tag_to_metrics = filename_to_tag_to_metrics.get(filename, {})
            for tag in ORDERED_TAGS:
                metrics_for_tag = tag_to_metrics.get(tag, None)
                if metrics_for_tag is None:
                    for metric in METRICS:
                        row_out[f"{metric}_{tag} (ms)"] = ""
                else:
                    row_out[f"Mean_{tag} (ms)"] = metrics_for_tag.get("Mean (ms)", "")
                    row_out[f"Std_{tag} (ms)"] = metrics_for_tag.get("Std (ms)", "")
                    row_out[f"Min_{tag} (ms)"] = metrics_for_tag.get("Min (ms)", "")
                    row_out[f"Max_{tag} (ms)"] = metrics_for_tag.get("Max (ms)", "")

            # Throughput columns: 1000 / Mean(tag)
            for tag in ORDERED_TAGS:
                label = tag.split("_")[0]
                mean_str = tag_to_metrics.get(tag, {}).get("Mean (ms)", "")
                value_str: str = ""
                try:
                    mean_val = float(mean_str) if mean_str != "" else None
                except ValueError:
                    mean_val = None
                if mean_val is not None and mean_val > 0:
                    tp = 1000.0 / mean_val
                    value_str = f"{tp:.6f}"
                    if label in throughput_values_by_label:
                        throughput_values_by_label[label].append(tp)
                row_out[f"throughput_{label}"] = value_str

            # Device throughput columns: only when abs_device_edges > 0, else 0
            abs_device_edges_str = row_out.get("abs_device_edges", "")
            try:
                abs_device_edges_val = int(float(abs_device_edges_str)) if abs_device_edges_str != "" else 0
            except ValueError:
                abs_device_edges_val = 0
            for tag in ORDERED_TAGS:
                label = tag.split("_")[0]
                col_name = f"throughput_{label}_d"
                if abs_device_edges_val <= 0:
                    row_out[col_name] = "0"
                    continue
                mean_str = tag_to_metrics.get(tag, {}).get("Mean (ms)", "")
                value_str_d: str = ""
                try:
                    mean_val = float(mean_str) if mean_str != "" else None
                except ValueError:
                    mean_val = None
                if mean_val is not None and mean_val > 0:
                    tp = 1000.0 / mean_val
                    value_str_d = f"{tp:.6f}"
                    if label in throughput_device_values_by_label:
                        throughput_device_values_by_label[label].append(tp)
                row_out[col_name] = value_str_d if value_str_d != "" else ""

            writer.writerow(row_out)

    # Geomean throughput per config (including vanilla in the summary)
    geomean_tp_by_label: Dict[str, float] = {}
    for label, values in throughput_values_by_label.items():
        if not values:
            continue
        sum_logs = 0.0
        count = 0
        for v in values:
            if v > 0:
                sum_logs += math.log(v)
                count += 1
        if count > 0:
            geomean_tp_by_label[label] = math.exp(sum_logs / count)

    geomean_tp_device_by_label: Dict[str, float] = {}
    for label, values in throughput_device_values_by_label.items():
        if not values:
            continue
        sum_logs = 0.0
        count = 0
        for v in values:
            if v > 0:
                sum_logs += math.log(v)
                count += 1
        if count > 0:
            geomean_tp_device_by_label[label] = math.exp(sum_logs / count)

    return geomean_tp_by_label, geomean_tp_device_by_label

def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    filename_to_size, filename_to_tag_to_metrics = read_input_csv(input_path)
    edge_info_map: Optional[Dict[str, Dict[str, str]]] = None
    if args.edge_info:
        edge_info_map = read_edge_info_csv(os.path.abspath(args.edge_info))
    geomean_by_label, geomean_device_by_label = write_output_csv(
        output_path,
        filename_to_size,
        filename_to_tag_to_metrics,
        args.app,
        edge_info_map,
    )

    # Print debug summary line of geomeans in the requested order
    summary_order = ["afl", "cufuzz", "asan", "memcheck", "racecheck", "initcheck"]
    numbers: List[str] = []
    for key in summary_order:
        val = geomean_by_label.get(key)
        numbers.append(f"{val:.6f}" if val is not None else "NA")
    numbers_device: List[str] = []
    for key in summary_order:
        val = geomean_device_by_label.get(key)
        numbers_device.append(f"{val:.6f}" if val is not None else "NA")
    print(
        f"Done app {args.app}: Geomean numbers for the following configs [afl, cufuzz, asan, memcheck, racecheck, initcheck, afl_d, cufuzz_d, asan_d, memcheck_d, racecheck_d, initcheck_d] are:\n"
        + "geomean," + args.app + ", " + ", ".join(numbers + numbers_device)
    )

    # If throughput output requested, generate and print throughput summary
    if args.output_throughput:
        tp_geomean_by_label, tp_geomean_device_by_label = write_output_csv_throughput(
            os.path.abspath(args.output_throughput),
            filename_to_size,
            filename_to_tag_to_metrics,
            args.app,
            edge_info_map,
        )
        summary_order = ["vanilla", "afl", "cufuzz", "asan", "memcheck", "racecheck", "initcheck"]
        nums = [f"{tp_geomean_by_label.get(k):.6f}" if tp_geomean_by_label.get(k) is not None else "NA" for k in summary_order]
        nums_d = [f"{tp_geomean_device_by_label.get(k):.6f}" if tp_geomean_device_by_label.get(k) is not None else "NA" for k in summary_order]
        print(
            f"Done app {args.app}: throughput numbers for the following configs [vanilla, afl, cufuzz, asan, memcheck, racecheck, initcheck, vanilla_d, afl_d, cufuzz_d, asan_d, memcheck_d, racecheck_d, initcheck_d] are:\n"
            + "throughput," + args.app + ", " + ", ".join(nums + nums_d)
        )


if __name__ == "__main__":
    main()


