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
report-time-to-exposure.py - Summarize time-to-exposure for discovered bugs

This script aggregates crash data to determine the minimum time required
to expose each unique bug across different fuzzing configurations. It
produces a summary table useful for comparing bug-finding effectiveness.

Usage:
    python3 report-time-to-exposure.py --input <CSV> --output <CSV> [options]

Options:
    --input FILE     - Input CSV with crash data (origin, signature, filename, time)
    --output FILE    - Output summary CSV
    --configs LIST   - Comma-separated list of configurations to include
    --regexbugs FILE - File with regex patterns for grouping similar bugs

Input CSV required columns:
    origin, signature, filename, time

Output CSV columns:
    app, signature, <config1>, <config2>, ...
    (values are time-to-exposure in HH:MM:SS format, or "None" if not found)

Example:
    python3 report-time-to-exposure.py \\
        --input crashes.csv \\
        --output tte_report.csv \\
        --configs "out_seeds_cufuzz_1,out_seeds_aflpp_1"
"""
import argparse
import csv
import os
import sys
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


def extract_app_from_origin(origin: str) -> str:
	base = os.path.basename(origin.strip())
	name_without_ext, _ = os.path.splitext(base)
	if "_" in name_without_ext:
		return name_without_ext.split("_", 1)[0]
	return name_without_ext


def extract_variant_from_filename(path: str) -> Optional[str]:
	# Extract the path segment immediately before the 'default' directory
	p = path.strip().strip('"').strip("'")
	parts = [part for part in p.split("/") if part]
	try:
		idx = parts.index("default")
		if idx > 0:
			return parts[idx - 1]
		return None
	except ValueError:
		# 'default' not found in path
		return None


def parse_configs_list(configs_args: Optional[List[str]]) -> Optional[List[str]]:
	if not configs_args:
		return None
	result: List[str] = []
	for item in configs_args:
		if not item:
			continue
		for token in item.split(","):
			tok = token.strip()
			if tok:
				result.append(tok)
	# Preserve order of first appearance while de-duplicating
	seen: Set[str] = set()
	unique: List[str] = []
	for v in result:
		if v not in seen:
			seen.add(v)
			unique.append(v)
	return unique if unique else None


def is_regex_like(token: str) -> bool:
	"""Heuristically determine if token looks like a regex pattern."""
	return bool(re.search(r"[.\^$*+?{}\[\]\\|()]", token))


def build_columns_from_requested(
	discovered_variants: Set[str],
	requested: Optional[List[str]],
) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
	"""Build report columns from requested tokens.

	Returns a tuple: (columns, unmatched_labels)
	- columns: list of (label, member_variants)
	- unmatched_labels: labels that matched nothing
	"""
	variants_sorted = sorted(discovered_variants)
	if not requested:
		return [(v, [v]) for v in variants_sorted], []

	columns: List[Tuple[str, List[str]]] = []
	unmatched: List[str] = []
	for tok in requested:
		members: List[str] = []
		if is_regex_like(tok):
			try:
				pattern = re.compile(tok)
				members = [v for v in variants_sorted if pattern.fullmatch(v)]
			except re.error:
				members = []
		else:
			if tok in discovered_variants:
				members = [tok]
			else:
				members = []
		if not members:
			unmatched.append(tok)
		columns.append((tok, members))
	return columns, unmatched


def load_regexbugs(regex_file: Optional[str]) -> Optional[List[Tuple[str, re.Pattern]]]:
	if not regex_file:
		return None
	patterns: List[Tuple[str, re.Pattern]] = []
	with open(regex_file, "r") as f:
		for line in f:
			label = line.strip()
			if not label or label.startswith("#"):
				continue
			try:
				patterns.append((label, re.compile(label)))
			except re.error:
				# Skip invalid regex lines
				continue
	return patterns if patterns else None


def normalize_signature(signature: str, regexbugs: Optional[List[Tuple[str, re.Pattern]]]) -> str:
	if not regexbugs:
		return signature
	for label, pattern in regexbugs:
		if pattern.search(signature):
			return label
	return signature


def read_and_aggregate(
	input_csv: str,
	regexbugs: Optional[List[Tuple[str, re.Pattern]]] = None,
) -> Tuple[Dict[Tuple[str, str], Dict[str, int]], Set[str]]:
	# Map: (app, signature) -> { variant -> min_time_ms }
	agg: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(dict)
	variants: Set[str] = set()

	with open(input_csv, "r", newline="") as f:
		reader = csv.DictReader(f)
		required_fields = {"origin", "signature", "filename", "time"}
		missing = required_fields - set(reader.fieldnames or [])
		if missing:
			raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

		for row in reader:
			try:
				origin = row.get("origin", "")
				signature = row.get("signature", "")
				filename = row.get("filename", "")
				time_str = row.get("time", "").strip()
				if not origin or not signature or not filename or not time_str:
					continue

				app = extract_app_from_origin(origin)
				variant = extract_variant_from_filename(filename)
				if not variant:
					continue

				canonical_signature = normalize_signature(signature, regexbugs)
				time_ms = int(time_str)
				key = (app, canonical_signature)
				prev = agg[key].get(variant)
				if prev is None or time_ms < prev:
					agg[key][variant] = time_ms
				variants.add(variant)
			except Exception:
				# Skip malformed rows silently to be robust
				continue

	return agg, variants


def write_pivot_csv(
	output_csv: str,
	agg: Dict[Tuple[str, str], Dict[str, int]],
	columns: List[Tuple[str, List[str]]],
) -> None:
	# Header: app, signature, <columns...>
	header = ["app", "signature"] + [label for (label, _) in columns]
	rows_out: List[List[str]] = []

	def ms_to_hhmmss(ms: int) -> str:
		"""Convert milliseconds to HH:MM:SS string."""
		total_seconds = ms // 1000
		hours = total_seconds // 3600
		minutes = (total_seconds % 3600) // 60
		seconds = total_seconds % 60
		return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

	for (app, signature) in sorted(agg.keys(), key=lambda k: (k[0], k[1])):
		row: List[str] = [app, signature]
		per_variant = agg[(app, signature)]
		for (_label, members) in columns:
			values = [per_variant[v] for v in members if v in per_variant]
			val = min(values) if values else None
			row.append(ms_to_hhmmss(val) if val is not None else "None")
		rows_out.append(row)

	with open(output_csv, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(header)
		writer.writerows(rows_out)



def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Summarize time-to-exposure (min time in ms) per app and bug signature across variants."
		)
	)
	parser.add_argument("--input", required=True, help="Path to input CSV with raw crash data")
	parser.add_argument(
		"--configs",
		action="append",
		help=(
			"Optional list of variants to include (comma-separated or multiple --configs). "
			"If omitted, include all discovered variants."
		),
	)
	parser.add_argument("--regexbugs", help="Path to file with one regex bug pattern per line")
	parser.add_argument("--output", required=True, help="Path to write the output CSV report")
	args = parser.parse_args(argv)

	regexbugs = load_regexbugs(args.regexbugs)
	agg, discovered_variants = read_and_aggregate(args.input, regexbugs)

	# Print discovered configurations first
	sorted_variants = sorted(discovered_variants)
	print("Discovered configurations (variants):")
	for v in sorted_variants:
		print(f"- {v}")

	# Apply optional filtering and regex grouping
	requested = parse_configs_list(args.configs)
	columns, unmatched = build_columns_from_requested(discovered_variants, requested)
	if unmatched:
		print(
			"Note: requested variants/patterns matched nothing; they will be included with None values: "
			+ ", ".join(unmatched),
			file=sys.stderr,
		)

	write_pivot_csv(args.output, agg, columns)
	print(f"Wrote report: {args.output}")
	labels = [label for (label, _) in columns]
	print(f"Variants in report: {', '.join(labels) if labels else '(none)'}")
	print(f"Rows: {len(agg)}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
