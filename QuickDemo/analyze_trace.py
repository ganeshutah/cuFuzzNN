#!/usr/bin/env python3
"""
analyze_trace.py — summarise a nixnan SASS trace alongside the Python-side stats.

Walks results/*.json (the Python NaN/Inf hook summary) and the matching
traces/*.nixnan files (the SASS-level FP exception trace from nixnan).
Emits a one-line summary per run plus a longer report on stdout.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
TRACES_DIR = HERE / "traces"

EXC_RE = re.compile(
    r"#nixnan: error \[(?P<kind>[^\]]+)\] detected in instruction "
    r"(?P<sass>\S+).*?in function (?P<kernel>\S+).*?of type (?P<dtype>\S+)"
)
BIN_THRESH_RE = re.compile(
    r"#nixnan: (?P<dtype>\S+) bin has reached threshold: "
    r"kernel=(?P<kernel>\S+) range=(?P<range>\[[^\]]+\]) count=(?P<count>\d+)"
)
REPORT_RE = re.compile(r"#nixnan: (?P<dtype>FP\d+) Operations")


def summarise_nixnan(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    kinds = Counter()
    by_kernel = Counter()
    by_sass = Counter()
    by_dtype = Counter()
    bin_thresholds = Counter()

    for m in EXC_RE.finditer(text):
        kinds[m.group("kind")] += 1
        by_kernel[m.group("kernel")] += 1
        by_sass[m.group("sass")] += 1
        by_dtype[m.group("dtype")] += 1
    for m in BIN_THRESH_RE.finditer(text):
        bin_thresholds[
            (m.group("dtype"), m.group("kernel"), m.group("range"))
        ] += int(m.group("count"))

    report_block = {}
    # nixnan emits sections like:
    #   #nixnan: --- FP16 Operations ---
    #   #nixnan: NaN:                    0 (0 repeats)
    #   ...
    for chunk in re.split(r"#nixnan: --- ", text):
        head_line = chunk.splitlines()[0] if chunk.splitlines() else ""
        if not head_line or "Operations" not in head_line:
            continue
        section_name = head_line.replace("---", "").strip()  # "FP16 Operations", "FP16 Memory Operations"
        section = {}
        for line in chunk.splitlines()[1:]:
            mm = re.match(
                r"#nixnan:\s+(?P<k>[^:]+):\s+(?P<v>\d+)(?:\s+\((?P<rep>\d+) repeats\))?",
                line)
            if mm:
                section[mm.group("k").strip()] = {
                    "count": int(mm.group("v")),
                    "repeats": int(mm.group("rep") or 0),
                }
        report_block[section_name] = section

    return {
        "trace": str(path),
        "trace_size_bytes": path.stat().st_size,
        "exceptions_by_kind": dict(kinds),
        "exceptions_by_dtype": dict(by_dtype),
        "top_kernels": dict(by_kernel.most_common(5)),
        "top_sass": dict(by_sass.most_common(8)),
        "bin_threshold_hits": {
            f"{d}/{k}/{r}": n for (d, k, r), n in bin_thresholds.most_common(10)
        },
        "final_report": report_block,
    }


def summarise_python(path: Path) -> dict:
    with path.open() as fh:
        rec = json.load(fh)
    return {
        "tag": rec.get("tag"),
        "prompt": rec.get("prompt"),
        "compute_capability": rec.get("compute_capability"),
        "quantization": rec.get("quantization"),
        "total_nans": rec["statistics"]["total_nans"],
        "total_infs": rec["statistics"]["total_infs"],
        "n_layers_with_nans": rec["statistics"]["n_layers_with_nans"],
        "n_layers_with_infs": rec["statistics"]["n_layers_with_infs"],
        "model_output": rec.get("model_output"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None,
                    help="Only analyse the run with this tag (default: every JSON in results/)")
    ap.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON instead of a text report")
    args = ap.parse_args()

    py_records = []
    for j in sorted(RESULTS_DIR.glob("*.json")):
        if args.tag and args.tag not in j.stem:
            continue
        py_records.append((j, summarise_python(j)))

    nx_records = []
    for n in sorted(TRACES_DIR.glob("*.nixnan")):
        if args.tag and args.tag not in n.stem:
            continue
        nx_records.append((n, summarise_nixnan(n)))

    combined = {
        "python_side": [r for _, r in py_records],
        "nixnan_side": [r for _, r in nx_records],
    }
    if args.json:
        json.dump(combined, sys.stdout, indent=2)
        print()
        return 0

    print("=" * 72)
    print("Python-side NaN/Inf summary")
    print("=" * 72)
    if not py_records:
        print("(no results/*.json found)")
    for path, rec in py_records:
        print(f"\n[{path.stem}]")
        print(f"  prompt        : {rec['prompt']!r}")
        print(f"  compute cap   : {rec['compute_capability']}")
        print(f"  quant         : {rec['quantization']}")
        print(f"  total NaNs    : {rec['total_nans']}")
        print(f"  total Infs    : {rec['total_infs']}")
        print(f"  output        : {rec['model_output']!r}")

    print()
    print("=" * 72)
    print("nixnan SASS trace summary")
    print("=" * 72)
    if not nx_records:
        print("(no traces/*.nixnan found)")
    for path, rec in nx_records:
        print(f"\n[{path.name}]  ({rec['trace_size_bytes']} bytes)")
        print(f"  by kind   : {rec['exceptions_by_kind']}")
        print(f"  by dtype  : {rec['exceptions_by_dtype']}")
        print(f"  kernels   : {rec['top_kernels']}")
        print(f"  SASS ops  : {rec['top_sass']}")
        if rec["bin_threshold_hits"]:
            print(f"  bin hits  : {rec['bin_threshold_hits']}")
        if rec["final_report"]:
            print(f"  report    : {rec['final_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
