#!/usr/bin/env bash
# =============================================================================
# quickDemoRun.sh — the fully-documented walk-through demo.
# =============================================================================
#
# WHAT THIS SCRIPT IS FOR
# -----------------------
# A one-command live demo that shows nixnan (NVBit-based SASS-level binary
# instrumentation) catching IEEE-754 floating-point exceptions inside the
# Tensor-Core matmuls of a stock BioMistral-7B fp16 forward pass.
#
# The "story arc" of the demo:
#
#   1. Load BioMistral-7B onto the GPU in fp16 (about 15 GB of weights).
#      Pure fp16 (no 4-bit quantization) means cuBLAS will dispatch HMMA
#      Tensor-Core kernels for every linear-layer matmul -- which is
#      exactly what nixnan is most interesting at instrumenting.
#
#   2. Run ONE prompt (default: "dizzy") through model.generate() under
#      LD_PRELOAD=nixnan.so.  nixnan installs an NVBit hook that
#      intercepts every CUDA kernel launch.  For each unique kernel
#      signature it cold-instruments (CPU-side SASS rewrite + upload of
#      the patched binary) once every SAMPLING invocations; the rest
#      run native.  SAMPLING=64 is the sweet spot on a 7B model where
#      SAMPLING=0 (instrument every launch) stalls forever and
#      SAMPLING=1000 misses too many events.
#
#   3. Tail the trace file.  The moment the FIRST "#nixnan: error [...]"
#      line appears, we know nixnan has caught something.  We SIGTERM
#      Python so nixnan's atexit handler can flush its internal buffers
#      cleanly; the exception lines themselves were already
#      line-flushed and persist.
#
#   4. Scan the trace and emit a structured summary: per kind
#      (NaN / Infinity / -Infinity / subnormal / div0), per dtype,
#      per SASS opcode, per kernel name, and -- crucially -- per
#      "kernel family" with a special call-out for any exceptions
#      caught inside PyTorch's flash-attention kernel
#      (pytorch_flash::flash_fwd_kernel).  See
#      explanationNanFiltering.md for why the flash-attention bucket
#      is the most theoretically interesting one.
#
# WHAT IT DOESN'T DO
# ------------------
# - It does NOT modify the model or the input.  No quantization, no
#   special masking, no custom kernel -- this is BioMistral-7B
#   verbatim, hitting cuBLAS + flash-attn the way every PyTorch
#   inference does.  The exceptions nixnan catches are present in
#   every fp16 transformer inference; nixnan just makes them visible.
#
# - It does NOT wait for generate() to finish.  We deliberately
#   SIGTERM on first exception so the demo finishes in well under a
#   minute (instead of the cold-instrumentation hour-plus that an
#   exhaustive run would take on a 7B model under nixnan).
#
# ON-BEAST DEFAULTS (the host this was developed on)
# --------------------------------------------------
# All env vars below default to the working install on Beast.  No
# setup needed -- just run ./quickDemoRun.sh and it goes.
#
#   NIXNAN_SO   ~/repos/claude-mistral/saved-mistral/nixnan.so
#               (the Oct-2025 build; pre fp-reset branch -- the
#               fp-reset branch has per-launch interception costs
#               that stall on a 7B model)
#
#   MODEL_PATH  ~/repos/claude-mistral/saved-mistral/biomistral-7b
#               (local HF checkpoint; ~14 GB)
#
#   PYTHON      ~/repos/claude-mistral/saved-mistral/newmed/venv/bin/python3
#               (a venv with torch 2.6 + transformers 4.57 + the
#               other deps in requirements.txt)
#
#   NV_LIBDIR   ~/opt/nv580.126/usr/lib/x86_64-linux-gnu
#               (matched userspace libcuda.so for the 580.126
#               kernel module; needed because the system libcuda on
#               Beast is 580.159 and mismatch trips CUDA error 804.
#               Harmless if missing on other hosts.)
#
# ON A FRESH MACHINE
# ------------------
# Run ./setup.sh --all once to (a) create QuickDemoNN/.venv with the
# right torch wheel, (b) clone+build parfloat/nixnan, (c) download
# BioMistral-7B from HuggingFace.  Then export the printed env vars
# and rerun this script.
#
# OUTPUT
# ------
# - saved_traces/<tag>.nixnan  : the raw SASS-level exception trace
# - stdout                     : the live status messages + the
#                                structured summary
# - exit code                  : 0 if at least one exception was
#                                captured, 1 if nothing was caught
#                                before Python exited on its own
#
# SEE ALSO
# --------
# explanationNanFiltering.md   the deeper writeup on the flash-
#                              attention `lse_sum != lse_sum` idiom
#                              and why this demo's trace surfaces
#                              the exact -infinity operands that
#                              that filter catches
# saved_traces/SUMMARY.txt     summary of the committed reference
#                              trace (650 exceptions from a prior
#                              run, useful if you want to compare
#                              against today's run)
# README.md                    project-level overview + dependency
#                              setup instructions
# =============================================================================

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# -----------------------------------------------------------------------------
# Configuration (every variable is overridable from the environment)
# -----------------------------------------------------------------------------

NIXNAN_SO="${NIXNAN_SO:-/home/ganesh/repos/claude-mistral/saved-mistral/nixnan.so}"
MODEL_PATH="${MODEL_PATH:-/home/ganesh/repos/claude-mistral/saved-mistral/biomistral-7b}"
PROMPT="${PROMPT:-dizzy}"
SAMPLING="${SAMPLING:-64}"
MAX_NEW="${MAX_NEW:-8}"
TAG="${TAG:-quickDemoRun_$(date -u +%Y%m%dT%H%M%SZ)}"

NV_LIBDIR="${NV_LIBDIR:-/home/ganesh/opt/nv580.126/usr/lib/x86_64-linux-gnu}"
if [[ -d "$NV_LIBDIR" ]]; then
    export LD_LIBRARY_PATH="$NV_LIBDIR:${LD_LIBRARY_PATH:-}"
fi
PYTHON="${PYTHON:-/home/ganesh/repos/claude-mistral/saved-mistral/newmed/venv/bin/python3}"

# -----------------------------------------------------------------------------
# Sanity checks (fail fast with a useful message)
# -----------------------------------------------------------------------------

[[ -f "$NIXNAN_SO" ]]   || { echo "[error] NIXNAN_SO not found: $NIXNAN_SO"  >&2; exit 2; }
[[ -d "$MODEL_PATH" ]]  || { echo "[error] MODEL_PATH not found: $MODEL_PATH" >&2; exit 2; }
[[ -x "$PYTHON" ]]      || { echo "[error] PYTHON not executable: $PYTHON"   >&2; exit 2; }

mkdir -p "$HERE/saved_traces"
LOGFILE="$HERE/saved_traces/${TAG}.nixnan"

# -----------------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------------

cat <<EOF

================================================================================
quickDemoRun.sh -- nixnan catching FP exceptions in BioMistral-7B
--------------------------------------------------------------------------------
  NIXNAN_SO   : $NIXNAN_SO
  MODEL_PATH  : $MODEL_PATH
  PROMPT      : '$PROMPT'
  SAMPLING    : $SAMPLING  (instrument 1 in $SAMPLING kernel invocations)
  MAX_NEW     : $MAX_NEW   (max tokens to try to generate)
  TAG         : $TAG
  LOGFILE     : $LOGFILE
================================================================================

Step 1: launching Python under LD_PRELOAD=nixnan.so ...
        (Python's stdout shows the model load + generate progress;
         nixnan's SASS-level trace lands in the LOGFILE above)

EOF

# -----------------------------------------------------------------------------
# Launch the inference under nixnan
# -----------------------------------------------------------------------------

LD_PRELOAD="$NIXNAN_SO" \
LOGFILE="$LOGFILE" \
PRINT_ILL_INSTR=1 \
TOOL_VERBOSE=0 \
INSTR_MEM=0 \
LINE_INFO=0 \
SAMPLING="$SAMPLING" \
NOBANNER=1 \
"$PYTHON" "$HERE/run_query.py" \
    --prompt "$PROMPT" \
    --tag "$TAG" \
    --max-new-tokens "$MAX_NEW" \
    --model-path "$MODEL_PATH" &
PYPID=$!
echo "[quickDemoRun] Python PID = $PYPID; tailing $LOGFILE for the first '#nixnan: error' line ..."

# -----------------------------------------------------------------------------
# Wait for either (a) the first exception to land in the trace, or
# (b) Python to exit on its own.  On (a), SIGTERM Python so nixnan's
# atexit flushes; on (b), just continue to the summary block.
# -----------------------------------------------------------------------------

caught_an_exception=0
while kill -0 "$PYPID" 2>/dev/null; do
    if [[ -f "$LOGFILE" ]] && grep -q "^#nixnan: error \[" "$LOGFILE" 2>/dev/null; then
        caught_an_exception=1
        echo
        echo "[quickDemoRun] >>> FIRST EXCEPTION DETECTED <<<"
        echo "[quickDemoRun] sending SIGTERM to Python so nixnan can finalise"
        kill -TERM "$PYPID" 2>/dev/null
        # give nixnan up to 15 s to drain its buffers and exit
        for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
            kill -0 "$PYPID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$PYPID" 2>/dev/null; then
            echo "[quickDemoRun] still alive after 15 s -- escalating to SIGKILL"
            kill -KILL "$PYPID" 2>/dev/null || true
        fi
        break
    fi
    sleep 2
done
wait "$PYPID" 2>/dev/null || true

# -----------------------------------------------------------------------------
# Step 2: scan the trace and print a structured summary of every
# exception captured so far.  This is what runs even when nixnan was
# killed mid-generation -- the previously-flushed lines are all there.
# -----------------------------------------------------------------------------

echo
echo "================================================================================"
echo "Step 2: scanning trace and summarizing exceptions found so far"
echo "================================================================================"
echo "Trace file : $LOGFILE"
echo "Size       : $(stat -c %s "$LOGFILE") bytes"
echo

total=$(grep -c "^#nixnan: error " "$LOGFILE" 2>/dev/null || echo 0)
echo "Total FP exception events captured : $total"
echo

if (( total == 0 )); then
    echo "(no FP exceptions were caught before Python exited)"
    echo
    echo "If this is unexpected:"
    echo "  - confirm SAMPLING is not so large that all events are skipped"
    echo "  - confirm the model is actually running on the GPU (check that the"
    echo "    'compute_capability' field in run_query.py's output is non-empty)"
    echo "  - try a longer generation: MAX_NEW=32 ./quickDemoRun.sh"
    exit 1
fi

echo "--------------------------------------------------------------------------------"
echo "Breakdown by exception kind:"
echo "--------------------------------------------------------------------------------"
grep -oE "error \[[^]]+\]" "$LOGFILE" | sort | uniq -c | sort -rn | sed 's/^/  /'
echo

echo "--------------------------------------------------------------------------------"
echo "Breakdown by floating-point dtype:"
echo "--------------------------------------------------------------------------------"
grep -oE "of type [a-z0-9]+" "$LOGFILE" | sort | uniq -c | sort -rn | sed 's/^/  /'
echo

echo "--------------------------------------------------------------------------------"
echo "Top SASS opcodes triggering exceptions:"
echo "--------------------------------------------------------------------------------"
grep -oE "instruction [A-Z0-9.]+" "$LOGFILE" | sort | uniq -c | sort -rn | head -10 | sed 's/^/  /'
echo

echo "--------------------------------------------------------------------------------"
echo "Breakdown by kernel (full demangled C++ name):"
echo "--------------------------------------------------------------------------------"
grep -oE "in function .+ of type" "$LOGFILE" \
    | sed -E 's/^in function //; s/ of type$//' \
    | sort | uniq -c | sort -rn | sed 's/^/  /'
echo

# -----------------------------------------------------------------------------
# Special call-out: flash-attention.  This is the headline finding --
# nixnan caught a substantial fraction of all exceptions inside
# pytorch_flash::flash_fwd_kernel, and those exceptions are exactly
# the -infinity operands that the flash-attention kernel's own
# `lse_sum != lse_sum` NaN-filter at line 1197 of flash_fwd_kernel.h
# was written to compensate for.  See explanationNanFiltering.md.
# -----------------------------------------------------------------------------

flash_count=$(grep -c "pytorch_flash::flash_fwd_kernel" "$LOGFILE" 2>/dev/null || echo 0)
flash_errors=$(grep -c "^#nixnan: error .*pytorch_flash::flash_fwd_kernel" "$LOGFILE" 2>/dev/null || echo 0)

echo "--------------------------------------------------------------------------------"
echo "Flash-attention focus (see explanationNanFiltering.md for the why)"
echo "--------------------------------------------------------------------------------"
echo "  Exception lines mentioning pytorch_flash::flash_fwd_kernel : $flash_errors / $total"

if (( flash_errors > 0 )); then
    echo
    echo "  Exception-kind breakdown WITHIN flash-attention:"
    grep "pytorch_flash::flash_fwd_kernel" "$LOGFILE" \
        | grep -oE "error \[[^]]+\]" | sort | uniq -c | sort -rn | sed 's/^/    /'
    echo
    echo "  These -infinity operands and subnormals are exactly the values that"
    echo "  feed into the lse_sum / lse_max / lse_logsum machinery in"
    echo "  Dao-AILab/flash-attention's csrc/flash_attn/src/flash_fwd_kernel.h"
    echo "  around lines 1185-1197.  The expression"
    echo
    echo "      ElementAccum lse_logsum ="
    echo "          (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY"
    echo "                                                 : logf(lse_sum) + lse_max;"
    echo
    echo "  uses the lse_sum != lse_sum self-inequality as a NaN filter for"
    echo "  fully-masked attention rows.  nixnan is observing the raw"
    echo "  pre-filter operands."
fi

echo
echo "================================================================================"
echo "First 5 exception lines verbatim (for context):"
echo "================================================================================"
grep "^#nixnan: error " "$LOGFILE" | head -5 | sed 's/^/  /'
echo
echo "Full trace : $LOGFILE"
echo "Reference  : saved_traces/dizzy_2026-06-25_FIRST_SUCCESS.nixnan"
echo "             saved_traces/SUMMARY.txt    (numbers from that prior run)"
echo

exit 0
