#!/usr/bin/env bash
# quickdemo.sh — single-script live demo of nixnan catching FP exceptions
# in BioMistral-7B's HMMA Tensor-Core matmuls.
#
# What it does:
#   1. Loads BioMistral-7B in fp16 (so cuBLAS dispatches HMMA Tensor-Core
#      kernels — the ones nixnan is most interesting at instrumenting).
#   2. Runs ONE prompt (default "dizzy") through model.generate(max=8).
#   3. The whole thing runs under LD_PRELOAD=nixnan.so with SAMPLING=64
#      (instrument one in 64 invocations of each kernel) so the
#      per-launch interception overhead stays manageable on a 7B model.
#   4. As soon as the FIRST `#nixnan: error [...]` line appears in the
#      trace file, this script SIGTERMs the Python process. nixnan's
#      atexit handler flushes whatever it has already buffered, but the
#      exception lines themselves were already line-flushed and are safe.
#
# Captured trace lands in saved_traces/<tag>.nixnan; a short text
# summary is printed at the end.
#
# Reference: an example trace from the original successful run is
# checked in at  saved_traces/dizzy_2026-06-25_FIRST_SUCCESS.nixnan
# (650 exception lines: 354 subnormal, 224 -infinity, 64 NaN, 8 mixed).
#
# Configuration knobs (env vars):
#   NIXNAN_SO     path to nixnan.so (old / pre fp-reset build)
#                 default:  ~/repos/claude-mistral/saved-mistral/nixnan.so
#                 (or set NIXNAN_SO=/path/to/your/nixnan.so before running)
#   MODEL_PATH    path to a local BioMistral-7B HF checkpoint
#                 default:  ~/repos/claude-mistral/saved-mistral/biomistral-7b
#   PROMPT        the prompt string  (default: "dizzy")
#   SAMPLING      one-in-N kernel instrumentation (default: 64)
#   MAX_NEW       how many new tokens to generate (default: 8)
#   NV_LIBDIR     dir with the matched libcuda*.so (default: ~/opt/nv580.126/...)
#                 prepended to LD_LIBRARY_PATH; only needed on hosts whose
#                 system libcuda doesn't match the running kernel module.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ------------- defaults --------------------------------------------------

NIXNAN_SO="${NIXNAN_SO:-/home/ganesh/repos/claude-mistral/saved-mistral/nixnan.so}"
MODEL_PATH="${MODEL_PATH:-/home/ganesh/repos/claude-mistral/saved-mistral/biomistral-7b}"
PROMPT="${PROMPT:-dizzy}"
SAMPLING="${SAMPLING:-64}"
MAX_NEW="${MAX_NEW:-8}"
TAG="${TAG:-dizzy_$(date -u +%Y%m%dT%H%M%SZ)}"

NV_LIBDIR="${NV_LIBDIR:-/home/ganesh/opt/nv580.126/usr/lib/x86_64-linux-gnu}"
if [[ -d "$NV_LIBDIR" ]]; then
    export LD_LIBRARY_PATH="$NV_LIBDIR:${LD_LIBRARY_PATH:-}"
fi
PYTHON="${PYTHON:-/home/ganesh/repos/claude-mistral/saved-mistral/newmed/venv/bin/python3}"

# ------------- sanity ----------------------------------------------------

if [[ ! -f "$NIXNAN_SO" ]]; then
    echo "[error] NIXNAN_SO not found: $NIXNAN_SO" >&2; exit 2
fi
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "[error] MODEL_PATH not found: $MODEL_PATH" >&2; exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "[error] PYTHON not executable: $PYTHON" >&2; exit 2
fi

mkdir -p "$HERE/saved_traces"
LOGFILE="$HERE/saved_traces/${TAG}.nixnan"

cat <<EOF

############################################################
#  nixnan QuickDemo — exception detection on BioMistral-7B
#  NIXNAN_SO  : $NIXNAN_SO
#  MODEL_PATH : $MODEL_PATH
#  PROMPT     : '$PROMPT'
#  SAMPLING   : $SAMPLING  (instrument 1 in $SAMPLING kernel invocations)
#  MAX_NEW    : $MAX_NEW   (tokens generated)
#  TAG        : $TAG
#  LOGFILE    : $LOGFILE
############################################################

EOF

# ------------- start the inference, capture python's PID -----------------

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
echo "[quickdemo] python PID = $PYPID"
echo "[quickdemo] waiting for first FP exception in $LOGFILE..."

# ------------- wait for first exception, then SIGTERM --------------------

# fall through if python dies on its own
while kill -0 "$PYPID" 2>/dev/null; do
    if [[ -f "$LOGFILE" ]] && grep -q "^#nixnan: error \[" "$LOGFILE" 2>/dev/null; then
        echo
        echo "[quickdemo] >>> FIRST EXCEPTION DETECTED — sending SIGTERM <<<"
        head_excs=$(grep "^#nixnan: error \[" "$LOGFILE" | head -5)
        echo "$head_excs"
        echo
        kill -TERM "$PYPID" 2>/dev/null
        # give nixnan up to 15s to finalise
        for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
            kill -0 "$PYPID" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$PYPID" 2>/dev/null || true
        break
    fi
    sleep 2
done
wait "$PYPID" 2>/dev/null || true

# ------------- final summary --------------------------------------------

echo
echo "============================================================"
echo "Trace : $LOGFILE  ($(stat -c %s "$LOGFILE") bytes)"
echo "============================================================"

total=$(grep -c "^#nixnan: error " "$LOGFILE" 2>/dev/null || echo 0)
echo "Total FP exceptions captured : $total"
echo
echo "By kind:"
grep -oE "error \[[^]]+\]" "$LOGFILE" 2>/dev/null | sort | uniq -c | sort -rn | head -10 | sed 's/^/  /'
echo
echo "By dtype:"
grep -oE "of type [a-z0-9]+" "$LOGFILE" 2>/dev/null | sort | uniq -c | sort -rn | head -5 | sed 's/^/  /'
echo
echo "Top SASS opcodes:"
grep -oE "instruction [A-Z0-9.]+" "$LOGFILE" 2>/dev/null | sort | uniq -c | sort -rn | head -8 | sed 's/^/  /'
echo
echo "First 5 exception lines (verbatim):"
grep "^#nixnan: error " "$LOGFILE" 2>/dev/null | head -5 | sed 's/^/  /'
echo

echo "Reference run (committed):  saved_traces/dizzy_2026-06-25_FIRST_SUCCESS.nixnan"
