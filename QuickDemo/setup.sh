#!/usr/bin/env bash
# setup.sh — opt-in bootstrap for a fresh machine.
#
# By default does NOTHING and just prints a checklist + suggested env
# vars. Pass one or more flags to actually do work:
#
#   --venv        create QuickDemo/.venv and install requirements.txt
#   --nixnan DIR  clone+build parfloat/nixnan at DIR (default: ./nixnan)
#                 — produces DIR/nixnan.so suitable for use as NIXNAN_SO
#   --model DIR   download BioMistral/BioMistral-7B (~14 GB) into DIR
#                 (default: ./biomistral-7b) — produces a local HF
#                 checkpoint suitable for use as MODEL_PATH
#   --all         shorthand for --venv --nixnan --model
#   --help        print this and exit
#
# On Beast (where nixnan.so and biomistral-7b already exist under
# ~/repos/claude-mistral/saved-mistral/), you don't need to run this
# at all — just `./quickdemo.sh`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DO_VENV=0
DO_NIXNAN=0
NIXNAN_DIR="$HERE/nixnan"
DO_MODEL=0
MODEL_DIR="$HERE/biomistral-7b"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)    DO_VENV=1; shift;;
        --nixnan)  DO_NIXNAN=1
                   if [[ $# -ge 2 && "$2" != --* ]]; then NIXNAN_DIR="$2"; shift; fi
                   shift;;
        --model)   DO_MODEL=1
                   if [[ $# -ge 2 && "$2" != --* ]]; then MODEL_DIR="$2"; shift; fi
                   shift;;
        --all)     DO_VENV=1; DO_NIXNAN=1; DO_MODEL=1; shift;;
        --help|-h) sed -n '2,20p' "$0"; exit 0;;
        *) echo "[error] unknown arg: $1"; sed -n '2,20p' "$0"; exit 2;;
    esac
done

echo "============================================================"
echo "QuickDemo setup"
echo "  --venv   : $DO_VENV"
echo "  --nixnan : $DO_NIXNAN  (target: $NIXNAN_DIR)"
echo "  --model  : $DO_MODEL   (target: $MODEL_DIR)"
echo "============================================================"

# ---------- pre-flight environment check -------------------------------------

echo
echo "[check] tools needed"
need_cmd() { command -v "$1" >/dev/null && echo "  ok      $1: $(command -v $1)" || echo "  MISSING $1 — install it"; }
need_cmd git
need_cmd make
need_cmd python3
need_cmd nvcc
need_cmd nvidia-smi
need_cmd wget

echo
echo "[check] GPU compute capability"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv 2>&1 | head -3 | sed 's/^/  /'
else
    echo "  (nvidia-smi unavailable — confirm an Ampere-or-newer GPU manually)"
fi

# ---------- step 1: venv -----------------------------------------------------

if (( DO_VENV )); then
    echo
    echo "[venv] creating $HERE/.venv ..."
    python3 -m venv "$HERE/.venv"
    # shellcheck disable=SC1091
    source "$HERE/.venv/bin/activate"
    python3 -m pip install --upgrade pip wheel

    cuda_major=""
    if command -v nvcc >/dev/null; then
        cuda_major=$(nvcc --version 2>/dev/null | grep -oE "release [0-9]+" | awk '{print $2}')
    fi
    case "$cuda_major" in
        12) torch_index="https://download.pytorch.org/whl/cu124";;
        13) torch_index="https://download.pytorch.org/whl/cu128";;
        *)  torch_index="https://download.pytorch.org/whl/cu124"
            echo "[venv] couldn't detect CUDA major (got '$cuda_major'); defaulting torch to cu124";;
    esac
    echo "[venv] installing torch from $torch_index ..."
    python3 -m pip install --index-url "$torch_index" torch
    echo "[venv] installing remaining requirements ..."
    python3 -m pip install -r "$HERE/requirements.txt"
    echo "[venv] done; activate with:  source $HERE/.venv/bin/activate"
    deactivate || true
fi

# ---------- step 2: build nixnan ---------------------------------------------

if (( DO_NIXNAN )); then
    echo
    if [[ -d "$NIXNAN_DIR/.git" ]]; then
        echo "[nixnan] $NIXNAN_DIR already a git checkout; reusing"
    else
        echo "[nixnan] cloning parfloat/nixnan into $NIXNAN_DIR"
        git clone https://github.com/parfloat/nixnan.git "$NIXNAN_DIR"
    fi
    cd "$NIXNAN_DIR"

    # Pick a SASS arch that matches the host GPU so the .so contains
    # native SASS and skips runtime PTX JIT (which is the easiest way
    # to dodge a "PTX compiled with unsupported toolchain" error when
    # the driver is older than the toolkit).
    arch_arg=""
    if command -v nvidia-smi >/dev/null; then
        cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
        if [[ -n "$cap" ]]; then
            arch_arg="ARCH=sm_$cap"
            echo "[nixnan] building with $arch_arg (matching this GPU)"
        fi
    fi
    if [[ -z "$arch_arg" ]]; then
        echo "[nixnan] no GPU detected — building with the Makefile default"
    fi
    make $arch_arg
    echo "[nixnan] built artifact: $NIXNAN_DIR/nixnan.so"
    cd "$HERE"
fi

# ---------- step 3: download BioMistral-7B -----------------------------------

if (( DO_MODEL )); then
    echo
    if [[ -f "$MODEL_DIR/config.json" ]]; then
        echo "[model] $MODEL_DIR/config.json already exists; reusing"
    else
        echo "[model] downloading BioMistral/BioMistral-7B into $MODEL_DIR (≈14 GB) ..."
        mkdir -p "$MODEL_DIR"
        # use the venv's huggingface_hub if available, otherwise fall back
        if [[ -x "$HERE/.venv/bin/python3" ]]; then
            PY="$HERE/.venv/bin/python3"
        else
            PY="python3"
        fi
        $PY -m pip install --quiet 'huggingface_hub[cli]'
        $PY -m huggingface_hub.commands.huggingface_cli download \
            BioMistral/BioMistral-7B --local-dir "$MODEL_DIR"
    fi
    echo "[model] checkpoint at: $MODEL_DIR"
fi

# ---------- summary ----------------------------------------------------------

echo
echo "============================================================"
echo "Suggested env vars for quickdemo.sh"
echo "============================================================"
if (( DO_NIXNAN )); then
    echo "  export NIXNAN_SO=$NIXNAN_DIR/nixnan.so"
fi
if (( DO_MODEL )); then
    echo "  export MODEL_PATH=$MODEL_DIR"
fi
if (( DO_VENV )); then
    echo "  export PYTHON=$HERE/.venv/bin/python3"
fi
echo
echo "Then:   cd $HERE && ./quickdemo.sh"
