# QuickDemo — nixnan catching FP exceptions in BioMistral-7B

A two-minute, one-script live demo for showing that nixnan
(NVBit-based SASS-level binary instrumentation) catches floating-point
exceptions inside the Tensor-Core matmuls of a real 7-billion-parameter
language model.

## What you'll see

When you run `./quickdemo.sh`, the script:

1. Loads BioMistral-7B in fp16 onto the GPU (≈15 GB on an RTX 3090).
2. Runs `model.generate("dizzy", max_new_tokens=8)` under
   `LD_PRELOAD=nixnan.so` with `SAMPLING=64` (one in 64 kernel
   invocations gets instrumented — keeps NVBit's per-launch overhead
   tolerable on a 7B model).
3. Tails the SASS-level trace; the moment the first
   `#nixnan: error [...]` line appears, the script sends `SIGTERM`
   to the Python process so nixnan can flush. The captured exception
   lines were already line-flushed and persist.
4. Prints a per-kind / per-dtype / per-SASS-op summary.

## The reference run

A committed reference run is at
[`saved_traces/dizzy_2026-06-25_FIRST_SUCCESS.nixnan`](saved_traces/dizzy_2026-06-25_FIRST_SUCCESS.nixnan)
with a text summary in
[`saved_traces/SUMMARY.txt`](saved_traces/SUMMARY.txt).

Headline numbers from that run:

| metric | count |
|---|---:|
| total FP exceptions captured | **650** |
| `subnormal` operands | 354 |
| `-infinity` operands | 224 |
| `NaN` operands | 64 |
| `NaN,subnormal` mixed | 8 |
| exceptions in `HMMA.16816.F32` (Tensor-Core fp16→fp32 matmul) | **394** |
| in cuBLAS `ampere_fp16_s16816gemm_*` kernels (Mistral linear layers) | **154** |

In other words: **the Tensor-Core matmuls inside a stock BioMistral-7B
fp16 forward pass routinely see subnormal / ±inf / NaN operands** —
exactly the numerical-stability behaviour that nixnan is designed to
expose.

## How to run

```bash
cd QuickDemo
./quickdemo.sh                       # default prompt is "dizzy"
PROMPT="severe chest pain" ./quickdemo.sh
SAMPLING=128 ./quickdemo.sh          # coarser, faster, fewer exceptions
SAMPLING=8   ./quickdemo.sh          # finer, slower, more exceptions
```

The script prints the captured trace summary at the end and tells you
where the full trace landed (`saved_traces/<tag>.nixnan`).

## Dependencies

The demo needs three things "available somewhere" on the host:

| dep | what | how to satisfy |
|---|---|---|
| **nixnan.so** | NVBit-based binary instrumenter, built into a shared object | clone `parfloat/nixnan` and `make ARCH=sm_<your-cc>`; produces `nixnan.so`. The Oct-2025 build of `main` is the reference. |
| **BioMistral-7B checkpoint** | HuggingFace model directory (~14 GB on disk) | `huggingface-cli download BioMistral/BioMistral-7B --local-dir <DIR>` |
| **Python with PyTorch + transformers** | venv with the packages in [`requirements.txt`](requirements.txt) | `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (plus a torch wheel matching your CUDA major — see the comment in requirements.txt) |

Then point the demo at them:

```bash
export NIXNAN_SO=/path/to/your/nixnan.so
export MODEL_PATH=/path/to/biomistral-7b
export PYTHON=/path/to/your/venv/bin/python3   # optional override
./quickdemo.sh
```

### On Beast (the host this was developed on)

Skip the setup — paths default to the working install:

```
NIXNAN_SO  = ~/repos/claude-mistral/saved-mistral/nixnan.so
MODEL_PATH = ~/repos/claude-mistral/saved-mistral/biomistral-7b
PYTHON     = ~/repos/claude-mistral/saved-mistral/newmed/venv/bin/python3
NV_LIBDIR  = ~/opt/nv580.126/usr/lib/x86_64-linux-gnu  (driver-mismatch shim)
```

Just `cd QuickDemo && ./quickdemo.sh` and the defaults work.

### On a fresh machine

```bash
cd QuickDemo
./setup.sh --help          # see what's available
./setup.sh --venv          # create .venv + install requirements (no model download)
./setup.sh --nixnan ~/nixnan       # clone + build nixnan with ARCH matched to local GPU
./setup.sh --model ~/biomistral-7b # download ~14 GB BioMistral-7B checkpoint
./setup.sh --all                   # all three of the above
```

`setup.sh` prints the env-var lines to copy-paste at the end. It's
opt-in — running it with no flags just prints a pre-flight check
(GPU, CUDA toolkit, required CLI tools).

## Why is it built this way?

Three design choices that matter for the demo:

1. **Use the Oct-2025 nixnan build, not the fp-reset branch.** The
   fp-reset branch stalls on the per-launch interception overhead
   even with a kernel whitelist — see the longer write-up in
   `~/repos/claude-mistral/biomistral-2026/README.md` for the
   diagnostic history. The Oct-2025 nixnan has the simpler hook path
   and finishes a generation step in well under a minute with
   `SAMPLING=64`.

2. **`SAMPLING=64`.** Each unique kernel signature is
   cold-instrumented on its first launch; that's a CPU-bound pass
   that walks the SASS, splices in callbacks, and reuploads the
   kernel binary. On a 7B model with hundreds of unique kernels,
   doing this on every launch is unmanageable. With `SAMPLING=64`,
   only one in 64 invocations triggers the slow path; the rest run
   native. The captured exceptions are still representative because
   the same kernels get re-launched many times.

3. **No Python-side NaN/Inf hooks.** Earlier iterations attached a
   forward hook to every `nn.Module` that called
   `torch.isnan().sum().item()` per tensor. The `.item()` forces a
   GPU→CPU sync per module per forward pass — thousands of syncs
   that interact pathologically with nixnan's interception. Removed.
   nixnan already catches every FP exception at the SASS level, which
   is what we want for the demo.

## Files

```
QuickDemo/
├── quickdemo.sh              the only script you need to run
├── run_query.py              the Python entry it invokes
├── analyze_trace.py          optional: parse trace into structured JSON
├── README.md                 this file
└── saved_traces/
    ├── dizzy_2026-06-25_FIRST_SUCCESS.nixnan   committed reference trace
    └── SUMMARY.txt           text summary of the above
```

## Environment knobs

| env var | default | purpose |
|---|---|---|
| `NIXNAN_SO` | `~/repos/claude-mistral/saved-mistral/nixnan.so` | path to the (Oct-2025) nixnan |
| `MODEL_PATH` | `~/repos/claude-mistral/saved-mistral/biomistral-7b` | local HF checkpoint |
| `PROMPT` | `dizzy` | the prompt to send |
| `SAMPLING` | `64` | nixnan's one-in-N instrumentation knob |
| `MAX_NEW` | `8` | how many tokens to generate |
| `NV_LIBDIR` | `~/opt/nv580.126/usr/lib/x86_64-linux-gnu` | userspace libcuda matched to the kernel module; only needed on this host |

If you're running on a fresh machine, point `NIXNAN_SO` and
`MODEL_PATH` at your local equivalents and you're done.

## Toolchain notes

- RTX 3090 (SM86) verified; should work on any Ampere+ Tensor-Core GPU
- CUDA 13.2 toolkit installed; driver kernel module 580.126.09
- PyTorch 2.6 (in `saved-mistral/newmed/venv`)
- transformers 4.57.1
- nixnan: Oct-2025 build (pre fp-reset); SASS-only `.so` (no PTX),
  so no runtime PTX JIT is needed
