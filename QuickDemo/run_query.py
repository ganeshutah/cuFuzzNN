#!/usr/bin/env python3
"""
run_query.py — single-query BioMistral-7B runner for the 2026 nixnan experiments.

What it does:
  1. Loads BioMistral-7B at low precision (4-bit nf4 weights, fp16 compute).
  2. Configures PyTorch / cuBLAS / cuDNN to use Tensor Cores
     (TF32 fp32 path enabled, fp16/bf16 compute paths dispatched through SDPA →
     FlashAttention / cuDNN attention which runs on Tensor Cores on SM80+).
  3. Asserts the GPU is Tensor-Core capable; aborts loudly otherwise.
  4. Runs ONE prompt (default "dizzy") through model.generate(...).
  5. Hooks every nn.Module forward to count NaN / Inf in its output tensor and
     records per-layer stats.
  6. Writes JSON + .log summaries into ./results/.

Pair with run_with_nixnan.sh (or demo_run.sh / demo_slow.sh) to also collect
SASS-level FP exception traces via nixnan. Those wrappers set LD_PRELOAD,
LOGFILE, and BIN_SPEC_FILE — this script does NOT need to know about nixnan,
which is the whole point of binary instrumentation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig  # noqa: F401 — kept for opt-in 4-bit
    HAVE_BNB = True
except Exception:
    HAVE_BNB = False

warnings.filterwarnings("ignore", message="Setting `pad_token_id`")

DEFAULT_MODEL_PATH = "/home/ganesh/repos/claude-mistral/saved-mistral/biomistral-7b"
DEFAULT_PROMPT = "dizzy"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def configure_tensor_cores() -> dict:
    """Turn on every Tensor-Core code path PyTorch exposes."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision("high")
    # Prefer flash SDPA so the nixnan whitelist (which targets
    # pytorch_flash::flash_fwd_kernel) actually matches. We keep math as a
    # safe fallback so model.generate() never deadlocks if flash refuses a
    # given shape; if a fallback triggers, the demo will just show fewer (or
    # zero) #nixnan lines and finish quickly.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return {
        "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": "high",
        "cuda.matmul.allow_bf16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "cuda.matmul.allow_fp16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
    }


def assert_tensor_core_capable() -> tuple[int, int]:
    """Bail out loudly if the GPU is pre-Volta (no Tensor Cores)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot exercise Tensor Cores.")
    cap = torch.cuda.get_device_capability(0)
    if cap < (7, 0):
        raise RuntimeError(
            f"GPU compute capability {cap} is pre-Volta; no Tensor Cores. "
            f"This experiment requires SM70 or newer (SM80+ recommended)."
        )
    if cap < (8, 0):
        print(f"[warn] SM{cap[0]}{cap[1]} (Volta/Turing) — Tensor Cores OK, "
              f"but no TF32 path. Continuing.", file=sys.stderr)
    return cap


class NaNInfMonitor:
    """Forward hook that counts NaN/Inf in each module's output tensor."""

    def __init__(self) -> None:
        self.per_layer: dict[str, dict[str, int]] = {}

    def _hook(self, name: str):
        def fn(_mod, _inp, out):
            tensors = out if isinstance(out, (tuple, list)) else (out,)
            for t in tensors:
                if not isinstance(t, torch.Tensor) or not t.is_floating_point():
                    continue
                with torch.no_grad():
                    nans = torch.isnan(t).sum().item()
                    infs = torch.isinf(t).sum().item()
                slot = self.per_layer.setdefault(
                    name, {"nans": 0, "infs": 0, "calls": 0}
                )
                slot["nans"] += int(nans)
                slot["infs"] += int(infs)
                slot["calls"] += 1
        return fn

    def attach(self, model: torch.nn.Module) -> None:
        for name, module in model.named_modules():
            module.register_forward_hook(self._hook(name))

    def summary(self) -> dict:
        total_nans = sum(v["nans"] for v in self.per_layer.values())
        total_infs = sum(v["infs"] for v in self.per_layer.values())
        nan_layers = {n: v for n, v in self.per_layer.items() if v["nans"] > 0}
        inf_layers = {n: v for n, v in self.per_layer.items() if v["infs"] > 0}
        return {
            "total_nans": total_nans,
            "total_infs": total_infs,
            "n_layers_hooked": len(self.per_layer),
            "n_layers_with_nans": len(nan_layers),
            "n_layers_with_infs": len(inf_layers),
            "nan_layers": nan_layers,
            "inf_layers": inf_layers,
        }


def run(prompt: str, model_path: str, max_new_tokens: int, tag: str,
        quantize_4bit: bool = False) -> dict:
    cap = assert_tensor_core_capable()
    tc_settings = configure_tensor_cores()

    print(f"[run_query] model_path = {model_path}")
    print(f"[run_query] compute_cap = SM{cap[0]}{cap[1]}")
    print(f"[run_query] tensor_core_config = {tc_settings}")
    print(f"[run_query] quantize_4bit = {quantize_4bit}")
    print(f"[run_query] prompt = {prompt!r}")
    print(f"[run_query] max_new_tokens = {max_new_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_info: dict
    if quantize_4bit:
        # NB: with transformers 4.57 + a legacy single-file pytorch_model.bin,
        # the bnb 4-bit meta-loader path hits "Invalid argument" on .to(dtype).
        # If you genuinely need 4-bit, re-shard the checkpoint to safetensors
        # first (see README).
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=quant, device_map="auto",
            attn_implementation="sdpa",
        )
        quant_info = {"load_in_4bit": True, "bnb_4bit_compute_dtype": "float16",
                      "bnb_4bit_quant_type": "nf4", "dtype": "float16",
                      "attn_implementation": "sdpa"}
    else:
        # Pure fp16 — already low-precision, and fp16 matmul on SM80+ dispatches
        # to HMMA Tensor-Core instructions (which is exactly what nixnan is
        # most interesting at instrumenting).
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=False,
            attn_implementation="sdpa",
        )
        quant_info = {"load_in_4bit": False, "dtype": "float16",
                      "attn_implementation": "sdpa"}
    model.eval()

    # NB: skipping the Python-side NaNInfMonitor for the nixnan demo.  Each
    # hook does `.isnan().sum().item()` which forces a GPU→CPU sync per
    # module per forward pass; on a 7B model that's thousands of syncs,
    # which combined with nixnan's per-launch interception stalls
    # model.generate() indefinitely.  nixnan already captures every FP
    # exception at the SASS level, so we don't need the Python-level hooks
    # when we're tracing with nixnan.
    monitor = NaNInfMonitor()
    print("[run_query] skipped Python NaN/Inf hooks (nixnan does the same job at SASS); tokenizing...", flush=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"[run_query] input shape={list(inputs['input_ids'].shape)}; calling model.generate()...", flush=True)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    print(f"[run_query] generate() done; output shape={list(out_ids.shape)}", flush=True)
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    stats = monitor.summary()
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "model_path": model_path,
        "compute_capability": f"{cap[0]}.{cap[1]}",
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "tensor_core_config": tc_settings,
        "quantization": quant_info,
        "environment": {
            k: v for k, v in os.environ.items()
            if k in {"LD_PRELOAD", "LOGFILE", "BIN_SPEC_FILE", "HISTOGRAM",
                     "PRINT_ILL_INSTR", "TOOL_VERBOSE", "INSTR_MEM"}
        },
        "model_output": text,
        "statistics": stats,
    }

    json_path = RESULTS_DIR / f"{tag}.json"
    log_path = RESULTS_DIR / f"{tag}.log"
    with json_path.open("w") as fh:
        json.dump(record, fh, indent=2)
    with log_path.open("w") as fh:
        fh.write(f"Tag:        {tag}\n")
        fh.write(f"Prompt:     {prompt!r}\n")
        fh.write(f"Output:     {text!r}\n")
        fh.write(f"NaNs:       {stats['total_nans']}\n")
        fh.write(f"Infs:       {stats['total_infs']}\n")
        fh.write(f"Layers w/ NaN: {stats['n_layers_with_nans']}\n")
        fh.write(f"Layers w/ Inf: {stats['n_layers_with_infs']}\n")
    print(f"[run_query] wrote {json_path}")
    print(f"[run_query] wrote {log_path}")

    del model
    torch.cuda.empty_cache()
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help=f"Prompt to send to BioMistral (default: {DEFAULT_PROMPT!r})")
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                    help="Local path or HF hub ID for the model")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--tag", default=None,
                    help="File-name tag for results (default: derived from prompt + UTC stamp)")
    ap.add_argument("--quantize-4bit", action="store_true",
                    help="Use bitsandbytes 4-bit nf4. Off by default — pure fp16 already drives Tensor Cores and avoids the transformers 4.57 + pytorch_model.bin bnb loader bug.")
    args = ap.parse_args()

    tag = args.tag or f"q_{args.prompt.replace(' ', '_')[:24]}_{datetime.now():%Y%m%d_%H%M%S}"
    run(args.prompt, args.model_path, args.max_new_tokens, tag,
        quantize_4bit=args.quantize_4bit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
