# `lse_sum != lse_sum` — the NaN filter inside flash-attention

This note answers a single question that came out of running
`quickDemoRun.sh` and looking at the trace: **the exceptions nixnan
captured inside `pytorch_flash::flash_fwd_kernel` — what do they
correspond to in the flash-attention source, and why does flash-
attention have an explicit `lse_sum != lse_sum` check in it?**

## The exact source location

PyTorch vendors Dao-AILab/flash-attention as
`aten/src/ATen/native/transformers/cuda/flash_attn/`. The CUDA kernel
PyTorch builds and exposes as the C++ symbol
`pytorch_flash::flash_fwd_kernel` is the one compiled from
[`csrc/flash_attn/src/flash_fwd_kernel.h`](https://github.com/Dao-AILab/flash-attention/blob/82d6441eec5d4dfec120153db2c0145ae855a083/csrc/flash_attn/src/flash_fwd_kernel.h)
in the upstream repo.

The NaN filter is at **line 1197**, in the LSE-finalisation block
that runs at the end of the kernel before writing the per-row
log-sum-exp out to global memory:

```cpp
// flash_fwd_kernel.h, lines 1185-1199
for (int l = 1; l < kNLsePerThread; ++l) {
    lse_max = max(lse_max, lse_accum(l));
}
MaxOp<float> max_op;
lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);
lse_max = lse_max == -INFINITY ? 0.0f : lse_max;  // [A] -- see below
float lse_sum = expf(lse_accum(0) - lse_max);
for (int l = 1; l < kNLsePerThread; ++l) {
    lse_sum += expf(lse_accum(l) - lse_max);
}
SumOp<float> sum_op;
lse_sum = Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);
// For the case where all local lse == -INFINITY, we want to set
// lse_logsum to INFINITY. Otherwise lse_logsum is log(0.0) = -INFINITY
// and we get NaN when we do lse_accum(l) - lse_logsum.
ElementAccum lse_logsum =                                  // [B]
    (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY
                                           : logf(lse_sum) + lse_max;
```

There are actually **three** copies of the same idiom in the Tri Dao
repo — for completeness:

| File | Variant |
|---|---|
| `csrc/flash_attn/src/flash_fwd_kernel.h:1197` | Ampere/Hopper CUDA C++ (the one PyTorch vendors) |
| `hopper/flash_fwd_combine_kernel.h` | FlashAttention-3 Hopper-only split-KV combine kernel |
| `flash_attn/cute/flash_fwd_combine.py` | CUTE-DSL Python reimplementation |

The Ampere path is what runs on your RTX 3090 / A100 / Ada; the
Hopper path runs on H100. The line above is the canonical one.

## Why the filter exists — the math

Flash-attention computes attention with the online-softmax trick:
instead of materialising the full `softmax(QKᵀ/√d)` matrix, it
maintains, for every query row, two running scalars across the K/V
tiles:

- `lse_max` — running max of the pre-softmax logits (`-∞` to start)
- `lse_sum` — running sum of `exp(logit_i - lse_max)`, which is the
  unnormalised softmax denominator after the max-subtraction trick

At the end the per-row log-sum-exp is

```
lse_logsum  =  log(lse_sum) + lse_max
```

and this gets used to compute the final probabilities and (when
needed) the gradient.

**The pathological case: a fully-masked row.** Under causal masking
or padding, an entire query row's K/V positions may all be masked
out, so every logit going into `lse_accum` is `-INFINITY`. Then:

1. `lse_max` is `-INFINITY` everywhere → clamped to `0.0` by line `[A]`.
2. Every `expf(-inf - 0) == 0.0`, so `lse_sum` accumulates to `0.0`.
3. The naïve `log(lse_sum) + lse_max` gives `log(0) + 0 = -INFINITY`.
4. Downstream code does `lse_accum(l) - lse_logsum`, i.e. `-inf - -inf`,
   which by IEEE-754 is **NaN**. That NaN then poisons all downstream
   gradients.

The line-1197 expression short-circuits both pathways to `INFINITY`:

```cpp
(lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max
```

- `lse_sum == 0.f` catches the clean "all -inf inputs" case described
  above.
- **`lse_sum != lse_sum`** is the IEEE-754 self-inequality test for
  NaN. It catches the messier case where the all-reduce or earlier
  arithmetic somehow already produced a NaN (e.g. from an `inf + (-inf)`
  partial sum during the warp/cluster reduce, or from a denormal
  flush in the `expf`).

In both cases the kernel writes `INFINITY` into `lse_logsum`, so
downstream `lse_accum(l) - INFINITY = -INFINITY` and `exp(-INFINITY) = 0`,
yielding a clean all-zero probability row instead of a NaN cascade.

This is the only NaN-filtering construct in the entire forward
kernel — and it sits at the most numerically delicate point of the
algorithm. The `!=`-against-self idiom is the standard portable C
spelling because:

- `isnan(x)` requires `<math.h>` and isn't always inlined in CUDA;
- `__isnanf` exists on some compute-caps but not all;
- `x != x` is guaranteed by IEEE-754-2008 to be `true` iff `x` is
  NaN, and compiles to a single SASS `FSETP.NEU` (or in this register
  context, just falls out of the comparison logic for free).

## How this maps to nixnan's trace

In the reference run committed at
[`saved_traces/dizzy_2026-06-25_FIRST_SUCCESS.nixnan`](saved_traces/dizzy_2026-06-25_FIRST_SUCCESS.nixnan),
nixnan captured 650 FP exception events; of those:

| where | how many |
|---|---:|
| inside `pytorch_flash::flash_fwd_kernel` | **328** |
| ↳ `error [-infinity]` operands to `HMMA.16816.F32` | 224 |
| ↳ `error [subnormal]` operands to `HMMA.16816.F32` | 104 |

Those 224 `-infinity` exceptions are exactly the values that feed
into `lse_accum` from masked-out positions. They are not bugs and
they are not a numerical mishap — they are the algorithm's
*designed inputs* to the all-masked-row corner case. The
`lse_sum != lse_sum` filter at line 1197 is there *because* those
values land in the kernel.

What nixnan adds is the ability to see, at SASS resolution, that
the inputs really are flowing through the Tensor-Core
matrix-multiply-accumulate instructions (`HMMA.16816.F32`) carrying
`-inf` values. Without that visibility you have to trust the
algorithm's author that the masking is wired correctly; with
nixnan you can verify that the only `-inf`-bearing instructions
are inside the masking-protected code path and not leaking into,
say, the Q@Kᵀ matmul of an unrelated layer.

## Why the subnormals are interesting too

The 104 `[subnormal]` exceptions inside the flash-attention kernel
are a different story — those are real numerical-stability
artifacts of fp16 arithmetic on near-zero activations. fp16's
smallest normal value is 2⁻¹⁴ ≈ 6.1e-5; anything between 0 and
that is subnormal and on many GPUs is flushed to zero. The
specific operand positions in the trace come from the matmul
between the softmax-rescaled probabilities and the V tensor, where
post-softmax probabilities for low-attention positions can land in
the subnormal range.

These are not filtered by the `lse_sum != lse_sum` check (which
only addresses NaN/Inf). They're a known source of
numerical-precision drift in fp16 attention and one of the
motivations for bf16 (which has a wider exponent range and so
fewer subnormal hits) and for fp8 mixed-precision schemes that
explicitly carry per-tile scaling factors.

## Reading list

- The originating paper:
  Dao et al., *FlashAttention: Fast and Memory-Efficient Exact
  Attention with IO-Awareness*, NeurIPS 2022.
- The originating fix:
  the all-masked-row edge case was added when flash-attention added
  support for variable-length (jagged) sequences;
  [PR #237 in Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/pull/237)
  is one of the early commits in this area.
- The PyTorch vendor copy: `aten/src/ATen/native/transformers/cuda/flash_attn/`
  in [pytorch/pytorch](https://github.com/pytorch/pytorch).
