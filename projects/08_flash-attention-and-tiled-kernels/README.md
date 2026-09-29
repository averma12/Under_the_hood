# Project 8: Flash Attention and Tiled Kernels

> The actual idea behind FlashAttention is not a new algorithm. It's a different schedule for the same algorithm — one that never materializes the full (T, T) score matrix in memory. This project demonstrates the schedule on CPU with explicit memory accounting.

## Hook

You hear "FlashAttention is 2-3× faster" so often that it's easy to assume there's a clever new math trick inside. There isn't. The math is identical. The contribution is a tiled execution schedule that respects the GPU memory hierarchy and never holds the `(T, T)` attention score matrix in main memory. That is what makes the wins possible — both wallclock and peak-memory.

## The Concept

**Naive attention** computes `scores = Q @ K.T` (shape `(T, T)`), softmaxes it, multiplies by `V`. That intermediate `(T, T)` matrix is the memory bottleneck. At `T = 8192`, fp16, it's 128 MB per head per layer.

**Tiled attention** processes `Q` in chunks of `q_block` rows. For each chunk, it walks over `K` and `V` in chunks of `kv_block` columns, maintaining a running `max` and softmax denominator. At the end of each `Q` chunk, it normalizes and writes back. The full `(T, T)` matrix is never built.

Connection to our [Chapter 7 RoPE experiment](../07_the-details-that-matter/README.md#rope-extension-what-changed-and-what-we-measured): RoPE changes the `Q` and `K` vectors *before* their dot product; FlashAttention changes how the resulting attention calculation is scheduled in memory. They solve different problems and can be combined. This chapter's small CPU reference compares naive and tiled attention without adding RoPE.

The trick is the **online softmax**:

```
m_new = max(m, max(scores_block))
alpha = exp(m - m_new)            # rescale old accumulator
beta  = exp(scores_block - m_new) # this block's contributions
output_so_far = output_so_far * alpha + beta @ V_block
denom         = denom         * alpha + sum(beta)
```

When you finish all `K/V` chunks for a given `Q` chunk, divide accumulator by denominator. The result is exactly what softmax-then-matmul-by-V produces — same math, never materializing `(T, T)`.

## Why It Matters

Long-context LLMs (32k, 100k tokens) are not possible with naive attention because the `(T, T)` matrix would be tens of GB at scale. FlashAttention's tiled schedule is the reason it's even feasible to run attention over a book's worth of tokens.

---

## What Got Built

A CPU reference implementation of tiled causal attention, side-by-side against naive attention, with explicit peak intermediate-memory accounting.

We also wired PyTorch's actual fused FlashAttention backend into the Chapter 5
GPT and paired it with four-stream mHC residual routing for a fresh A100 run.
The implementation and measurements are in
[`EXPERIMENT_MHC_FLASH.md`](EXPERIMENT_MHC_FLASH.md).

### Files in this folder

| File | What it is |
|------|------------|
| [`build.py`](build.py) | `naive_attention` and `tiled_attention`; both causal, both return `(output, peak_intermediate_floats)` |
| `step_*.py` | The book's code blocks, extracted step-by-step. Reference material. |
| [`FLASH_ATTENTION_EXPLAINED.md`](FLASH_ATTENTION_EXPLAINED.md) | My deep-dive notes: how attention runs on a GPU, roofline and HBM traffic, streaming softmax derivation, FlashAttention 1/2/3, backward via logsumexp, Flash-Decoding |
| [`explain_flash_attention.py`](explain_flash_attention.py) | Runnable checks for every numeric claim in the notes (`tests/test_explain_flash_attention.py`) |
| `tests/test_unit.py` | 10 tests: shape checks, naive peak memory is `2*T²`, tiled matches naive across multiple `(T, d_head, q_block, kv_block)` parametrizations including non-divisible cases, tiled peak < naive peak |

### How to run

```bash
python build.py --tiny      # T=128, d_head=32, instant
python build.py --full      # T=512
pytest projects/08_flash-attention-and-tiled-kernels/
```

---

## Outputs (from `python build.py --tiny`)

```
Sequence length T = 128, d_head = 32
Q,K,V each: (128, 32) = 4096 floats each

naive  :  output_shape=(128, 32)  peak_intermediate=32768 floats
tiled  :  output_shape=(128, 32)  peak_intermediate=2048 floats

max |naive - tiled|  = 3.58e-07
mean |naive - tiled| = 2.87e-08
(should be near zero — both compute the same attention)

Naive peak intermediate: 32768 floats  (= 128.0 KB at fp32)
Tiled peak intermediate: 2048 floats  (= 8.0 KB at fp32)
Memory reduction: 16.00x
```

Two things to take away:

1. **Outputs match within floating-point precision.** Max diff is 3.6e-7, mean 2.9e-8. These are the FP rounding errors you get from doing the same arithmetic in a different order. The schedule preserves correctness.

2. **Peak intermediate memory dropped 16×.** From 32k floats down to 2k. At small T this doesn't matter on CPU; the script doesn't run measurably faster. But the ratio scales with T.

### What it looks like at real LLM scale

| T | d_head | naive (peak fp16) | tiled (peak fp16) | reduction |
|---|--------|--------------------|---------------------|------------|
| 128 | 32 | 64 KB | 4 KB | 16× |
| 2048 | 64 | 16 MB | 256 KB | 64× |
| 8192 | 64 | 256 MB | 1 MB | 256× |
| 32768 | 128 | 4 GB | 8 MB | 512× |

That 4 GB → 8 MB at `T = 32k, d_head = 128` is the reason long-context fine-tuning is feasible at all on accessible hardware. The actual FlashAttention paper goes further — it tiles for the **GPU's** memory hierarchy specifically (SRAM/HBM), which is why it also wins on wallclock. But the peak-memory win is what this CPU reference makes visible.

---

## No separate BREAK IT

The bug everyone hits when writing tiled attention by hand is the **online softmax**: forgetting to rescale the accumulator when a later block produces a higher max. That bug doesn't crash — the output just silently disagrees with naive attention. The parametrized tests in `tests/test_unit.py` catch exactly this; if you mess up the rescale logic, `test_tiled_matches_naive` fails immediately. That test suite IS the break-detector.

---

## Read in the book

This project is Chapter 8 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.

Read the chapter for: the full derivation of the online-softmax recurrence, the GPU memory hierarchy walkthrough (HBM vs. SRAM, why tile sizes are tuned for SRAM), and the reason FlashAttention-2 reordered the loops to maximize parallelism over `Q` blocks rather than `K/V` blocks.
