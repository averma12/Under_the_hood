# Our Chapter 5 notes — a GPT from a blank file

We built a next-token autocomplete model ourselves in [`my_gpt.py`](my_gpt.py). The book's [`build.py`](build.py) remains a separate reference implementation. Our question was: can a tiny GPT learn the shape of seven Harry Potter books, and which changes actually make its continuations more readable?

## Data and BPE

- The source was the combined seven-book text at `~/Desktop/NanoGPT/data/harry_potter.txt` (6,296,935 characters). We split it **before training BPE**: first 90% for training (5,667,241 characters), last 10% for validation (629,694 characters). The validation portion is mostly the end of Book 7, so it is a held-out continuation of the same corpus, not a random mixture of all books.
- Our byte-level BPE starts with 256 byte tokens and learns frequent pair merges **only from training text**. `encode()` produces subword IDs; `decode()` reconstructs the original text exactly. It is case-sensitive: `Harry` and `harry` can have different IDs or splits.
- With a 512-token vocabulary, we had 3,397,182 train and 376,104 validation tokens. With 2,048 tokens, the same text became 2,631,736 train and 294,701 validation tokens. For example, the newer tokenizer can encode `Harry opened the door.` as `Harry | space | opened | space | the | space | door | .`.
- Changing BPE changes every token ID, the output vocabulary, and the meaning of loss *per token*. The 512-token checkpoint cannot be used with the 2,048-token tokenizer.

## What we implemented

For each sampled window, `x` contains `block_size` BPE IDs and `y` contains the same IDs shifted left by one position. A batch contains many such windows. We kept train and validation samplers separate.

The GPT is:

```text
token IDs → token embeddings + learned position embeddings
          → repeated pre-norm blocks:
              x = x + causal multi-head attention(LayerNorm(x))
              x = x + feed-forward network(LayerNorm(x))
          → final LayerNorm → LM head → vocabulary logits
```

`forward(x, targets)` runs that computation once and, when targets are supplied, returns next-token cross-entropy loss. `generate(prompt, n)` repeatedly calls `forward()`, samples one next token, appends it, and keeps only the latest `block_size` tokens in its context. Causal masking prevents a position from reading future tokens.

The token embedding table and LM head **share the same weight matrix**. The table maps an input ID to a vector; the LM head compares the final hidden vector against vocabulary vectors to score possible next tokens. For the 2,048 × 256 table, tying saves 524,288 separate parameters.

We used Normal(0, 0.02) initialization for linear/embedding weights, zero linear biases, LayerNorm scale 1 and bias 0, AdamW, gradient clipping at norm 1, linear learning-rate warmup followed by cosine decay, fixed validation windows, sample printing, and checkpoints. The current AdamW call uses its default weight decay for all parameters; Chapter 6 will split parameters into separate decay groups. Checkpoints include model and optimizer states, step, config, tokenizer merges, and (for newer CPU/Modal runs) random-number states.

## Training results

| Experiment | Model and data | Steps | Train loss | Validation loss | What we saw |
|---|---|---:|---:|---:|---|
| CPU prototype | 470,528 params; 512 BPE tokens; context 64 | 3,000 | 2.6537 | 2.6561 | Dialogue-shaped text, many malformed words. |
| Same CPU model, resumed | Same tokenizer and model | 18,000 total | 2.3192 | 2.3403 | Loss kept falling; samples remained largely incoherent. |
| Modal A100 model | 3,716,608 params; 2,048 BPE tokens; context 128; 4 layers × 256 dimensions | 5,000 | 2.1644 | 2.3123 | More recognizable phrases and dialogue; grammar/meaning still imperfect. |

The A100 run used 200 warmup steps, then cosine decay from `3e-4` to `3e-5`; its 5,000 optimizer updates took about 157 seconds of GPU training. Train/validation gap grew to about 0.15 nats per token, although validation loss was still decreasing at the end. The [A100 loss/LR graph](outputs/my_gpt_2048/modal_runs/gpt-2048-20260922-185701/loss_and_lr.png), [samples](outputs/my_gpt_2048/modal_runs/gpt-2048-20260922-185701/samples.json), and [checkpoint](outputs/my_gpt_2048/modal_runs/gpt-2048-20260922-185701/checkpoint.pt) are saved locally. Our [Modal script](modal_gpt_a100.py) used the dedicated `qwen-tts-lab/scripts/modal-tts` wrapper.

**Do not compare the two vocabularies' raw loss-per-token numbers as if they were on the same scale.** An approximate normalization using each tokenizer's validation token count and the same validation text gives 1.398 nats/character for the CPU run versus 1.082 for the A100 run. Those are estimates from sampled validation windows, not a full-corpus evaluation; several things changed at once (vocabulary, model size, context, and training setup), so they do not isolate which change helped.

## Two follow-up experiments

We used the *same* A100 checkpoint, three prompts, and fixed sampling seeds in [`experiment_decoding_and_context.py`](experiment_decoding_and_context.py):

1. **Decoding:** temperature 0.8 with unrestricted sampling, top-k 40, and top-p 0.9. Filtering sometimes removed malformed fragments—for `The door opened and`, unrestricted sampling began `the desef`, while top-k began `the crowd was clattering...`. Other continuations were still awkward. This changes token selection, not model weights or validation loss; three prompts are too few to claim a consistent quality gain.
2. **Context:** on 512 *matched held-out next tokens*, we compared the same model with the last 32, 64, or 128 input tokens. Mean losses were **2.3211**, **2.3256**, and **2.3267** nats/token, respectively. Paired differences from 128-token context were smaller than their roughly 0.011–0.012 standard errors. We therefore saw no measurable benefit from more context for these *single next-token predictions*. That does not establish whether a future 256-token model would help long-form coherence.

The [full experiment results](outputs/my_gpt_2048/modal_runs/gpt-2048-20260922-185701/decoding_and_context_experiments.json) include every generated continuation and the context-loss estimates.

## Faster generation: KV cache and split-KV decoding

Originally, `generate()` reran the whole context through the model for every new token. That means token 100 recomputed attention for the first 99 tokens again. `my_gpt.py` now keeps a **KV cache**. The prompt runs once (**prefill**). After that, each step feeds only the newest token (**decode**): one query attends to the cached keys and values of every layer. The motivation and the GPU view are in the [Chapter 8 FlashAttention notes](../08_flash-attention-and-tiled-kernels/FLASH_ATTENTION_EXPLAINED.md#8-inference-decoding-and-flash-decoding).

- `model.generate(prompt, n)` uses the cache by default. `use_kv_cache=False` restores the old recompute path.
- `decode_splits=k` computes each decode step with **split-KV attention** (the Flash-Decoding algorithm). The cache is cut into `k` chunks. Each chunk produces a partial softmax state (max, sum, unnormalized output). The chunks are then merged with the same rescaling rule as FlashAttention's streaming softmax. On a GPU, the chunks could run on different SMs. Here it is a PyTorch reference of the algorithm, not a custom kernel.
- Prefill still uses the configured attention path. With `attention_impl="flash"` on a GPU, that is the fused FlashAttention kernel. The mHC routes and RoPE positions both work with the cache.
- Once the context passes `block_size`, the window slides, and positions restart at 0 for the cropped window, exactly as in the original loop. Cached keys are then stale, so each step re-prefills. This keeps outputs identical to the old path, but the speedup shrinks beyond the window.

[`benchmark_kv_cache.py`](benchmark_kv_cache.py) measured this on CPU (Apple M4 Pro, FP32, one prompt of 11 tokens, single run):

| Checkpoint | New tokens | Recompute | KV cache | KV + split-KV ×4 | Same tokens |
|---|---|---|---|---|---|
| mHC + Flash 30k (`best.pt`, run with SDPA on CPU) | 100 | 0.94 s | 0.28 s (**3.3×**) | 0.27 s | yes |
| Chapter 7 RMSNorm + SwiGLU | 117 (fills the 128 window) | 0.38 s | 0.08 s (**4.9×**) | 0.08 s | yes |
| Chapter 7 RMSNorm + SwiGLU | 250 (window slides) | 1.02 s | 0.72 s (1.4×) | 0.73 s | yes |

Cached logits match a full forward pass to about `1e-5` (FP32). With the same seed, all three modes sampled identical tokens. Split-KV adds no speed on CPU. Its GPU benefit appears when batch × heads is smaller than the number of SMs, which this benchmark does not measure. These are single wall-clock runs on one machine, not a careful latency study. [`tests/test_my_gpt_kv_cache.py`](tests/test_my_gpt_kv_cache.py) checks cached logits against full recompute for learned/RoPE positions, manual/SDPA attention, and mHC, with and without split-KV.

## Takeaways for Chapter 6

- A lower loss and a plausible dialogue format do not guarantee coherent paragraphs. Check both held-out loss and fixed-prompt generations.
- A sampling rule can improve the surface of generated text without teaching the model anything new. Keep training and decoding experiments distinct.
- Our measured 32/64/128-token context results do not justify prioritizing a longer context for this model yet.
- Chapter 6's next controlled changes are **weight-decay parameter groups** (matrices versus biases/LayerNorm) and **scaled residual-output initialization**. Compare them against this prototype with the same data, tokenizer, budget, and validation windows.
