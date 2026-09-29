# Project 9: Pretraining On The Real Web

## Hook

Why does training on Shakespeare feel clean and understandable, then the moment you switch to real web text everything turns into engineering: data shards, special tokens, memory mapping, mixed precision, gradient accumulation, learning rate sweeps, validation bits-per-byte, and loss curves that seem to argue with each other? You built a model already. It learned. So why is the next step not "same thing, bigger file"?

Because a real language model is more than a neural net. It is a machine for moving absurd amounts of text through a GPU without wasting compute, corrupting documents, blowing up memory, or fooling you with a training curve that looks healthy right up until it starts memorizing.

The first time I tried to scale from a single-file toy dataset to real web text I assumed the bottleneck would be the model. It was not. The bottleneck was every step around the model: how the tokens were stored, how they were sampled, how the loss was measured, and how I decided when to stop. The model itself was almost the easy part.

## The Concept

Your Project 5 loader was like cooking from one giant pot. You had one text file, you chopped it into tokens, you grabbed random windows, and you fed them to the model. That works because the whole system is tiny — the data is small, the rules are loose, and if you waste a little compute, nobody cares.

Real pretraining is closer to running a warehouse. You are not serving one pot of soup. You are receiving truckloads of ingredients, sorting them, packaging them, labeling them, putting them on shelves, and making sure the kitchen can grab the next box instantly without unpacking the whole building every time. If one part of that pipeline is sloppy, your expensive hardware sits idle.

That is the real project — not "train on the web" in the vague sense, but train on the web with systems that respect scale. Let's name the parts first.

A **dataset shard** is one chunk of already-processed training data saved to disk. Instead of storing one giant token file, you split it into many manageable pieces. Think shipping containers, not one mountain of loose cargo.

**Pre-tokenized data** means the raw text has already been converted into token IDs before training starts. You pay the tokenization cost once, not every batch forever. The reason this matters is that tokenization is deterministic and expensive: doing it once and storing the result as shards means the GPU never waits on string processing. Without that separation, data loading becomes the bottleneck, and a GPU sitting idle while text gets tokenized is exactly the kind of waste that makes long training runs unpredictable.

I lived through this exact failure in some early fine-tuning runs. On a couple of experiments at the few-hundred-million-parameter range I had not pre-tokenized cleanly, and the GPU utilization kept dipping below 60 percent for reasons I could not explain until I instrumented the data loader. The string processing was the bottleneck, not the model. Once the tokens were on disk as integers, the same GPU pinned at 90-plus.

A **memory-mapped file** is a file on disk that your program can treat almost like an array in memory. The operating system only loads the pieces you touch. Think of a giant book where you can open directly to page 812,441 without copying the whole book onto your desk. This matters for GPU utilization because if loading data into RAM requires copying large portions of a file at startup, you pay that cost before training even begins. And on datasets larger than physical memory, that approach can be impossible entirely.

**Document boundaries** are the places where one source text ends and the next begins. If you ignore them, your model may learn fake transitions, like "end of Stack Overflow answer immediately followed by random Wikipedia heading," as if they were natural language. So you insert a special token that says, in effect, "this document ended; a new one begins here."

**Gradient accumulation** means pretending you had a larger batch size than fits in memory. You run several small batches, add up their gradients, and only then update the weights. It is like taking four small notes before making one decision, instead of making a decision after every single note. This matters because batch size is more than a memory constraint. It is connected to training stability. Larger effective batches tend to produce more reliable gradient estimates, and accumulation lets you reach a useful effective batch size even when your GPU has room for only a fraction of it.

**Mixed precision** (introduced in **Project 6: From Prototype to nanoGPT**) means doing much of the math in 16-bit formats like FP16 or BF16 to reduce memory traffic and speed up training, provided numerical stability is maintained.

**Bits-per-byte**, usually written **bpb**, is a training metric that asks: how many bits of surprise does the model need, on average, to encode the next byte of text? Lower is better. If the model predicts well, the text looks less surprising, so the number drops.

There is one more thing that everyone talks around because they prefer big architectural words to boring operational truth: pretraining is a throughput problem. Yes, it is learning, but it is also this. Can you keep the GPU busy? Can you feed it clean data? Can you measure progress honestly? Can you stop before "getting better at the training set" turns into "getting worse at the real world"?

Most beginner explanations of pretraining are honestly terrible because they skip the throughput question entirely. They linger on the architecture and treat the pipeline as plumbing. The pipeline is the project. The architecture is a few hundred lines.

That last question matters enough to state plainly. Training loss and validation loss are not the same thing. **Training loss** measures how wrong the model is on data it is being trained on. **Validation loss** measures how wrong it is on held-out data it did not train on. If both fall together, good. If training loss falls while validation loss stalls, be cautious. If training loss keeps falling while validation loss rises, you are watching memorization win. That is overfitting, and the model is getting better at the wrong job.

Here is the intuition before we earn any metric. Imagine studying for an exam by reading the answer key to one practice sheet over and over. Your score on that sheet keeps rising. Your score on a new sheet stops improving, then gets worse because you start overcommitting to quirks of the first one. That is exactly what your loss curves will show when you train too long.

Now the metric. If the model assigns probability \(p\) to the correct next token, the surprise is related to \(-\log p\). Cross-entropy loss adds that surprise across the sequence. Bits-per-byte is that surprise normalized in a way that is easy to compare across text corpora. You do not need the full formula first — you need the meaning: lower bpb means the model compresses text better because it predicts text better. That is why bpb is a more honest signal than training loss alone. Training loss answers "how well did I memorize what I practiced on?" while val_bpb answers "how well do I handle new examples from the same world?"

The learning rate matters for the same reason a thermostat matters. Too low, and nothing moves. Too high, and the system overshoots, oscillates, or explodes. There is usually a middle band where training drops quickly and stays stable. That is why you sweep learning rates instead of trusting your first guess.

This is where your mental model of LLM training changes. The model is not the whole story anymore. The system is: storage layout, numeric formats, batch simulation, schedules, metrics, stopping rules. Figure 8.1 shows how all those stages connect into a single pretraining pipeline.

![Figure 8.1. Real-web pretraining is a pipeline: raw crawl data gets filtered, deduplicated, tokenized, sharded, memory-mapped, and then measured with held-out validation during GPU training.](figures/fig_pretraining_data_pipeline.png)

## Why It Matters

All of these pieces connect to a single underlying pressure: keeping the GPU doing useful work on clean data, measuring that work honestly, and stopping at the right time.

Pre-tokenized shards are the reason the GPU never waits on string processing. Tokenization is expensive and deterministic, so doing it once and storing integer arrays on disk means the training loop simply reads numbers: fast, repeatable, and memoryless. Without that separation, every batch would trigger redundant text processing, making training slower and non-reproducible across runs.

Memory mapping is what makes pre-tokenized shards actually usable at scale. The operating system pages in only the slices you touch, so your data loader can handle datasets far larger than RAM without copying huge files upfront. This is directly connected to GPU utilization: if the data loading pipeline stalls, the GPU sits idle, and idle GPU time is the most expensive form of waste in a long training run.

Document boundary tokens protect the targets the model is being trained against. Without them, the next-token prediction task contains a steady stream of nonsense transitions. The model is penalized for not predicting a random Wikipedia sentence following an unrelated Stack Overflow answer. That is label noise, and enough of it corrupts the gradient signal in ways that are hard to diagnose later.

Gradient accumulation is the bridge between what fits in GPU memory and what the training dynamics actually need. Batch size is more than a memory question. It shapes gradient variance. Small batches produce noisy updates; larger effective batches produce steadier ones. Accumulation lets you simulate a batch size that would otherwise be physically impossible on your hardware.

Mixed precision works because modern GPUs are dramatically faster at 16-bit arithmetic than 32-bit, and the precision cost is manageable when implemented carefully. BF16 in particular keeps a wide enough exponent range to stay numerically stable through most of training. The combined effect of FP16/BF16 and memory mapping is that a training run that would otherwise require a larger or more expensive machine can often complete on one.

The val_bpb metric is more honest than training loss because it measures what you actually care about: whether the model is learning patterns that generalize beyond the data it was shown. A steadily falling training loss can coexist with a model that is quietly memorizing, and without a held-out metric like val_bpb you have no alarm. This is not a nice extra. It is the control panel.

A learning rate sweep exists because the right learning rate depends on nearly everything: model depth, width, batch size, optimizer, schedule, data, precision, and token budget. There is no single correct value to guess. Running four candidates on the same token budget is cheap compared to spending a full run on a bad setting. And crucially, "training is unstable" is not a diagnosis. A sweep gives you the curve shapes to reason from.

Across a long set of training runs I logged at scale, the single most expensive class of mistakes was committing too much wall-clock to a learning rate that the first 500 steps had already told us was wrong. Sweeps look slow on paper. They are far cheaper than a bad full run.

Taken together, these choices explain why benchmark comparisons between models are so hard to trust at face value. "Model A beat Model B" often means one team chose a better learning rate, one stopped at the right point, one had a cleaner data pipeline, or one matched batch size and schedule better to their token budget. The engineering layer is not separate from the modeling quality. It is underneath it.

This is also the point where toy intuition fails in a useful way. On Shakespeare, almost anything reasonable appears to work. On real web text, the cracks show. That is good. Cracks are where engineering starts.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/09_pretraining-on-the-real-web/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/09_pretraining-on-the-real-web/build.py --full

# The BREAK IT experiment:
python projects/09_pretraining-on-the-real-web/break_it.py
```

## Outputs

**Our run:** we built this chapter's pipeline around our own Chapter 5 GPT and trained a 51 M-parameter model on 1 B tokens of FineWeb-Edu on one A100. It covers pre-tokenized `uint16` shards, memory-mapped loading, EOT document boundaries, gradient accumulation, BF16 with FlashAttention, validation bits-per-byte, a 4-way LR sweep, and a repeated-data BREAK IT. Results, figures, and learnings are in [`NOTES.md`](NOTES.md). The book's Step 2 exercise (annotate the data pipeline like a systems engineer), with diagrams and a walkthrough of `np.memmap`, is in [`PIPELINE_ANNOTATED.md`](PIPELINE_ANNOTATED.md).

```bash
# CPU smoke test on any local text file
.venv/bin/python projects/09_pretraining-on-the-real-web/prepare_fineweb.py --source text \
    --text-files book.txt --out-dir /tmp/shards --shard-tokens 400000 --val-tokens 100000
.venv/bin/python projects/09_pretraining-on-the-real-web/pretrain.py --data-dir /tmp/shards \
    --out-dir /tmp/run --preset tiny --total-tokens 409600 --batch-tokens 4096 --micro-batch 16

# A100 on Modal: prepare | smoke | sweep | train | breakit | sample
modal run projects/09_pretraining-on-the-real-web/modal_pretrain.py --mode sweep
```

## Read in the book

This project is Chapter 9 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
