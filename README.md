# Under the Hood

> **📝 About this copy — Abhinav Verma's study repo**
>
> This is my personal working copy of [mechramc/Under-the-hood](https://github.com/mechramc/Under-the-hood), the code companion to Ramchand Kumaresan's book *Under the Hood*. I am working through the book project by project. This repo records how I understood it, what I built myself, and what I learned.
>
> 📖 **If this repo helps you, please support the author and buy the book: [leanpub.com/under-the-hood](https://leanpub.com/under-the-hood).** Reading the book alongside the code is how I learned all of this. My notes here don't replace it.
>
> All credit for the book, the project structure, and the reference code (`build.py`, `break_it.py`, per-step files) goes to the original author. My additions are my own interpretation and experiments. They are not part of the official companion, and any mistakes in them are mine.
>
> **What I added so far:**
> - **Project 5:** [`my_gpt.py`](projects/05_your-gpt-from-a-blank-file/my_gpt.py), a GPT written from a blank file. It comes with tests, Modal A100 training scripts, decoding and context-length experiments, and my [learning notes](projects/05_your-gpt-from-a-blank-file/NOTES.md).
> - **Project 6:** A100 BPE runs comparing the baseline with nanoGPT-style refinements, including a 10k-step continuation.
> - **Project 7:** matched comparisons of SwiGLU vs GELU, RMSNorm vs LayerNorm, and RoPE vs learned positions.
> - **Project 8:** an mHC + flash-attention experiment trained for 30k steps ([write-up](projects/08_flash-attention-and-tiled-kernels/EXPERIMENT_MHC_FLASH.md)), [FlashAttention explained from the GPU up](projects/08_flash-attention-and-tiled-kernels/FLASH_ATTENTION_EXPLAINED.md), and KV-cache / Flash-Decoding generation for my GPT.
> - **Project 9:** real-web pretraining. My GPT (51M params) trained on 1B tokens of FineWeb-Edu with a sharded memmap pipeline, gradient accumulation, validation bits-per-byte, an LR sweep, and a repeated-data BREAK IT ([notes](projects/09_pretraining-on-the-real-web/NOTES.md)).
>
> Look for the sections I added to each project's README. They are written as "we trained / we measured" results, with figures. Every result comes from a single seed on small models. Treat them as learning notes, not as benchmarks.
>
> Want the real thing? **[Get the book on Leanpub](https://leanpub.com/under-the-hood)** and use the [original repo](https://github.com/mechramc/Under-the-hood).

---

### Build Every Layer of a Large Language Model from Scratch

> *A practical manual for understanding how modern language models are built, where they fail, and how to reason about their behavior like an engineer instead of a spectator.*

**By Ramchand Kumaresan**

**Build it. Break it. Measure it.**

📖 **Buy the book at [leanpub.com/under-the-hood](https://leanpub.com/under-the-hood)** — this repository is the code companion.

---

## What This Repo Is

This is the runnable code companion for the book *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Each of the book's **36 projects** has its own folder under `projects/`, with a runnable `build.py`, a `break_it.py` experiment, per-step reference files, and captured outputs.

You can do the entire book by reading the book and typing the code yourself — that's the recommended path. But when you get stuck, or want to compare your code to a reference, or want to skip ahead and read the working version before building it yourself, this repo is where you look.

The book itself (PDF + EPUB) is distributed separately via Leanpub. This repo intentionally does **not** ship the book — it ships the code.

---

## Quick Start

```bash
git clone https://github.com/mechramc/Under-the-hood.git
cd Under-the-hood

# Create an environment (any tool works; this is one option)
python -m venv .venv && source .venv/bin/activate     # or .venv\Scripts\activate on Windows

# Install
pip install -r requirements.txt

# Run any project's main script
python projects/01_the-learning-machine/build.py --tiny
```

Full setup instructions: see [`setup/`](setup/).

---

## The 36 Projects

Read the book and the per-project READMEs in order. Each project builds on the previous one.

### Foundations

| #  | Project | Folder |
|----|---------|--------|
| 1  | The Learning Machine — scalar autograd, neurons, MLP, training loop. | [`projects/01_the-learning-machine`](projects/01_the-learning-machine) |
| 2  | Predicting the Next Character — bigram counts → learned embeddings → NLL. | [`projects/02_predicting-the-next-character`](projects/02_predicting-the-next-character) |
| 3  | Building a Tokenizer — BPE from scratch; vocab size as a tunable knob. | [`projects/03_building-a-tokenizer`](projects/03_building-a-tokenizer) |
| 4  | Attention from Scratch — Q/K/V, scaled dot-product, masking, multi-head. | [`projects/04_attention-from-scratch`](projects/04_attention-from-scratch) |

### Building a GPT

| #  | Project | Folder |
|----|---------|--------|
| 5  | Your GPT from a Blank File — the smallest complete system. | [`projects/05_your-gpt-from-a-blank-file`](projects/05_your-gpt-from-a-blank-file) |
| 6  | From Prototype to nanoGPT — side-by-side with the reference. | [`projects/06_from-prototype-to-nanogpt`](projects/06_from-prototype-to-nanogpt) |
| 7  | The Details That Matter — norms, activations, positional encodings. | [`projects/07_the-details-that-matter`](projects/07_the-details-that-matter) |
| 8  | Flash Attention and Tiled Kernels — memory-efficient attention. | [`projects/08_flash-attention-and-tiled-kernels`](projects/08_flash-attention-and-tiled-kernels) |

### Training at Scale

| #  | Project | Folder |
|----|---------|--------|
| 9  | Pretraining on the Real Web — FineWeb-EDU, mixed precision, val-bpb. | [`projects/09_pretraining-on-the-real-web`](projects/09_pretraining-on-the-real-web) |
| 10 | Data Curation and Contamination — what makes good pretraining data. | [`projects/10_data-curation-and-contamination`](projects/10_data-curation-and-contamination) |
| 11 | Training Debugging: Spikes, NaNs, and Profiling. | [`projects/11_training-debugging-spikes-nans-and-profiling`](projects/11_training-debugging-spikes-nans-and-profiling) |
| 12 | Distributed Training: FSDP and ZeRO (single-box proxy). | [`projects/12_distributed-training-fsdp-and-zero-single-box-proxy`](projects/12_distributed-training-fsdp-and-zero-single-box-proxy) |

### Inference and Serving

| #  | Project | Folder |
|----|---------|--------|
| 13 | Fast Inference: The KV Cache. | [`projects/13_fast-inference-the-kv-cache`](projects/13_fast-inference-the-kv-cache) |
| 14 | Speculative Decoding. | [`projects/14_speculative-decoding`](projects/14_speculative-decoding) |
| 15 | Grouped Query Attention. | [`projects/15_grouped-query-attention`](projects/15_grouped-query-attention) |
| 16 | Long-Context Extension (RoPE, YaRN, NTK-Aware). | [`projects/16_long-context-extension-rope-yarn-ntk-aware`](projects/16_long-context-extension-rope-yarn-ntk-aware) |
| 17 | Production Serving: Continuous Batching and PagedAttention. | [`projects/17_production-serving-continuous-batching-and-pagedattention`](projects/17_production-serving-continuous-batching-and-pagedattention) |

### Scaling and Experimentation

| #  | Project | Folder |
|----|---------|--------|
| 18 | Mixture of Experts — router, top-k, expert utilization, load balancing. | [`projects/18_mixture-of-experts`](projects/18_mixture-of-experts) |
| 19 | Scaling Laws — depth sweeps, compute-optimal training. | [`projects/19_scaling-laws`](projects/19_scaling-laws) |
| 20 | Autonomous Experimentation — point an agent at `program.md`. | [`projects/20_autonomous-experimentation`](projects/20_autonomous-experimentation) |

### Aligning and Evaluating

| #  | Project | Folder |
|----|---------|--------|
| 21 | Fine-Tuning and Instruction Tuning — SFT, conversational tuning, LoRA. | [`projects/21_fine-tuning-and-instruction-tuning`](projects/21_fine-tuning-and-instruction-tuning) |
| 22 | Evaluation Methodology. | [`projects/22_evaluation-methodology`](projects/22_evaluation-methodology) |
| 23 | Reward Models and RLHF — reward dataset, RM training, GRPO, KL leash. | [`projects/23_reward-models-and-rlhf`](projects/23_reward-models-and-rlhf) |
| 24 | DPO and Preference Optimization. | [`projects/24_dpo-and-preference-optimization`](projects/24_dpo-and-preference-optimization) |

### Reasoning, Tools, and Retrieval

| #  | Project | Folder |
|----|---------|--------|
| 25 | Test-Time Reasoning (CoT, Self-Consistency, Best-of-N). | [`projects/25_test-time-reasoning-cot-self-consistency-best-of-n`](projects/25_test-time-reasoning-cot-self-consistency-best-of-n) |
| 26 | Tool Use and Function Calling. | [`projects/26_tool-use-and-function-calling`](projects/26_tool-use-and-function-calling) |
| 27 | Quantization and Deployment — FP32 → INT8 → INT4 → GGUF. | [`projects/27_quantization-and-deployment`](projects/27_quantization-and-deployment) |
| 28 | Retrieval-Augmented Generation. | [`projects/28_retrieval-augmented-generation`](projects/28_retrieval-augmented-generation) |

### Beyond the Transformer

| #  | Project | Folder |
|----|---------|--------|
| 29 | Multimodal: A Tiny Vision-Language Model. | [`projects/29_multimodal-a-tiny-vision-language-model`](projects/29_multimodal-a-tiny-vision-language-model) |
| 30 | Non-Transformer Architectures (Mamba, RWKV). | [`projects/30_non-transformer-architectures-mamba-rwkv`](projects/30_non-transformer-architectures-mamba-rwkv) |
| 31 | Diffusion and Non-Autoregressive Decoding — iterative unmasking; confidence-ranked commits; the speed-vs-coherence dial. | [`projects/31_diffusion-and-non-autoregressive-decoding`](projects/31_diffusion-and-non-autoregressive-decoding) |

### Modular Composition

| #  | Project | Folder |
|----|---------|--------|
| 32 | Layer Freezing and Transfer — freeze ratios, CKA, training speed vs forgetting. | [`projects/32_layer-freezing-and-transfer`](projects/32_layer-freezing-and-transfer) |
| 33 | Fusing Independently Trained Specialists — shared base, specialist heads, router. | [`projects/33_fusing-independently-trained-specialists`](projects/33_fusing-independently-trained-specialists) |
| 34 | The Interface Specification — machine-readable contracts; loud vs silent failures. | [`projects/34_the-interface-specification`](projects/34_the-interface-specification) |
| 35 | Incremental Assembly — order independence; what "modular" actually means. | [`projects/35_incremental-assembly`](projects/35_incremental-assembly) |
| 36 | Your Architecture — final project; build one of your own. | [`projects/36_your-architecture`](projects/36_your-architecture) |

---

## The Method: Build → Break → Measure

Each project follows the same disciplined rhythm:

1. **Hook** — the question the project answers.
2. **The Concept** — the idea in plain English, before any code.
3. **Why It Matters** — what fails in the real world without this piece.
4. **The Build** — step-by-step implementation, written so shapes stay honest.
5. **Break It** — a deliberate sabotage of the system, with predictions and observations.
6. **Optional Homework** — full lab, proxy lab (for limited hardware), result-guided version.
7. **Questions To Answer** — what you should be able to defend after finishing.
8. **Go Further** — research anchors and next directions.
9. **What You Now Know** — the explicit deltas in your mental model.

Reading the book without breaking the code is half the experience. The breaks are where the lessons actually live.

---

## Who This Is For

- **Engineers** who can ship software but feel like LLM internals are wizardry happening behind an API.
- **Researchers** who want a working mental model of the full stack before reading their hundredth attention paper.
- **Students** who learned PyTorch from notebooks and want to know what a training run *actually* does.
- **Practitioners** tuning, deploying, or fine-tuning models who keep hitting failure modes they cannot explain.

You should be comfortable with Python and have seen a tensor before. Everything else is built up from scratch.

---

## Hardware Notes

Every project supports a `--tiny` mode that runs on CPU in under 60 seconds. The full lab versions have realistic compute requirements:

| Project Range | Realistic Hardware for the Full Lab |
|---|---|
| Projects 1–7 | CPU or any laptop GPU |
| Projects 8–17 | One consumer GPU (8–24 GB) or short cloud rental |
| Projects 18–27 | One consumer GPU; LoRA paths designed for tight VRAM |
| Projects 28–35 | One GPU for full labs; proxy versions documented for all |

See [`setup/03_gpu-and-hardware-tiers.md`](setup/03_gpu-and-hardware-tiers.md) for the detailed breakdown.

---

## Repository Layout

```
Under-the-hood/
├── README.md            # this file
├── LICENSE              # MIT
├── pyproject.toml       # deps + pytest + ruff + pyright config
├── requirements.txt     # pinned dependencies
├── conftest.py          # shared pytest fixtures (tiny_model_config, seed control)
├── setup/               # environment, dependencies, GPU tiers, datasets
├── tools/               # extract_code.py (regenerates per-project files from book source)
└── projects/            # 35 project folders (NN_slug/)
    └── NN_slug/
        ├── README.md    # prose: Hook, Concept, Why It Matters, outputs
        ├── build.py     # canonical runnable
        ├── break_it.py  # the sabotage experiment
        ├── step_*.py    # one file per Build step (pedagogical references)
        ├── outputs/     # captured loss curves, samples, benchmarks
        └── tests/       # pytest smoke + unit tests
```

---

## Running Tests

```bash
pytest                                  # all projects, default tier 1 + tier 2
pytest projects/01_the-learning-machine # one project
pytest -m "not slow and not gpu"        # skip the heavy ones (this is the default)
UTH_RUN_SLOW=1 pytest                   # include @pytest.mark.slow tests
```

---

## Contributing

Issues and corrections are welcome. Open a GitHub issue with:
- The project number and file
- What you observed
- What you expected

For substantive contributions (new BREAK IT experiments, additional benchmarks, better proxy datasets), open a pull request and link the relevant book chapter.

---

## Philosophy

> *Build it. Break it. Measure it.*

This is not a tagline. It is the entire epistemology of the book.

- **Build it** because reading is not understanding. Understanding is being able to reproduce.
- **Break it** because a working system tells you what it does. A broken system tells you why it has to be that way.
- **Measure it** because intuition without numbers is the most expensive failure mode in machine learning.

Modern LLMs are not magic. They are a stack of carefully-tuned engineering decisions, each one earned by a specific failure that someone, somewhere, had to debug at 2am. This book makes you the person who debugged it.

---

## About the Author

**Ramchand Kumaresan** writes systems-level engineering manuals for AI practitioners. His work focuses on the parts of the stack that are usually skipped: the failure modes, the diagnostics, the parts of the codebase that exist as scar tissue from a real outage. *Under the Hood* is the first book in a series on building production-grade machine learning systems from first principles.

Contact: <ramchand@murailabs.com>

---

## License

Code in this repository is released under the [MIT License](LICENSE). The book text (PDF / EPUB, distributed via Leanpub) is copyrighted separately — see the copyright page in the book.

You are free to copy, modify, and redistribute the code in this repository, with attribution appreciated but not required.

---

## Citation

If you use this work in academic writing:

```bibtex
@book{kumaresan2026underthehood,
  author    = {Kumaresan, Ramchand},
  title     = {Under the Hood: Build Every Layer of a Large Language Model from Scratch},
  year      = {2026},
  publisher = {Leanpub},
  url       = {https://leanpub.com/under-the-hood}
}
```

---

*If you finish this book and can still be surprised by the behavior of an LLM, you did not break enough of it.*
