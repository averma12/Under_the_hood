# Project 10: Data Curation and Contamination

## Hook

In **Project 9: Pretraining on the Real Web**, you pulled FineWeb-EDU off Hugging Face, fed it into your training loop, and watched the loss come down. Most readers stop there. The data was already filtered by someone else, the file format was clean, the dataset card promised "high-quality educational content," and the model trained without exploding. That feels like enough.

It is not.

This chapter asks a question that almost no one asks before training: what is actually in this data, and is any of it sabotaging the evaluation you are about to run? By the end you will know, for the specific corpus you are training on, what fraction is duplicate, what fraction is multilingual junk that your tokenizer cannot handle, what fraction reads like template spam, and (most importantly) what fraction silently leaks the answers to the benchmark you were going to use to measure progress.

## The Concept

Imagine you are a chef who orders a shipment of ingredients from a wholesale supplier, opens one box at the top of the pile, sees clean produce, and decides the whole shipment is fine. You then cook a meal for guests using ingredients from the bottom of the pile that you never inspected. Sometimes the meal is fine. Sometimes a guest gets sick, and you have no way to trace which ingredient was responsible because you never looked at any of them. Pretraining data is the same. The Hugging Face card is the top of the box. The bottom is whatever the crawler swept off the web on the day it ran.

The analogy I actually use in my head for this comes from a different industry. In a past role I ran a multi-state SaaS platform with hundreds of thousands of customer records spread across dozens of sites. Some of those records had been silently corrupted by a third-party importer years before anyone noticed. The corruption did not crash anything. It just slowly biased every report we ran until somebody happened to spot-check the source. Pretraining data is the same hazard at a different scale. The model does not crash on bad data. It just learns from it, and the bias does not become visible until you have already shipped.

Now picture a second analogy. You are building a house, and someone hands you a foundation that was poured by a contractor you have never met. The contractor's website has nice photos. The foundation looks flat. You build twelve stories on top of it before discovering that one corner was poured over an unmarked sinkhole. The whole building does not immediately fall down. It just tilts a little more every year. Training a model on uninspected data is the same. The model does not refuse to learn. It learns whatever is there, and what is there might include thousands of copies of the same SEO spam page, machine-translated text that your tokenizer butchers, and verbatim copies of the test set you were planning to evaluate on.

Real pretraining data curation comes down to four operations, and each one is a measurable quantity rather than a vibe. The first is **quality filtering**: scoring each document on signals like length, repetition within the document, language identification confidence, and a small bag of heuristic features, then dropping documents below some threshold. The second is **deduplication**: removing copies of the same document, and near-copies that differ in a few words, so the model does not memorize one essay by seeing it forty times. The third is **mixing**: deciding what fraction of training tokens come from each source (web, code, math, books) and how those fractions should change as training progresses. The fourth is **decontamination**: removing any training document that contains a long verbatim chunk of an evaluation set, so the benchmark you are about to report is measuring something other than your model's ability to recall training data.

Across a long set of fine-tuning experiments I ran multiple times on different data mixes for the same model family, the lesson that kept coming back was that the mix mattered more than people expected. A 410M model trained on one mix and a 410M model trained on a slightly different mix produced different downstream behavior on identical compute. Each of those runs was, in part, a small reminder that "the data" is not a single thing.

Each of these has a specific number attached. Quality filtering produces a document score and a threshold. Deduplication produces a duplication rate, typically 20 to 40 percent for raw crawl data and 5 to 10 percent for already-curated corpora. Mixing produces a vector of per-source weights that sum to one. Decontamination produces a count of leaked documents and an overlap rate.

A new term that will appear throughout this chapter: a **shingle**. A shingle is a fixed-length sliding window of tokens or characters from a document. If your shingle size is 5 and your document is "the cat sat on the mat today," the shingles are "the cat sat on the", "cat sat on the mat", "sat on the mat today". Shingles are the unit of comparison that almost every deduplication and contamination check is built on, because two documents that share many shingles are similar in a way that the human eye recognizes immediately.

Two more terms, defined here so they can be used freely later. **MinHash** is a hash-based trick for estimating how many shingles two documents have in common without actually comparing every shingle against every shingle, which would cost quadratic time in the corpus. **LSH**, short for **locality-sensitive hashing**, is a follow-up trick that groups documents into buckets so that similar documents land in the same bucket and you only have to compare documents within a bucket, which cuts the comparison count from billions to millions.

We will build both. The construction is small enough to hold in your head.

A last term worth defining now: **n-gram overlap**. An n-gram is a contiguous sequence of n tokens, almost the same idea as a shingle. Contamination is usually measured by counting how many 13-grams from an evaluation set also appear verbatim somewhere in the training data. Thirteen is large enough that a chance match is essentially impossible (the probability of two arbitrary 13-token spans matching by accident is astronomically small) and small enough that paraphrased contamination still leaves a fingerprint. Some labs prefer 8-grams, some prefer 50-character spans. Thirteen tokens is a defensible default and the one we will use.

The four operations and these few terms are the whole vocabulary of data curation. Once you can compute each number for your own corpus, the entire field stops feeling like a black art and starts looking like accounting.

## Why It Matters

A model trained on uncurated data is not a worse model than a model trained on curated data. It is a model that lies to you about its own quality. The training loss goes down. The validation loss goes down. The benchmark numbers look acceptable. Then the model goes into the hands of real users and reveals that it has memorized one paragraph from one SEO farm and recites it under any prompt that vaguely resembles the trigger, or that it scores 60 percent on a benchmark whose answers were in the training data and 25 percent on a fresh held-out variant of the same benchmark. The gap between those two numbers (sometimes 10 percent, sometimes 30 percent) is the cost of skipping decontamination.

This is the kind of failure mode that does not announce itself. There is no traceback, no spike in the loss curve, no broken assertion. The model trains, the metrics look fine, and you find out months later because a user notices something off.

There is a second cost that matters even more for small training budgets. Every duplicate document in your corpus is a token your model saw twice during pretraining. If 30 percent of your corpus is duplicates and you have one trillion tokens of compute budget, you have effectively given the model 700 billion unique tokens and 300 billion redundant tokens. The redundant tokens are not free. They cost the same compute, they fill the same number of optimizer steps, and they teach the model to overfit on whatever happened to be duplicated. Dedup is one of the cheapest interventions you can run, and the lift from a properly deduplicated corpus often shows up immediately in val_bpb at fixed compute.

There is a third cost that is harder to see and harder to fix. Mixing weights determine what your model is good at. A corpus that is 95 percent web crawl and 5 percent code produces a model that writes English fluently and writes Python badly. A corpus that is 30 percent code, 30 percent math, and 40 percent web produces a different model with different strengths. The mixing weight schedule is one of the most consequential hyperparameters in pretraining, and almost nobody studies it carefully because it is not a single knob. It is a vector that interacts with everything.

I want to push back on a piece of conventional advice here. The standard line is "use a popular public mix and copy what the big labs do." That advice protects you from the worst mistakes but it also commits you to whatever those labs happened to optimize for, which may not be what you want. The fine-tuning runs I ran at scale taught me that even small reweightings of the mix can move downstream behavior in ways that no architecture change at the same compute could. Mix is content. Treat it as content, not as configuration.

This chapter sits where it does for a structural reason. **Project 9: Pretraining on the Real Web** trained a model on a corpus you trusted. **Project 11: Training Debugging — Spikes, NaNs, Profiling** taught you to read a sick training run and name the failure. **Project 22: Evaluation Methodology** will teach you to measure what your model actually knows. None of those chapters do their job if the data is corrupt. The training looks healthy in Project 9. The dashboard looks healthy in Project 11. The benchmark looks healthy in Project 22. And the model is still bad. Curation is the work you do to make every other measurement honest.

## How to run this project

```bash
# Proxy run (tiny model, runs on CPU in <60s):
python projects/10_data-curation-and-contamination/build.py --tiny

# Full lab (requires hardware — see setup/03_gpu-and-hardware-tiers.md):
python projects/10_data-curation-and-contamination/build.py --full

# The BREAK IT experiment:
python projects/10_data-curation-and-contamination/break_it.py
```

## Outputs

**Our run:** we audited the exact 1.07 M-document FineWeb-Edu corpus our Chapter 9 model trained on, in 4.5 minutes on 16 CPUs. The audit covers quality heuristics, MinHash + LSH deduplication, 13-gram contamination against our validation split and against MMLU/ARC, and the mixing schedule. We then retrained on a decontaminated corpus to measure how much the leaked validation text actually inflated our numbers. Results and learnings are in [`NOTES.md`](NOTES.md).

```bash
modal run projects/10_data-curation-and-contamination/modal_curate.py --mode audit
.venv/bin/python projects/10_data-curation-and-contamination/analyze.py val
```

## Read in the book

This project is Chapter 10 of *Under the Hood: Build Every Layer of a Large Language Model from Scratch*. Buy the book at <https://leanpub.com/under-the-hood>.
