"""Compare toy-GPT checkpoints beyond perplexity.

Generate with fixed prompts/settings, report repetition and exact training
overlap, and export randomized, blinded pairs for human review.

Examples:
  python open_generation_eval.py generate --device cpu --prompts 3 --seeds 1
  python open_generation_eval.py score --input /path/to/generations.json
  python open_generation_eval.py blind --input /path/to/generations.json
  python open_generation_eval.py summarize --ratings /path/to/ratings.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA = HERE / "outputs" / "my_gpt_2048"
RUN = DATA / "modal_runs" / "gpt-mhc-flash-stable-30k"
PROMPTS = [
    "Harry looked at",
    "Hermione said,",
    "The door opened and",
    "Ron stared at the empty corridor.",
    "The classroom fell silent when",
    "At the edge of the forest,",
    "The letter on the table said",
    "Before anyone could answer,",
    "The train slowed as",
    "Harry reached for his wand, but",
    "Professor McGonagall looked at the students.",
    "Hermione opened the old book and",
]
CHECKPOINTS = ["best", "step_15000", "step_25000", "step_30000"]
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
MASK = (1 << 64) - 1
HASH_BASE = 1_000_003
COPY_N = 20  # A conservative, exact BPE-token overlap screen.


def generate_rows(checkpoint_dir, tokenizer_path, prompts, seeds, new_tokens,
                  temperature, device):
    import torch
    import my_gpt as m

    if not 0 < temperature or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    if not 1 <= new_tokens:
        raise ValueError("new_tokens must be positive")
    tokenizer = m.BPETokenizer.load(tokenizer_path)
    rows = []
    for name in CHECKPOINTS:
        path = Path(checkpoint_dir) / f"{name}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        cfg = m.TransformerConfig(**saved["config"])
        model = m.GPT(cfg).to(device).eval()
        model.load_state_dict(saved["model"])
        step = int(saved["step"])
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.startswith("cuda") else nullcontext())
        with torch.inference_mode(), autocast:
            for prompt_id, prompt in enumerate(prompts):
                prefix = torch.tensor([tokenizer.encode(prompt)], device=device)
                for sample_id in range(seeds):
                    seed = 10_000 + prompt_id * 100 + sample_id
                    torch.manual_seed(seed)
                    if device.startswith("cuda"):
                        torch.cuda.manual_seed_all(seed)
                    generated = model.generate(
                        prefix, new_tokens, temperature=temperature,
                    )
                    token_ids = generated[0, prefix.shape[1]:].tolist()
                    rows.append({
                        "checkpoint": name, "step": step,
                        "prompt_id": prompt_id, "prompt": prompt,
                        "sample_id": sample_id, "seed": seed,
                        "continuation": tokenizer.decode(token_ids),
                        "token_ids": token_ids,
                    })
                print(f"{name}: {prompt_id + 1}/{len(prompts)} prompts", flush=True)
        del model, saved
    return {
        "protocol": {
            "checkpoints": CHECKPOINTS, "prompts": prompts, "seeds_per_prompt": seeds,
            "new_tokens": new_tokens, "temperature": temperature,
            "sampler": "my_gpt.GPT.generate, multinomial, no top-p",
        },
        "generations": rows,
    }


def rolling_hashes(ids, n):
    """64-bit rolling hashes, used only as an overlap screen."""
    if len(ids) < n:
        return []
    power = pow(HASH_BASE, n - 1, 1 << 64)
    value = 0
    for token in ids[:n]:
        value = (value * HASH_BASE + token + 1) & MASK
    hashes = [value]
    for left, right in zip(ids, ids[n:]):
        value = ((value - (left + 1) * power) * HASH_BASE + right + 1) & MASK
        hashes.append(value)
    return hashes


def training_overlap_index(train_ids, n=COPY_N):
    return set(rolling_hashes(train_ids, n))


def copy_stats(ids, index, n=COPY_N):
    """Count generated tokens covered by matching n-grams (prompt excluded).

    This is an overlap flag, not proof of memorization: common phrases and
    public-domain-like repeated text can also match. Hash collision is highly
    unlikely; flagged spans should be verified against source text if used as
    evidence of copying.
    """
    covered = bytearray(len(ids))
    matches = 0
    for start, value in enumerate(rolling_hashes(ids, n)):
        if value in index:
            matches += 1
            covered[start:start + n] = b"\x01" * n
    return {
        "matching_20_token_windows": matches,
        "copy_covered_token_fraction": sum(covered) / len(ids) if ids else 0.0,
        "has_20_token_match": bool(matches),
    }


def diversity_stats(text):
    """Lexical diversity and within-continuation repetition, length-aware."""
    words = [word.casefold() for word in WORD_RE.findall(text)]
    result = {"word_count": len(words)}
    for n in (2, 3, 4):
        grams = [tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]
        unique = len(set(grams))
        result[f"distinct_{n}"] = unique / len(grams) if grams else None
        result[f"repeated_{n}_fraction"] = 1 - unique / len(grams) if grams else None
    return result


def score_data(data, train_ids):
    index = training_overlap_index(train_ids)
    scored = []
    for row in data["generations"]:
        scored.append({
            **row,
            **diversity_stats(row["continuation"]),
            **copy_stats(row["token_ids"], index),
        })
    grouped = defaultdict(list)
    for row in scored:
        grouped[row["checkpoint"]].append(row)
    summary = []
    fields = ("distinct_2", "distinct_3", "distinct_4",
              "repeated_4_fraction", "copy_covered_token_fraction",
              "has_20_token_match")
    for name in CHECKPOINTS:
        rows = grouped.get(name, [])
        if not rows:
            continue
        item = {"checkpoint": name, "step": rows[0]["step"], "samples": len(rows)}
        for field in fields:
            values = [float(row[field]) for row in rows if row[field] is not None]
            item[field] = sum(values) / len(values) if values else None
        summary.append(item)
    return {"protocol": data["protocol"], "summary": summary, "samples": scored,
            "metric_notes": {
                "distinct_n": "Mean within-sample fraction of unique lowercased word n-grams; larger is not always better.",
                "repeated_4_fraction": "Mean within-sample fraction of repeated word 4-grams; smaller usually indicates fewer loops.",
                "copy_covered_token_fraction": "Mean fraction of continuation BPE tokens in exact 20-token windows also found in training; not proof of memorization.",
                "human_quality": "Fluency, coherence, consistency and prompt fit require blinded review; no automatic quality score is inferred.",
            }}


def blind_pairs(data, seed=2026):
    grouped = defaultdict(dict)
    for row in data["generations"]:
        key = (row["prompt_id"], row["sample_id"])
        if row["checkpoint"] in grouped[key]:
            raise ValueError(f"Duplicate generation: {key} {row['checkpoint']}")
        grouped[key][row["checkpoint"]] = row
    rng = random.Random(seed)
    pairs = []
    for prompt_id, sample_id in sorted(grouped):
        by_checkpoint = grouped[prompt_id, sample_id]
        if any(name not in by_checkpoint for name in CHECKPOINTS):
            raise ValueError(f"Missing checkpoint for prompt {prompt_id}, sample {sample_id}")
        for challenger in CHECKPOINTS[1:]:
            left, right = CHECKPOINTS[0], challenger
            if rng.randrange(2):
                left, right = right, left
            a, b = by_checkpoint[left], by_checkpoint[right]
            opaque = f"{seed}:{prompt_id}:{sample_id}:{challenger}"
            pair_id = hashlib.blake2s(opaque.encode(), digest_size=6).hexdigest()
            pairs.append({"pair_id": pair_id, "prompt": a["prompt"],
                          "a": a, "b": b})
    rng.shuffle(pairs)
    return pairs


def write_blind(data, out_dir, seed=2026):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = blind_pairs(data, seed)
    key = {}
    md = ["# Blinded checkpoint comparison", "",
          "For each pair, read the same prompt and continuations A/B. Score",
          "fluency, coherence, consistency and prompt fit from 1 (poor) to 5",
          "(good). Enter A, B, or tie as overall_winner in ratings.csv.",
          "Keep blind_key.json closed until ratings are complete. Empty ratings",
          "mean no human evaluation has been done.", ""]
    rating_fields = ["pair_id", "overall_winner"] + [
        f"{aspect}_{side}" for aspect in
        ("fluency", "coherence", "consistency", "prompt_fit")
        for side in ("a", "b")
    ] + ["notes"]
    with (out_dir / "ratings.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rating_fields)
        writer.writeheader()
        for pair in pairs:
            writer.writerow({"pair_id": pair["pair_id"]})
            key[pair["pair_id"]] = {"a": pair["a"]["checkpoint"],
                                     "b": pair["b"]["checkpoint"],
                                     "prompt_id": pair["a"]["prompt_id"],
                                     "sample_id": pair["a"]["sample_id"]}
            md.extend([f"## {pair['pair_id']}", "",
                       f"Prompt: {pair['prompt']}", "", "**A**", "",
                       pair["a"]["continuation"], "", "**B**", "",
                       pair["b"]["continuation"], ""])
    (out_dir / "blind_pairs.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "blind_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    return len(pairs)


def summarize_ratings(ratings_path, key_path):
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    counts = defaultdict(lambda: {"win": 0, "loss": 0, "tie": 0})
    by_aspect = defaultdict(lambda: defaultdict(list))
    with Path(ratings_path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            winner = row["overall_winner"].strip().lower()
            if not winner:
                continue
            if winner not in ("a", "b", "tie"):
                raise ValueError(f"Invalid winner for {row['pair_id']}: {winner}")
            sides = key[row["pair_id"]]
            for side in ("a", "b"):
                result = "tie" if winner == "tie" else "win" if winner == side else "loss"
                counts[sides[side]][result] += 1
                for aspect in ("fluency", "coherence", "consistency", "prompt_fit"):
                    raw = row[f"{aspect}_{side}"].strip()
                    if raw:
                        rating = int(raw)
                        if rating not in range(1, 6):
                            raise ValueError(f"Invalid {aspect} rating: {rating}")
                        by_aspect[sides[side]][aspect].append(rating)
    return {name: {**counts[name], "decisions": sum(counts[name].values()),
                   "win_share_including_half_ties":
                   (counts[name]["win"] + 0.5 * counts[name]["tie"])
                   / sum(counts[name].values()) if sum(counts[name].values()) else None,
                   "mean_ratings": {aspect: sum(values) / len(values)
                                    for aspect, values in by_aspect[name].items()}}
            for name in CHECKPOINTS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--checkpoint-dir", type=Path, default=RUN)
    gen.add_argument("--tokenizer", type=Path, default=DATA / "tokenizer.json")
    gen.add_argument("--output", type=Path, default=RUN / "open_generation_eval" / "generations.json")
    gen.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    gen.add_argument("--prompts", type=int, default=len(PROMPTS))
    gen.add_argument("--seeds", type=int, default=2)
    gen.add_argument("--new-tokens", type=int, default=128)
    gen.add_argument("--temperature", type=float, default=0.8)
    score = sub.add_parser("score")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--train-ids", type=Path, default=DATA / "train_ids.json")
    score.add_argument("--output", type=Path)
    blind = sub.add_parser("blind")
    blind.add_argument("--input", type=Path, required=True)
    blind.add_argument("--out-dir", type=Path)
    summary = sub.add_parser("summarize")
    summary.add_argument("--ratings", type=Path, required=True)
    summary.add_argument("--key", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        if not 1 <= args.prompts <= len(PROMPTS) or args.seeds < 1:
            parser.error("prompts must be 1-12 and seeds must be positive")
        data = generate_rows(args.checkpoint_dir, args.tokenizer,
                             PROMPTS[:args.prompts], args.seeds,
                             args.new_tokens, args.temperature, args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(args.output)
    elif args.command == "score":
        data = json.loads(args.input.read_text(encoding="utf-8"))
        train_ids = json.loads(args.train_ids.read_text(encoding="utf-8"))
        result = score_data(data, train_ids)
        output = args.output or args.input.with_name("automatic_scores.json")
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result["summary"], indent=2))
        print(output)
    elif args.command == "blind":
        data = json.loads(args.input.read_text(encoding="utf-8"))
        out_dir = args.out_dir or args.input.parent
        print(f"Wrote {write_blind(data, out_dir)} pairs to {out_dir}")
    else:
        key_path = args.key or args.ratings.with_name("blind_key.json")
        print(json.dumps(summarize_ratings(args.ratings, key_path), indent=2))


if __name__ == "__main__":
    main()
