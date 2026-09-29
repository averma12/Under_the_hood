"""Audit a document stream: quality flags, exact/near duplicates, contamination.

The first `val_docs` documents are the validation split (as in Chapter 9's
shards). Two passes over a process pool:
  A. validation documents -> their 13-grams become the validation index
  B. training documents -> stats, MinHash, overlap with validation and benchmarks
Then exact-duplicate groups and MinHash/LSH near-duplicate clusters over all
documents, and a per-document keep mask for a curated training set.
"""

from __future__ import annotations

import itertools
from multiprocessing import Pool

import mmh3
import numpy as np

import curate as c

FLAGS = list(c.QUALITY_RULES)
STAT_KEYS = ["chars", "words", "mean_word_len", "repetition", "non_ascii", "symbol_ratio",
             "bullet_lines", "alpha_words"]
PARAMS = c.minhash_params(128)
SNIPPET = 300

_VAL_INDEX = np.empty(0, dtype=np.uint64)
_BENCH: dict = {}


def benchmark_index(items: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    """Sorted 13-gram hashes of eval items, the item each came from, and how many
    items are long enough (>= 13 words) to be checked at all."""
    hashes, owners, checkable = [], [], 0
    for item_id, text in enumerate(items):
        grams = c.ngram_hashes(c.normalize(text))
        checkable += grams.size > 0
        hashes.append(grams)
        owners.append(np.full(grams.size, item_id, dtype=np.int64))
    hashes, owners = np.concatenate(hashes), np.concatenate(owners)
    order = np.argsort(hashes, kind="stable")
    return hashes[order], owners[order], checkable



def _init(val_index, bench):
    global _VAL_INDEX, _BENCH
    _VAL_INDEX, _BENCH = val_index, bench


def _analyze(text: str):
    words = c.normalize(text)
    stats = c.doc_stats(text)
    flags = sum(1 << i for i, name in enumerate(FLAGS) if c.QUALITY_RULES[name](stats))
    exact = mmh3.hash64(" ".join(words), signed=False)[0]
    if len(words) >= 5:
        sig = c.minhash_signature(c.shingle_hashes(words), PARAMS)
    else:  # too short to shingle: a random signature so it never clusters
        sig = np.random.default_rng(exact).integers(0, 2**32, 128, dtype=np.uint64).astype(np.uint32)
    grams = c.ngram_hashes(words)
    bench_hits = {}
    for name, (hashes, owners, _) in _BENCH.items():
        pos = c.index_positions(grams, hashes)
        if pos.size:
            items, counts = np.unique(owners[pos], return_counts=True)
            bench_hits[name] = dict(zip(items.tolist(), counts.tolist()))  # item -> matched grams
    return ([stats[k] for k in STAT_KEYS], flags, exact, sig, grams, bench_hits,
            text[:SNIPPET])


def _val_chunk(texts):
    return [_analyze(t) for t in texts]


def _train_chunk(texts):
    out = []
    for text in texts:
        stats, flags, exact, sig, grams, bench_hits, snippet = _analyze(text)
        val_pos = c.index_positions(grams, _VAL_INDEX)
        full = text if (val_pos.size or bench_hits) else None  # keep evidence only
        out.append((stats, flags, exact, sig, val_pos, grams.size, bench_hits, snippet, full))
    return out


def _chunks(iterable, size):
    iterator = iter(iterable)
    while chunk := list(itertools.islice(iterator, size)):
        yield chunk


def audit(documents, val_docs, benchmarks, processes=8, chunk=256, log=print):
    documents = iter(documents)
    bench = {name: benchmark_index(items) for name, items in benchmarks.items()}
    rows = []  # per document: stats, flags, exact, sig, snippet
    val_grams, val_texts = [], []

    with Pool(processes, initializer=_init, initargs=(np.empty(0, np.uint64), bench)) as pool:
        val_list = list(itertools.islice(documents, val_docs))
        for texts, results in zip(_chunks(val_list, chunk),
                                  pool.imap(_val_chunk, _chunks(val_list, chunk))):
            for text, (stats, flags, exact, sig, grams, _, snippet) in zip(texts, results):
                rows.append((stats, flags, exact, sig, snippet))
                val_grams.append(grams)
                val_texts.append(text)
    val_index = np.unique(np.concatenate(val_grams)) if val_grams else np.empty(0, np.uint64)
    log(f"validation: {len(val_grams)} docs, {val_index.size:,} distinct 13-grams")

    val_hit = np.zeros(val_index.size, dtype=bool)
    overlap = []  # (doc_id, shared 13-grams, doc 13-grams, text)
    bench_docs = {name: [] for name in bench}
    with Pool(processes, initializer=_init, initargs=(val_index, bench)) as pool:
        doc_id = len(rows)
        for results in pool.imap(_train_chunk, _chunks(documents, chunk)):
            for stats, flags, exact, sig, val_pos, n_grams, bench_hits, snippet, full in results:
                rows.append((stats, flags, exact, sig, snippet))
                if val_pos.size:
                    val_hit[val_pos] = True
                    overlap.append((doc_id, int(val_pos.size), int(n_grams), full))
                for name, items in bench_hits.items():
                    bench_docs[name].append((doc_id, items, full))
                doc_id += 1
            if doc_id % 100_000 < chunk:
                log(f"{doc_id:,} documents audited")

    n = len(rows)
    stats = np.array([r[0] for r in rows], dtype=np.float32)
    flags = np.array([r[1] for r in rows], dtype=np.uint16)
    exact = np.array([r[2] for r in rows], dtype=np.uint64)
    sigs = np.stack([r[3] for r in rows])
    snippets = [r[4] for r in rows]
    is_val = np.arange(n) < val_docs
    log("clustering near duplicates")
    clusters = c.near_duplicate_clusters(sigs, threshold=0.8, bands=16)
    _, exact_group, exact_counts = np.unique(exact, return_inverse=True, return_counts=True)

    # Per validation doc: what fraction of its 13-grams appear somewhere in training?
    val_leak = np.array([val_hit[np.searchsorted(val_index, g)].mean() if g.size else 0.0
                         for g in val_grams])
    # A cluster "touches validation" if its root (earliest member) is a validation doc.
    val_cluster = is_val[clusters]
    doc_overlap = np.zeros(n, dtype=np.int32)
    for doc_id, shared, _, _ in overlap:
        doc_overlap[doc_id] = shared
    bench_hit = np.zeros(n, dtype=bool)
    for docs in bench_docs.values():
        bench_hit[[d for d, _, _ in docs]] = True

    quality_bad = flags > 0
    near_dup = clusters != np.arange(n)  # not the earliest member of its cluster
    keep = (~is_val & ~quality_bad & ~near_dup & ~val_cluster & (doc_overlap < 50)
            & ~bench_hit)
    arrays = dict(stats=stats, flags=flags, exact=exact, clusters=clusters,
                  exact_group=exact_group, is_val=is_val, val_leak=val_leak,
                  doc_overlap=doc_overlap, bench_hit=bench_hit, keep=keep)
    summary = _summarize(arrays, exact_counts, bench, bench_docs, overlap, val_index, val_hit,
                         benchmarks)
    examples = _examples(arrays, snippets, overlap, bench_docs, benchmarks, val_texts)
    examples["benchmark_hit_details"] = benchmark_hit_details(bench, bench_docs)
    return summary, examples, arrays


def _summarize(a, exact_counts, bench, bench_docs, overlap, val_index, val_hit, benchmarks):
    n = len(a["flags"])
    train = ~a["is_val"]
    clusters = a["clusters"]
    sizes = np.bincount(clusters, minlength=n)
    flag_counts = {name: int(((a["flags"] >> i) & 1).astype(bool)[train].sum())
                   for i, name in enumerate(FLAGS)}
    val_cluster_train = train & a["is_val"][clusters]
    return {
        "documents": n, "validation_docs": int(a["is_val"].sum()),
        "training_docs": int(train.sum()),
        "quality": {"flagged_training_docs": int((a["flags"][train] > 0).sum()),
                    "by_flag": flag_counts},
        "dedup": {
            "exact_duplicate_docs": int(n - exact_counts.size),
            "near_duplicate_docs": int((clusters != np.arange(n)).sum()),
            "clusters_with_copies": int((sizes > 1).sum()),
            "largest_cluster": int(sizes.max()),
            "training_docs_near_duplicating_validation": int(val_cluster_train.sum()),
        },
        "val_contamination": {
            "validation_13grams": int(val_index.size),
            "validation_13grams_seen_in_training": float(val_hit.mean()) if val_hit.size else 0.0,
            "training_docs_sharing_any_13gram": len(overlap),
            "training_docs_sharing_50plus_13grams": int((a["doc_overlap"] >= 50).sum()),
            "validation_docs_leak_ge_50pct": int((a["val_leak"] >= 0.5).sum()),
            "validation_docs_leak_ge_80pct": int((a["val_leak"] >= 0.8).sum()),
            "mean_validation_leak": float(a["val_leak"].mean()),
        },
        "benchmarks": {
            name: {"items": len(benchmarks[name]),
                   "checkable_items": checkable,
                   "items_found_in_training": len({i for _, hits, _ in bench_docs[name]
                                                   for i in hits}),
                   "training_docs_with_hits": len(bench_docs[name])}
            for name, (_, _, checkable) in bench.items()},
        "curated_training_docs": int(a["keep"].sum()),
    }


def _examples(a, snippets, overlap, bench_docs, benchmarks, val_texts, k=5):
    n = len(snippets)
    sizes = np.bincount(a["clusters"], minlength=n)
    biggest = np.argsort(-sizes)[:k]
    by_flag = {}
    for i, name in enumerate(FLAGS):
        ids = np.flatnonzero((a["flags"] >> i) & 1)[:k]
        by_flag[name] = [snippets[j] for j in ids]
    worst_val = np.argsort(-a["val_leak"])[:k]
    return {
        "largest_duplicate_clusters": [
            {"root": int(r), "size": int(sizes[r]), "text": snippets[r]} for r in biggest],
        "quality_flag_examples": by_flag,
        "most_leaked_validation_docs": [
            {"val_doc": int(i), "leak": float(a["val_leak"][i]), "text": val_texts[i][:SNIPPET]}
            for i in worst_val],
        "training_docs_overlapping_validation": [
            {"doc": d, "shared_13grams": s, "doc_13grams": g, "text": (t or "")[:SNIPPET]}
            for d, s, g, t in sorted(overlap, key=lambda o: -o[1])[:k]],
        "benchmark_hits": {
            name: [{"doc": d, "items": [benchmarks[name][i][:SNIPPET] for i in list(hits)[:3]],
                    "doc_text": (t or "")[:1000]} for d, hits, t in docs[:k]]
            for name, docs in bench_docs.items()},
    }


def benchmark_hit_details(bench, bench_docs):
    """Every (training doc, eval item) hit with matched and total 13-gram counts."""
    details = {}
    for name, docs in bench_docs.items():
        _, owners, _ = bench[name]
        totals = np.bincount(owners)
        details[name] = [{"doc": d, "item": int(i), "matched": int(m), "item_grams": int(totals[i]),
                          "doc_text": t} for d, hits, t in docs for i, m in hits.items()]
    return details
