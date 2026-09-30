"""Chapter 10 building blocks: quality stats, MinHash + LSH dedup, n-gram contamination.

Everything works on words from normalize(): lowercase, punctuation stripped,
whitespace collapsed. So "Hello, World!" and "hello world" compare equal.

MinHash note: the book's version hashes shingles to signed 64-bit ints, which
crashes np.fromiter(..., uint64). Its (a*x + b) mod p also wraps silently in
uint64. Here each "permutation" is multiply-add-shift hashing (Dietzfelbinger,
1996): h(x) = ((a*x + b) mod 2^64) >> 32 with random 64-bit a, b and 32-bit
shingle hashes x. That family is designed for wrap-around arithmetic, so the
uint64 overflow is exactly the mod 2^64 we want.
"""

from __future__ import annotations

import re
from collections import Counter

import mmh3
import numpy as np

_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?")


def normalize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


# ---------------------------------------------------------------- quality


def doc_stats(text: str) -> dict:
    """Cheap per-document signals; the filters in quality_flags use them."""
    raw_words = text.split()
    words = normalize(text)
    n = len(words)
    fives = [tuple(words[i:i + 5]) for i in range(n - 4)]
    counts = Counter(fives)
    repeated = sum(c for c in counts.values() if c > 1)
    lines = [line for line in text.splitlines() if line.strip()]
    return {
        "chars": len(text),
        "words": n,
        "mean_word_len": sum(map(len, words)) / n if n else 0.0,
        # Share of 5-gram occurrences that belong to a 5-gram seen more than once.
        "repetition": repeated / len(fives) if fives else 0.0,
        "non_ascii": sum(ord(c) > 127 for c in text) / max(len(text), 1),
        "symbol_ratio": sum(w.startswith(("#", "…", "...")) for w in raw_words) / max(len(raw_words), 1),
        "bullet_lines": sum(line.lstrip().startswith(("•", "-", "*")) for line in lines) / max(len(lines), 1),
        "alpha_words": n / max(len(raw_words), 1),
    }


QUALITY_RULES = {
    # Gopher-style heuristics (Rae et al., 2021), loosened for an already-filtered corpus.
    "too_short": lambda s: s["words"] < 50,
    "too_long": lambda s: s["words"] > 100_000,
    "odd_word_length": lambda s: not 3 <= s["mean_word_len"] <= 10,
    "repetitive": lambda s: s["repetition"] > 0.30,
    "non_english_chars": lambda s: s["non_ascii"] > 0.20,
    "symbol_heavy": lambda s: s["symbol_ratio"] > 0.10,
    "mostly_bullets": lambda s: s["bullet_lines"] > 0.90,
    "few_real_words": lambda s: s["alpha_words"] < 0.80,
}


def quality_flags(stats: dict) -> list[str]:
    return [name for name, rule in QUALITY_RULES.items() if rule(stats)]


# ---------------------------------------------------------------- MinHash


def shingle_hashes(words: list[str], k: int = 5) -> np.ndarray:
    """Unsigned 32-bit hashes of every k-word shingle (as a set)."""
    if len(words) < k:
        return np.empty(0, dtype=np.uint64)
    return np.unique(np.fromiter(
        (mmh3.hash(" ".join(words[i:i + k]), signed=False) for i in range(len(words) - k + 1)),
        dtype=np.uint64))


def minhash_params(num_perm: int = 128, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    b = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64) * np.uint64(2)
    return a, b  # a is odd, as multiply-shift hashing requires


def minhash_signature(hashes: np.ndarray, params) -> np.ndarray:
    """For each of num_perm hash functions, the minimum hash over the shingle set."""
    a, b = params
    if hashes.size == 0:
        return np.full(a.size, np.iinfo(np.uint32).max, dtype=np.uint32)
    # uint64 array arithmetic wraps mod 2^64 by design here; keep the top 32 bits.
    permuted = (np.outer(a, hashes) + b[:, None]) >> np.uint64(32)
    return permuted.min(axis=1).astype(np.uint32)


def estimated_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """P(min-hashes agree) = Jaccard similarity of the two shingle sets."""
    return float((sig_a == sig_b).mean())


def lsh_candidate_pairs(signatures: np.ndarray, bands: int = 16) -> np.ndarray:
    """Pairs (i, j) whose signatures agree exactly on at least one band.

    With r = num_perm / bands rows per band, a pair with Jaccard s collides in
    some band with probability 1 - (1 - s^r)^bands: a steep S-curve. For 16
    bands x 8 rows it crosses 50% near s = 0.71.
    """
    n, num_perm = signatures.shape
    rows = num_perm // bands
    pairs = []
    for band in range(bands):
        block = np.ascontiguousarray(signatures[:, band * rows:(band + 1) * rows])
        keys = block.view(np.dtype((np.void, block.dtype.itemsize * rows))).ravel()
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        sizes = np.diff(np.r_[starts, n])
        for start, size in zip(starts[sizes > 1], sizes[sizes > 1]):
            members = order[start:start + size]
            first = members.min()
            # Link every member to the smallest id; union-find does the rest.
            pairs.extend((first, m) for m in members if m != first)
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.array(pairs, dtype=np.int64), axis=0)


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # the earliest document is the root


def near_duplicate_clusters(signatures: np.ndarray, threshold: float = 0.8,
                            bands: int = 16) -> np.ndarray:
    """Cluster id per document (its earliest member); verify LSH pairs by Jaccard."""
    uf = UnionFind(len(signatures))
    for i, j in lsh_candidate_pairs(signatures, bands):
        if estimated_jaccard(signatures[i], signatures[j]) >= threshold:
            uf.union(int(i), int(j))
    return np.array([uf.find(i) for i in range(len(signatures))])


# ---------------------------------------------------------------- contamination


def ngram_hashes(words: list[str], n: int = 13) -> np.ndarray:
    """64-bit hashes of every n-word window (unsigned, as a sorted unique array)."""
    if len(words) < n:
        return np.empty(0, dtype=np.uint64)
    return np.unique(np.fromiter(
        (mmh3.hash64(" ".join(words[i:i + n]), signed=False)[0] for i in range(len(words) - n + 1)),
        dtype=np.uint64))


def index_positions(doc_ngrams: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Positions in a sorted index of the document n-grams it contains.

    Binary search per n-gram: O(m log n). np.isin would re-sort the whole
    multi-million-entry index for every document.
    """
    if doc_ngrams.size == 0 or index.size == 0:
        return np.empty(0, dtype=np.int64)
    pos = np.searchsorted(index, doc_ngrams)
    inside = pos < index.size
    pos, grams = pos[inside], doc_ngrams[inside]
    return pos[index[pos] == grams]


def overlap_count(doc_ngrams: np.ndarray, index: np.ndarray) -> int:
    """How many of a document's n-grams appear in a sorted eval index."""
    return int(index_positions(doc_ngrams, index).size)


# ---------------------------------------------------------------- mixing


def mixing_weights(step: int, total_steps: int, start=None, end=None, anneal_from=0.8) -> dict:
    """The book's schedule: a fixed web-heavy mix, then a linear ramp toward
    code and math over the last 20% of training ("annealing" on higher-value data)."""
    start = start or {"web": 0.85, "code": 0.10, "math": 0.05}
    end = end or {"web": 0.55, "code": 0.30, "math": 0.15}
    t = min(max((step / total_steps - anneal_from) / (1 - anneal_from), 0.0), 1.0)
    return {k: (1 - t) * start[k] + t * end[k] for k in start}


def sample_sources(weights: dict, batch: int, rng: np.random.Generator) -> list[str]:
    """Which source each sequence in a batch comes from."""
    names = list(weights)
    probs = np.array([weights[k] for k in names], dtype=np.float64)
    return [names[i] for i in rng.choice(len(names), size=batch, p=probs / probs.sum())]
