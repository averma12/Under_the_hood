"""Chapter 10: quality stats, MinHash/LSH dedup, and 13-gram contamination."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import curate as c  # noqa: E402

rng = random.Random(0)
VOCAB = [f"w{i}" for i in range(5000)]


def random_doc(n=300):
    return " ".join(rng.choice(VOCAB) for _ in range(n))


def edit(text, fraction):
    words = text.split()
    for i in rng.sample(range(len(words)), int(len(words) * fraction)):
        words[i] = rng.choice(VOCAB)
    return " ".join(words)


def true_jaccard(a, b, k=5):
    sa = set(c.shingle_hashes(c.normalize(a), k).tolist())
    sb = set(c.shingle_hashes(c.normalize(b), k).tolist())
    return len(sa & sb) / len(sa | sb)


def test_normalize_ignores_case_and_punctuation():
    assert c.normalize("Hello, World! It's 2024.") == ["hello", "world", "it's", "2024"]


def test_quality_flags():
    good = ("Cells turn food into usable energy through respiration. Glucose is split in the "
            "cytoplasm, and the pieces enter the mitochondria, where oxygen accepts electrons. "
            "Plants also make glucose during photosynthesis, using light captured by "
            "chlorophyll in their chloroplasts. Both processes move carbon between living "
            "things and the atmosphere, which is why forests and oceans matter for climate.")
    assert c.quality_flags(c.doc_stats(good)) == []
    assert "too_short" in c.quality_flags(c.doc_stats("Buy now. Click here."))
    spam = "best cheap flights book today " * 40
    stats = c.doc_stats(spam)
    assert stats["repetition"] > 0.9 and "repetitive" in c.quality_flags(stats)
    assert "non_english_chars" in c.quality_flags(c.doc_stats("это текст на русском " * 30))


def test_minhash_has_no_overflow_and_estimates_jaccard():
    params = c.minhash_params(256)
    base = random_doc()
    for fraction in (0.0, 0.02, 0.1, 0.3):
        other = edit(base, fraction)
        sigs = [c.minhash_signature(c.shingle_hashes(c.normalize(t)), params) for t in (base, other)]
        assert sigs[0].dtype == np.uint32
        assert c.estimated_jaccard(*sigs) == pytest.approx(true_jaccard(base, other), abs=0.08)


def test_lsh_clusters_near_duplicates_only():
    params = c.minhash_params(128)
    originals = [random_doc() for _ in range(30)]
    docs = originals + [edit(originals[3], 0.01), originals[7], edit(originals[3], 0.015)]
    sigs = np.stack([c.minhash_signature(c.shingle_hashes(c.normalize(d)), params) for d in docs])
    clusters = c.near_duplicate_clusters(sigs, threshold=0.8, bands=16)
    assert clusters[30] == clusters[32] == 3  # both edits of doc 3 join its cluster
    assert clusters[31] == 7  # the exact copy of doc 7
    singles = [i for i in range(30) if i not in (3, 7)]
    assert all(clusters[i] == i for i in singles)  # unrelated docs stay alone


def test_thirteen_gram_contamination():
    question = ("Which organelle is known as the powerhouse of the cell because it "
                "produces most of the chemical energy? Mitochondria")
    index = c.ngram_hashes(c.normalize(question))
    leaked = random_doc(200) + " " + question.upper() + " " + random_doc(200)
    clean = random_doc(400)
    assert c.overlap_count(c.ngram_hashes(c.normalize(leaked)), index) == index.size
    assert c.overlap_count(c.ngram_hashes(c.normalize(clean)), index) == 0
    assert c.ngram_hashes(c.normalize("too short to have a thirteen gram")).size == 0


def test_audit_end_to_end_finds_planted_problems():
    import audit_corpus

    def sentence_doc(i, n=120):
        r = random.Random(i)
        return " ".join(r.choice(VOCAB) for _ in range(n))

    val = [sentence_doc(i) for i in range(10)]
    train = [sentence_doc(100 + i) for i in range(40)]
    train[5] = val[3]  # validation document copied into training
    train[6] = train[7]  # exact duplicate inside training
    train[8] = edit(train[9], 0.01)  # near duplicate inside training
    question = ("In what year did the first humans land on the moon as part of the "
                "Apollo eleven mission launched by NASA")
    train[10] = train[10] + " " + question  # benchmark item leaked into training
    train[11] = "cheap pills online " * 40  # repetitive spam
    summary, examples, arrays = audit_corpus.audit(
        val + train, val_docs=10, benchmarks={"quiz": [question, "too short"]},
        processes=2, chunk=8, log=lambda *_: None)
    assert summary["documents"] == 50
    assert summary["dedup"]["exact_duplicate_docs"] == 2  # train[5] and the train[6/7] pair
    assert summary["dedup"]["training_docs_near_duplicating_validation"] == 1
    assert summary["val_contamination"]["validation_docs_leak_ge_80pct"] == 1
    assert examples["most_leaked_validation_docs"][0]["val_doc"] == 3
    quiz = summary["benchmarks"]["quiz"]
    assert quiz == {"items": 2, "checkable_items": 1, "items_found_in_training": 1,
                    "training_docs_with_hits": 1}
    assert summary["quality"]["by_flag"]["repetitive"] == 1
    keep = arrays["keep"]
    assert not keep[:10].any()  # validation is never training data
    for bad in (5, 7, 9, 10, 11):  # val copy, later exact dup, later near dup, leak, spam
        assert not keep[10 + bad], bad
    assert keep[10 + 6] and keep[10 + 8]  # the earliest copy in each cluster survives
    assert summary["curated_training_docs"] == 40 - 5
    hit = examples["benchmark_hit_details"]["quiz"]
    assert hit == [{"doc": 20, "item": 0, "matched": 9, "item_grams": 9,
                    "doc_text": train[10]}]


def test_mixing_schedule_and_sampling():
    for step in range(0, 1001, 50):
        w = c.mixing_weights(step, 1000)
        assert sum(w.values()) == pytest.approx(1.0)
    assert c.mixing_weights(0, 1000) == c.mixing_weights(800, 1000) == pytest.approx(
        {"web": 0.85, "code": 0.10, "math": 0.05})
    assert c.mixing_weights(900, 1000) == pytest.approx({"web": 0.70, "code": 0.20, "math": 0.10})
    assert c.mixing_weights(1000, 1000) == pytest.approx({"web": 0.55, "code": 0.30, "math": 0.15})
    draws = c.sample_sources(c.mixing_weights(1000, 1000), 20_000, np.random.default_rng(0))
    assert draws.count("code") / len(draws) == pytest.approx(0.30, abs=0.01)
