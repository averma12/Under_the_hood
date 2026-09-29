"""Checks for the non-perplexity generation evaluation helpers."""

import importlib.util
import json
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "open_generation_eval.py"
spec = importlib.util.spec_from_file_location("open_generation_eval", SOURCE)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_rolling_overlap_excludes_nonmatching_and_covers_matching_tokens():
    train = list(range(30))
    index = evaluation.training_overlap_index(train, n=4)
    assert evaluation.copy_stats([9, 10, 11, 12, 99], index, n=4) == {
        "matching_20_token_windows": 1,
        "copy_covered_token_fraction": 0.8,
        "has_20_token_match": True,
    }
    assert not evaluation.copy_stats([9, 10, 11, 99], index, n=4)["has_20_token_match"]


def test_diversity_flags_repetition():
    repeated = evaluation.diversity_stats("cat sat cat sat cat sat cat sat")
    varied = evaluation.diversity_stats("cat sat dog ran bird flew fish swam")
    assert repeated["distinct_2"] < varied["distinct_2"]
    assert repeated["repeated_4_fraction"] > varied["repeated_4_fraction"]


def test_blind_pairs_have_no_checkpoint_names_and_ratings_round_trip(tmp_path):
    rows = []
    for name in evaluation.CHECKPOINTS:
        rows.append({"checkpoint": name, "step": 13000 if name == "best" else 15000,
                     "prompt_id": 0, "prompt": "A door", "sample_id": 0,
                     "continuation": " opened slowly.", "token_ids": [1, 2]})
    pairs = evaluation.blind_pairs({"generations": rows})
    assert len(pairs) == 3
    assert all("best" not in pair["pair_id"] and "step" not in pair["pair_id"]
               for pair in pairs)
    evaluation.write_blind({"generations": rows}, tmp_path)
    key = json.loads((tmp_path / "blind_key.json").read_text())
    assert len(key) == 3
    assert "step_15000" not in (tmp_path / "blind_pairs.md").read_text()
    empty = evaluation.summarize_ratings(tmp_path / "ratings.csv",
                                         tmp_path / "blind_key.json")
    assert all(item["decisions"] == 0 for item in empty.values())
