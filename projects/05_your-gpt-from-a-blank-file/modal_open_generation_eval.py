"""Run the open-generation comparison on the saved A100 checkpoints.

From the project directory:
  modal run modal_open_generation_eval.py --prompts 12 --seeds 2

This only generates text; automatic scoring and blinded sheets are produced
locally from the returned generations.json so training tokens stay local.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


HERE = Path(__file__).resolve().parent
DATA = HERE / "outputs" / "my_gpt_2048"
RUN_ID = "gpt-mhc-flash-stable-30k"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1")
    .add_local_file(HERE / "my_gpt.py", "/root/my_gpt.py")
    .add_local_file(HERE / "open_generation_eval.py", "/root/open_generation_eval.py")
    .add_local_file(DATA / "tokenizer.json", "/root/data/tokenizer.json")
)
app = modal.App("under-the-hood-gpt-open-generation-eval")
volume = modal.Volume.from_name("under-the-hood-gpt-a100")


@app.function(image=image, gpu="A100-40GB", timeout=3600,
              volumes={"/artifacts": volume})
def generate_on_gpu(prompts: int, seeds: int, new_tokens: int,
                    temperature: float) -> dict:
    import sys

    sys.path.insert(0, "/root")
    import open_generation_eval as evaluation

    volume.reload()
    return evaluation.generate_rows(
        Path("/artifacts") / RUN_ID, Path("/root/data/tokenizer.json"),
        evaluation.PROMPTS[:prompts], seeds, new_tokens, temperature, "cuda",
    )


@app.local_entrypoint()
def main(prompts: int = 12, seeds: int = 2, new_tokens: int = 128,
         temperature: float = 0.8):
    if not 1 <= prompts <= 12 or seeds < 1 or new_tokens < 1:
        raise ValueError("Need 1-12 prompts, positive seeds and new_tokens")
    import open_generation_eval as evaluation

    data = generate_on_gpu.remote(prompts, seeds, new_tokens, temperature)
    out_dir = (DATA / "modal_runs" / RUN_ID / "open_generation_eval"
               / f"{prompts}p_{seeds}s_{new_tokens}t")
    out_dir.mkdir(parents=True, exist_ok=True)
    generations = out_dir / "generations.json"
    generations.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    train_ids = json.loads((DATA / "train_ids.json").read_text(encoding="utf-8"))
    scored = evaluation.score_data(data, train_ids)
    (out_dir / "automatic_scores.json").write_text(
        json.dumps(scored, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    evaluation.write_blind(data, out_dir)
    print(json.dumps(scored["summary"], indent=2), flush=True)
    print(f"Outputs: {out_dir}", flush=True)
