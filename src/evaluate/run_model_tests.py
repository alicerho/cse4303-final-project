"""
run_model_tests_optimized.py
Fast, batch-based GPT-2 evaluation with:
- Warning suppression
- Batch inference (10× faster)
- Progress bar
- Auto-resume if interrupted
- Safe local model loading
"""

import argparse
import json
from pathlib import Path
import warnings
from transformers import pipeline, logging
from tqdm import tqdm

# -------------------------------
# Silence HuggingFace warnings
# -------------------------------
warnings.filterwarnings("ignore")
logging.set_verbosity_error()


def load_prompts(prompts_path, start_index=0, max_examples=None):
    """Load prompts from jsonl and optionally resume from an index."""
    prompts = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    if max_examples:
        prompts = prompts[:max_examples]
    return prompts[start_index:]


def run(prompts_path, model_name, out_path, max_examples=None, batch_size=16):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Determine resume point
    resume_at = 0
    if Path(out_path).exists():
        with open(out_path, "r", encoding="utf-8") as f:
            resume_at = sum(1 for _ in f)
        print(f"[INFO] Resuming at line {resume_at}")

    # Load prompts starting from resume index
    prompts = load_prompts(prompts_path, start_index=resume_at, max_examples=max_examples)

    print(f"[INFO] Loading local model pipeline: {model_name}")
    gen = pipeline(
        "text-generation",
        model=model_name,
        device=-1,
        model_kwargs={"torch_dtype": "float32"}  # explicit for CPU stability
    )

    # Open output file in append mode
    fout = open(out_path, "a", encoding="utf-8")

    # Iterate in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing"):
        batch = prompts[i: i + batch_size]

        texts = [p["prompt"] for p in batch]
        results = gen(texts, max_length=150, do_sample=False)

        # FIXED: results is a list of lists → use results[j][0]["generated_text"]
        for entry, gen_out in zip(batch, results):
            generated_text = gen_out[0]["generated_text"]

            fout.write(json.dumps({
                "id": entry.get("id"),
                "prompt": entry.get("prompt"),
                "generated": generated_text
            }, ensure_ascii=False) + "\n")

    fout.close()
    print(f"[OK] Completed. Total saved = {resume_at + len(prompts)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, default="data/augmented_prompts.jsonl")
    parser.add_argument("--model", type=str, default="src/data/models/gpt2")
    parser.add_argument("--out", type=str, default="data/results.jsonl")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    run(args.prompts, args.model, args.out, args.max, args.batch)


if __name__ == "__main__":
    main()