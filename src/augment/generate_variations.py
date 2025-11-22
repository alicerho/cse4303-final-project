"""
generate_variations.py
Create simple paraphrases / obfuscations of sanitized prompts.
By default, creates surface-level variants that are safe to store in repo.
If you choose to run deeper paraphrasing, do so **locally** and keep outputs private.
USAGE:
    python src/augment/generate_variations.py --in data/raw_prompts.jsonl --out data/augmented_prompts.jsonl
"""
import argparse
import json
from pathlib import Path
import random
import re

def surface_variants(text):
    variants = []
    variants.append(text)
    variants.append(text.strip() + " Please explain.")
    variants.append("Briefly: " + text)
    variants.append(text.lower())
    variants.append(text.upper())
    variants.append(re.sub(r'\s+', ' ', text).strip())
    # dedupe
    out = []
    for v in variants:
        if v not in out:
            out.append(v)
    return out

def generate(in_path, out_path, max_variants=4):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_records = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("id")
            prompt = r.get("prompt","")
            label = r.get("label",0)
            variants = surface_variants(prompt)[:max_variants]
            for i, v in enumerate(variants):
                out_records.append({
                    "id": f"{pid}_v{i}",
                    "orig_id": pid,
                    "prompt": v,
                    "label": label,
                    "category": r.get("category", "unknown"),
                    "variant_type": "surface"
                })
    with open(out_path, "w", encoding="utf-8") as fout:
        for rec in out_records:
            fout.write(json.dumps(rec) + "\n")
    print(f"[OK] Wrote {len(out_records)} variant records to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inpath", type=str, default="data/raw_prompts.jsonl")
    parser.add_argument("--out", type=str, default="data/augmented_prompts.jsonl")
    parser.add_argument("--max_variants", type=int, default=4)
    args = parser.parse_args()
    generate(args.inpath, args.out, args.max_variants)

if __name__ == "__main__":
    main()
