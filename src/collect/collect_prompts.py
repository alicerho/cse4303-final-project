"""
collect_prompts.py
Create a sanitized demo dataset; optionally sample public datasets but replace prompt text with placeholders.
USAGE:
    python src/collect/collect_prompts.py --out data/raw_prompts.jsonl
    python src/collect/collect_prompts.py --use_public --out data/raw_prompts.jsonl
"""
import argparse
import json
from pathlib import Path

def make_sanitized_demo(out_path):
    # Demo: sanitized placeholders only
    records = [
        {"id":"demo_0001","prompt":"How do I boil an egg?","label":0,"source":"demo","category":"benign"},
        {"id":"demo_0002","prompt":"Summarize Hamlet in one paragraph.","label":0,"source":"demo","category":"benign"},
        {"id":"san_0001","prompt":"[SANITIZED_JAILBREAK_ROLEPLAY_PLACEHOLDER]","label":1,"source":"sanitized","category":"roleplay"},
        {"id":"san_0002","prompt":"[SANITIZED_JAILBREAK_RULE_NEGATION_PLACEHOLDER]","label":1,"source":"sanitized","category":"rule_negation"}
    ]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[OK] Wrote {len(records)} sanitized demo records to {out_path}")

def download_and_sanitize(out_path, max_examples=200):
    # Optional: sample public dataset and sanitize prompt text
    # WARNING: this will download content that may contain sensitive text.
    from datasets import load_dataset
    print("[INFO] Downloading sample 'allenai/wildjailbreak' (may contain sensitive content).")
    ds = load_dataset("allenai/wildjailbreak", split="train")
    records = []
    for i, item in enumerate(ds):
        records.append({
            "id": f"wj_{i:06d}",
            "prompt": "[PUBLIC_DATA_PLACEHOLDER]",
            "label": 1,
            "source": "wildjailbreak",
            "category": item.get("category", "unknown")
        })
        if i+1 >= max_examples:
            break
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[OK] Wrote {len(records)} sanitized public samples to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/raw_prompts.jsonl")
    parser.add_argument("--use_public", action="store_true")
    parser.add_argument("--max_public", type=int, default=200)
    args = parser.parse_args()
    if args.use_public:
        download_and_sanitize(args.out, args.max_public)
    else:
        make_sanitized_demo(args.out)

if __name__ == "__main__":
    main()
