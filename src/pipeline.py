"""
pipeline.py
High-level script to run the full safe demo pipeline end-to-end.
It uses sanitized placeholders by default.
USAGE:
    python src/pipeline.py
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "src/collect/collect_prompts.py --out data/raw_prompts.jsonl",
    "src/augment/generate_variations.py --in data/raw_prompts.jsonl --out data/augmented_prompts.jsonl",
    "src/analysis/extract_features.py --in data/augmented_prompts.jsonl --out data/features.npz",
    "src/analysis/analyze_features.py --in data/features.npz --out results/analysis_report.txt"
]

def run_cmd(cmd):
    print(f"[RUN] {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[ERR] Command failed: {cmd}")
        sys.exit(res.returncode)

def main():
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    for cmd in SCRIPTS:
        run_cmd(cmd)
    print("[OK] Pipeline completed. See results/analysis_report.txt")

if __name__ == "__main__":
    main()
