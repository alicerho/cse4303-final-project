"""
load_github_jailbreak_data.py
Downloads the jailbreak_llms CSV dataset from GitHub (using raw URLs)
and converts them into augmented_prompts.jsonl for feature extraction.

Usage:
    python src/data/load_github_jailbreak_data.py
"""

import pandas as pd
from pathlib import Path
import json
import requests

# ------------------------------------------------------------
# URLs for RAW GitHub files
# ------------------------------------------------------------
BASE_RAW = "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/"

CSV_FILES = {
    "data/forbidden_question/forbidden_question_set.csv": 1,
    "data/prompts/jailbreak_prompts_2023_05_07.csv": 1,
    "data/prompts/jailbreak_prompts_2023_12_25.csv": 1,
    "data/prompts/regular_prompts_2023_05_07.csv": 0,
    "data/prompts/regular_prompts_2023_12_25.csv": 0,
}


# ------------------------------------------------------------
# Download helper
# ------------------------------------------------------------
def download_file(remote_path, local_path):
    raw_url = BASE_RAW + remote_path
    print(f"[INFO] Downloading {raw_url}")

    resp = requests.get(raw_url)
    if resp.status_code == 200:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(resp.content)
        print(f"[OK] Saved to {local_path}")
    else:
        print(f"[ERROR] Could not download {raw_url} (status {resp.status_code})")


# ------------------------------------------------------------
# Load CSV → list of dicts {prompt: "...", label: 0/1}
# ------------------------------------------------------------
def load_csv(path, label):
    df = pd.read_csv(path)

    # column names differ → detect text column
    for col in ["prompt", "question", "text"]:
        if col in df.columns:
            text_col = col
            break
    else:
        raise ValueError(f"No prompt/question/text column found in {path}")

    entries = []
    for p in df[text_col].astype(str):
        p = p.strip()
        if p:
            entries.append({"prompt": p, "label": label})

    return entries


# ------------------------------------------------------------
# Main logic
# ------------------------------------------------------------
def main():
    print("\n=== Step 1: Downloading all CSV files from GitHub ===")
    for remote_path, label in CSV_FILES.items():
        download_file(remote_path, remote_path)

    print("\n=== Step 2: Loading CSVs and building prompt dataset ===")
    all_entries = []

    for csv_path, label in CSV_FILES.items():
        path = Path(csv_path)
        if path.exists():
            print(f"[INFO] Loading {csv_path} (label={label})")
            all_entries.extend(load_csv(path, label))
        else:
            print(f"[WARN] Missing file: {csv_path}")

    # Write jsonl
    out_path = Path("data/augmented_prompts.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\n=== DONE! ===")
    print(f"[OK] Saved {len(all_entries)} total prompts to {out_path}\n")


if __name__ == "__main__":
    main()
