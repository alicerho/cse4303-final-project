"""
io_utils.py
Small helpers for reading/writing JSON/CSV safely.
"""
import json
from pathlib import Path
import pandas as pd

def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def save_df_csv(path, df):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def load_csv(path):
    return pd.read_csv(path)
