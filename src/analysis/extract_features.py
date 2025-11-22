"""
extract_features.py
Extract embedding + surface features and save to a compressed .npz
USAGE:
    python src/analysis/extract_features.py --in data/augmented_prompts.jsonl --out data/features.npz
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
import nltk
from nltk import sent_tokenize, word_tokenize
nltk.download('punkt', quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = spacy.blank("en")

def surface_features(text):
    sents = sent_tokenize(text)
    words = word_tokenize(text)
    num_chars = len(text)
    num_words = len(words)
    avg_word_len = sum(len(w) for w in words)/num_words if num_words>0 else 0.0
    num_sents = max(1, len(sents))
    punct_count = sum(1 for ch in text if ch in '.,;:!?')
    is_question = 1 if '?' in text else 0
    doc = nlp(text)
    num_verbs = sum(1 for tok in doc if tok.pos_ == 'VERB') if doc else 0
    num_subords = sum(1 for tok in doc if tok.dep_ == 'mark') if doc else 0
    return {
        "num_chars": num_chars,
        "num_words": num_words,
        "avg_word_len": avg_word_len,
        "num_sents": num_sents,
        "punct_count": punct_count,
        "is_question": is_question,
        "num_verbs": num_verbs,
        "num_subords": num_subords
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inpath", type=str, default="data/augmented_prompts.jsonl")
    parser.add_argument("--out", type=str, default="data/features.npz")
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    Path("data").mkdir(parents=True, exist_ok=True)
    prompts = []
    metas = []
    with open(args.inpath, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            prompts.append(str(r.get("prompt","")))
            metas.append(r)

    print("[INFO] Computing embeddings with", args.embed_model)
    embedder = SentenceTransformer(args.embed_model)
    embeddings = embedder.encode(prompts, show_progress_bar=True, convert_to_numpy=True)

    surface_list = []
    for t in prompts:
        surface_list.append(surface_features(t))
    feat_df = pd.DataFrame(surface_list).fillna(0.0)
    labels = np.array([int(m.get("label",0)) for m in metas], dtype=np.int64)

    np.savez_compressed(args.out,
                        embeddings=embeddings.astype(np.float32),
                        surface_feats=feat_df.to_numpy(dtype=np.float32),
                        feat_names=feat_df.columns.tolist(),
                        labels=labels,
                        meta=metas)
    print(f"[OK] Saved features to {args.out}")

if __name__ == "__main__":
    main()
