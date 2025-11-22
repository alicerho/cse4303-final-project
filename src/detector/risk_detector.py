"""
risk_detector.py
A small rule-based detector that blends simple heuristics with an optional trained classifier.
USAGE:
    python src/detector/risk_detector.py --text "some prompt" --model models/firewall.joblib
"""
import argparse
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

def heuristic_score(text):
    score = 0.0
    low = text.lower()
    # NOTE: do not include lists of exploit tokens. Use generic patterns only.
    if "ignore" in low or "disregard" in low:
        score += 0.5
    if "pretend" in low or "act as" in low:
        score += 0.25
    if len(text) > 400:
        score += 0.15
    if "please" in low:
        score -= 0.05
    # cap
    return max(0.0, min(1.0, score))

def load_model(path):
    try:
        data = joblib.load(path)
        return data.get("model"), data.get("embed_dim"), data.get("feat_names")
    except Exception as e:
        print("[WARN] Could not load model:", e)
        return None, None, None

def score_text(text, clf_path=None, embed_model="all-MiniLM-L6-v2"):
    heur = heuristic_score(text)
    clf_prob = None
    if clf_path is not None:
        clf, embed_dim, feat_names = load_model(clf_path)
        if clf is not None:
            emb = SentenceTransformer(embed_model).encode([text], convert_to_numpy=True)
            # create zeros for surface features if missing
            zeros = np.zeros((1, len(feat_names))) if feat_names is not None else np.zeros((1,0))
            X = np.concatenate([emb, zeros], axis=1)
            clf_prob = float(clf.predict_proba(X)[:,1][0])
    # blend: 70% clf (if available) + 30% heur; otherwise heur only
    if clf_prob is None:
        final = heur
    else:
        final = 0.7*clf_prob + 0.3*heur
    label = "JAILBREAK" if final >= 0.5 else "SAFE"
    return {"risk_score": final, "label": label, "clf_prob": clf_prob, "heuristic": heur}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model", type=str, default=None, help="Optional trained model path")
    args = parser.parse_args()
    out = score_text(args.text, args.model)
    print(out)

if __name__ == "__main__":
    main()
