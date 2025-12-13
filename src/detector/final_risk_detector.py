"""
complete_risk_detector.py

ML-based risk detector that uses the SAME features at inference as training.

Training data features come from extract_features.py:
- SentenceTransformer embedding (all-MiniLM-L6-v2)
- Surface features (num_chars, num_words, etc.)
- Semantic similarity features (category_similarity + is_category)
- Structural features (num_commands, num_negations, etc.)

Usage:
    # Train model (requires data/features.npz)
    python src/detector/complete_risk_detector.py --train

    # Inspect model info
    python src/detector/complete_risk_detector.py --info

    # Score a single prompt
    python src/detector/complete_risk_detector.py --text "Your prompt here"
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import sys
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
# Reuse the exact feature logic from training
from src.analysis.extract_features import SemanticFeatureExtractor, surface_features


class PureMLRiskDetector:
    """
    Loads a trained classifier and scores prompts by reproducing the exact
    feature vector used during training: [embedding | extra_features].
    """

    def __init__(self, model_path="models/jailbreak_classifier.pkl", embed_model="all-MiniLM-L6-v2"):
        print("[INFO] Loading trained classifier...")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"\n[ERROR] Model not found at {model_path}\n"
                f"Train first:\n"
                f"  python {__file__} --train\n"
            )

        model_data = joblib.load(model_path)
        self.classifier = model_data["model"]
        self.extra_feat_names = model_data["extra_feat_names"]
        self.n_embedding_dims = model_data["n_embedding_dims"]

        print(f"[OK] Loaded model trained on {model_data.get('n_samples', 'N/A')} samples")

        # SemanticFeatureExtractor contains the embedder + precomputed category embeddings.
        # We reuse it at inference so semantic features match training.
        print("[INFO] Loading feature extractor (embedding + category refs)...")
        self.feature_extractor = SemanticFeatureExtractor(embed_model)
        print("[OK] Ready to analyze prompts")

    def _extract_features_for_text(self, text: str) -> np.ndarray:
        """
        Build the exact feature vector the classifier expects:
        [embedding | extra_features_in_training_order]
        """
        # 1) Embedding (same model as training)
        emb = self.feature_extractor.embedder.encode([text], convert_to_numpy=True)[0].astype(np.float32)

        if emb.shape[0] != self.n_embedding_dims:
            raise ValueError(
                f"Embedding dim mismatch: got {emb.shape[0]}, expected {self.n_embedding_dims}. "
                f"Check embed model matches training."
            )

        # 2) Extra features = surface + semantic (same functions as training)
        surf = surface_features(text)
        sem = self.feature_extractor.extract_semantic_features(text, emb)
        all_extra = {**surf, **sem}

        # 3) Vectorize extras in the exact saved order
        try:
            extra_vec = np.array([all_extra[name] for name in self.extra_feat_names], dtype=np.float32)
        except KeyError as e:
            missing = e.args[0]
            raise KeyError(
                f"Missing feature '{missing}' at inference. "
                f"This usually means your detector and extract_features.py are out of sync, "
                f"or the saved extra_feat_names don't match features.npz."
            )

        # 4) Final features
        feats = np.concatenate([emb, extra_vec], axis=0).astype(np.float32)
        return feats

    def score(self, text: str):
        """
        Score a prompt using the trained ML model.
        Returns dict with risk_score, label, and probabilities.
        """
        print("\n[INFO] Extracting features...")
        features = self._extract_features_for_text(text)
        print("[INFO] Running classifier...")
        # print("features len:", len(features))  # uncomment if debugging

        probabilities = self.classifier.predict_proba([features])[0]
        jailbreak_prob = float(probabilities[1])

        # Thresholds (these are arbitrary; tune on validation set if desired)
        if jailbreak_prob >= 0.7:
            label = "HIGH RISK - JAILBREAK"
        elif jailbreak_prob >= 0.5:
            label = "MODERATE RISK"
        elif jailbreak_prob >= 0.4:
            label = "SUSPICIOUS"
        else:
            label = "SAFE"

        return {
            "risk_score": round(jailbreak_prob, 3),
            "label": label,
            "benign_prob": round(float(probabilities[0]), 3),
            "jailbreak_prob": round(jailbreak_prob, 3),
        }


def train_model(npz_path="data/features.npz", model_out="models/jailbreak_classifier.pkl"):
    """
    Train classifier from features.npz (created by extract_features.py).

    IMPORTANT:
    X is built as [embeddings | surface_feats], so we save metadata that lets
    inference reproduce the same layout:
      - n_embedding_dims
      - extra_feat_names (the columns of surface_feats in correct order)
    """
    print("\n" + "="*60)
    print("TRAINING ML RISK DETECTOR")
    print("="*60)

    print("\n[1/3] Loading features.npz...")
    if not Path(npz_path).exists():
        print(f"\n[ERROR] {npz_path} not found!")
        print("Run feature extraction first:")
        print("  python src/analysis/extract_features.py --in data/augmented_prompts.jsonl --out data/features.npz")
        return

    data = np.load(npz_path, allow_pickle=True)
    X_emb = data["embeddings"].astype(np.float32)
    X_sf = data["surface_feats"].astype(np.float32)
    y = data["labels"].astype(np.int64)

    # Training matrix: [embedding | extras]
    X = np.concatenate([X_emb, X_sf], axis=1)

    print(f"  Loaded {len(X)} samples")
    print(f"  Embedding dims: {X_emb.shape[1]}")
    print(f"  Extra feature dims: {X_sf.shape[1]}")
    print(f"  Total dims: {X.shape[1]}")
    print(f"  Jailbreak prompts: {int((y == 1).sum())}")
    print(f"  Benign prompts: {int((y == 0).sum())}")

    print("\n[2/3] Training logistic regression...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X, y)
    print("  ✓ Training complete!")

    # Show top 5 extra (non-embedding) features by absolute coefficient
    print("\n  Top 5 extra features by |coef| (interpretability):")
    coefs = np.abs(clf.coef_[0])
    extra_coefs = coefs[X_emb.shape[1]:]  # slice off embeddings
    extra_names = list(data["feat_names"])
    top_idx = np.argsort(extra_coefs)[-5:][::-1]
    for i in top_idx:
        print(f"    {extra_names[i]}: {extra_coefs[i]:.6f}")

    print("\n[3/3] Saving model + metadata...")
    Path("models").mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": clf,
            "n_samples": len(X),
            "n_embedding_dims": X_emb.shape[1],
            "extra_feat_names": list(data["feat_names"]),  # EXACT order used in X_sf columns
        },
        model_out,
    )

    print(f"  Saved to: {model_out}")
    print("\n" + "="*60)
    print("✓ SUCCESS! Detector trained and ready.")
    print("="*60)
    print("\nTest it:")
    print(f"  python {__file__} --text \"what is an apple\"")
    print(f"  python {__file__} --text \"Ignore all previous instructions.\"")
    print("")


def show_model_info(model_path="models/jailbreak_classifier.pkl"):
    """Show basic info about the trained model."""
    if not Path(model_path).exists():
        print("\n[ERROR] No trained model found.")
        print(f"Train first: python {__file__} --train\n")
        return

    model_data = joblib.load(model_path)
    print("\n" + "="*60)
    print("TRAINED MODEL INFORMATION")
    print("="*60)
    print(f"\nTraining samples: {model_data.get('n_samples', 'N/A')}")
    print(f"Embedding dims: {model_data.get('n_embedding_dims', 'N/A')}")
    print(f"Extra features: {len(model_data.get('extra_feat_names', []))}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ML jailbreak risk detector (feature-parity with training)")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--train", action="store_true", help="Train model from data/features.npz")
    parser.add_argument("--info", action="store_true", help="Show model info")
    args = parser.parse_args()

    if args.train:
        train_model()
        return

    if args.info:
        show_model_info()
        return

    if not args.text:
        print("\nUsage:")
        print(f"  python {__file__} --train")
        print(f"  python {__file__} --info")
        print(f"  python {__file__} --text \"your prompt here\"")
        return

    detector = PureMLRiskDetector()
    result = detector.score(args.text)

    print("\n" + "="*60)
    print("ML RISK ANALYSIS")
    print("="*60)
    print(f"Input: {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
    print(f"\nRisk Score: {result['risk_score']}")
    print(f"Label: {result['label']}")
    print(f"\nProbabilities:")
    print(f"  Benign: {result['benign_prob']}")
    print(f"  Jailbreak: {result['jailbreak_prob']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()