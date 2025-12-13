"""
analyze_features.py
Run basic model training + feature importance and save a short report.
USAGE:
    python src/analysis/analyze_features.py --in data/features.npz --out results/analysis_report.txt
"""
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_file", type=str, default="data/features.npz")
    parser.add_argument("--out", dest="output_file", type=str, default="results/analysis_report.txt")
    args = parser.parse_args()

    Path("results").mkdir(parents=True, exist_ok=True)

    # load data safely
    data = np.load(args.input_file, allow_pickle=True)

    X_emb = data['embeddings']
    X_sf = data['surface_feats']
    X = np.concatenate([X_emb, X_sf], axis=1)
    y = data['labels']
    feat_names = list(data['feat_names']) + [f"emb_{i}" for i in range(X_emb.shape[1])]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Logistic regression baseline
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)


    Path("models").mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": clf,
        "n_embedding_dims": X_emb.shape[1],
        "extra_feat_names": list(data["feat_names"]),
        "n_samples": len(X)
    }, "models/jailbreak_classifier.pkl")

    print("[OK] Saved trained model to models/jailbreak_classifier.pkl")

    probs = clf.predict_proba(X_test)[:,1]
    preds = (probs >= 0.5).astype(int)
    report = classification_report(y_test, preds)

    try:
        auc = roc_auc_score(y_test, probs)
    except:
        auc = None

    # Random forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = getattr(rf, "feature_importances_", None)

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("=== Logistic Regression Report ===\n")
        f.write(report + "\n")
        f.write(f"ROC AUC: {auc}\n\n")

        if importances is not None:
            n_surface = X_sf.shape[1]
            # surface features are at the end of X
            surface_importances = importances[-n_surface:]

            f.write("=== Surface Feature Importances ===\n")
            for name, imp in sorted(zip(data["feat_names"], surface_importances),
                                    key=lambda x: -x[1]):
                f.write(f"{name}: {imp:.6f}\n")
    print("X.shape[1]:", X.shape[1])
    print(f"[OK] Wrote analysis report to {args.output_file}")

if __name__ == "__main__":
    main()
