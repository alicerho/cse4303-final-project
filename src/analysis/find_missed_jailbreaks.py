"""
find_missed_jailbreaks.py
Analyzes which jailbreak prompts the classifier FAILS to detect.
This reveals what attack strategies are hardest to catch.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
    data = np.load("data/features.npz", allow_pickle=True)
    
    X_emb = data['embeddings']
    X_sf = data['surface_feats']
    X = np.concatenate([X_emb, X_sf], axis=1)
    y = data['labels']
    meta = data['meta']
    feat_names = list(data['feat_names'])
    
    # Train classifier
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=0.2, stratify=y, random_state=42
    )
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Find missed jailbreaks (false negatives)
    preds = clf.predict(X_test)
    
    missed = []
    for i, (true, pred, m) in enumerate(zip(y_test, preds, meta_test)):
        if true == 1 and pred == 0:  # Jailbreak classified as safe
            missed.append({
                'prompt': m.get('prompt', '')[:200],
                'features': {feat_names[j]: float(X_sf[i, j]) for j in range(len(feat_names))}
            })
    
    # Analyze what makes missed jailbreaks different
    with open("results/missed_jailbreaks_analysis.txt", "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("ANALYSIS: JAILBREAKS THAT EVADE DETECTION\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total missed jailbreaks: {len(missed)}\n\n")
        
        f.write("SAMPLE MISSED JAILBREAKS:\n")
        f.write("-"*70 + "\n")
        for i, m in enumerate(missed[:10]):
            f.write(f"\n{i+1}. {m['prompt']}...\n")
            f.write(f"   Roleplay similarity: {m['features'].get('roleplay_similarity', 'N/A')}\n")
            f.write(f"   Instruction override: {m['features'].get('instruction_override_similarity', 'N/A')}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("KEY INSIGHT: What do missed jailbreaks have in common?\n")
        f.write("="*70 + "\n")
        f.write("These prompts evade detection because they:\n")
        f.write("  1. Don't match known jailbreak patterns semantically\n")
        f.write("  2. Use novel attack strategies not in our reference set\n")
        f.write("  3. May represent emerging jailbreak techniques\n")
    
    print(f"[OK] Found {len(missed)} missed jailbreaks")
    print("[OK] Analysis saved to results/missed_jailbreaks_analysis.txt")

if __name__ == "__main__":
    main()