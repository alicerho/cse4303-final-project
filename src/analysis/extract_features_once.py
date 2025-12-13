"""
extract_features_once.py

Extract features for a SINGLE prompt, matching extract_features.py exactly.
Used at inference time to ensure feature parity with training.

Usage (from another file):
    from extract_features_once import extract_features_once
"""

import numpy as np
import joblib

# Import the SAME code used during training
from src.analysis.extract_features import (
    SemanticFeatureExtractor,
    surface_features
)

def extract_features_once(
    text: str,
    model_path: str = "models/jailbreak_classifier.pkl",
    embed_model: str = "all-MiniLM-L6-v2"
) -> np.ndarray:
    """
    Extract full feature vector for one prompt.

    Returns:
        np.ndarray of shape (n_features,), dtype float32
        Format: [embedding | surface_feats | semantic_feats]
    """

    # --------------------------------------------------
    # 1. Load model metadata (to get feature order)
    # --------------------------------------------------
    model_data = joblib.load(model_path)

    # IMPORTANT: this must be the same list used in training
    extra_feat_names = model_data["extra_feat_names"]
    n_emb_dims = model_data["n_embedding_dims"]

    # --------------------------------------------------
    # 2. Initialize feature extractor (same as training)
    # --------------------------------------------------
    extractor = SemanticFeatureExtractor(embed_model)

    # --------------------------------------------------
    # 3. Compute embedding
    # --------------------------------------------------
    emb = extractor.embedder.encode(
        [text],
        convert_to_numpy=True
    )[0].astype(np.float32)

    # Safety check
    if emb.shape[0] != n_emb_dims:
        raise ValueError(
            f"Embedding dim mismatch: got {emb.shape[0]}, expected {n_emb_dims}"
        )

    # --------------------------------------------------
    # 4. Compute surface + semantic features
    # --------------------------------------------------
    surf = surface_features(text)
    sem = extractor.extract_semantic_features(text, emb)

    # Merge into single dict
    all_extra_feats = {**surf, **sem}

    # --------------------------------------------------
    # 5. Build extra feature vector IN TRAINING ORDER
    # --------------------------------------------------
    extra_vec = np.array(
        [all_extra_feats[name] for name in extra_feat_names],
        dtype=np.float32
    )

    # --------------------------------------------------
    # 6. Final model input: [embedding | extra_features]
    # --------------------------------------------------
    features = np.concatenate([emb, extra_vec], axis=0)

    return features