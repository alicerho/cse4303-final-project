"""
model_utils.py
Helpers to load local HF models safely for offline testing.
This file intentionally avoids any code that calls hosted APIs.
"""
from transformers import pipeline, set_seed
from sentence_transformers import SentenceTransformer

def load_text_generator(model_name="gpt2", device=-1):
    """
    Loads a local text-generation pipeline. Ensure model is downloaded locally.
    device=-1 forces CPU use; change to int(0) for GPU if available.
    """
    # user must ensure the model_name points to a locally available HF model
    gen = pipeline("text-generation", model=model_name, device=device)
    set_seed(42)
    return gen

def load_embedder(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)
