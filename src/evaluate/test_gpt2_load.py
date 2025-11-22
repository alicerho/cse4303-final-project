from transformers import AutoModelForCausalLM, AutoTokenizer
import time

MODEL_PATH = "src/data/models/gpt2"   # <-- update if needed

print("[1] Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH)

print("[2] Loading model...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
t1 = time.time()

print(f"[OK] Model loaded in {t1 - t0:.2f} seconds")

# Verify model type
print(f"Model class: {model.__class__.__name__}")
print(f"Model framework: {'PyTorch' if model.__class__.__module__.startswith('transformers') else 'UNKNOWN'}")

print("\nSanity generation test:")
inputs = tok("Hello world", return_tensors="pt")
t0 = time.time()
out = model.generate(**inputs, max_length=30)
t1 = time.time()

print("Generated:", tok.decode(out[0]))
print(f"Generation time: {t1 - t0:.3f} sec")