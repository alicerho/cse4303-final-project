"""
pipeline.py
Complete research pipeline that runs all steps end-to-end.
USAGE:
    python src/pipeline.py
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    # Step 1: Data Collection
    ("Data Collection", "python src/data/load_github_jailbreak_data.py"),
    
    # Step 2: Generate Variations
    ("Generate Variations", "python src/augment/generate_variations.py --in data/augmented_prompts.jsonl --out data/augmented_v2.jsonl"),
    
    # Step 3: Feature Extraction
    ("Feature Extraction", "python src/analysis/extract_features.py --in data/augmented_v2.jsonl --out data/features.npz"),
    
    # Step 4: Train & Analyze
    ("Train Classifier", "python src/analysis/analyze_features.py --in data/features.npz --out results/analysis_report.txt"),
    
    # Step 5: Test on LLM
    ("LLM Testing", "python src/evaluate/run_model_tests.py --prompts data/augmented_v2.jsonl --model gpt2 --out results/model_outputs.jsonl --max 100"),
    
    # Step 6: Classify Outputs
    ("Classify Outputs", "python src/classify_outputs.py"),
    
    # Step 7: Interpret Patterns
    ("Interpret Patterns", "python src/analysis/interpret_patterns.py"),
]

def run_cmd(name, cmd):
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"[RUN] {cmd}")
    print('='*60)
    
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[ERR] {name} failed!")
        sys.exit(res.returncode)
    print(f"[OK] {name} completed")

def main():
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("JAILBREAK ANALYSIS PIPELINE")
    print("="*60)
    
    for name, cmd in SCRIPTS:
        run_cmd(name, cmd)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nOutput files:")
    print("  - results/analysis_report.txt")
    print("  - results/classification_report.txt")
    print("  - results/interpretation_report.txt")

if __name__ == "__main__":
    main()