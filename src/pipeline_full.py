import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    # Data + Training
     "python src/data/load_wildjailbreak.py",
     "python src/analysis/extract_features.py --in data/augmented_prompts.jsonl --out data/features.npz",
     "python src/analysis/analyze_features.py --in data/features.npz --out results/analysis_report.txt",
    
    # Effectiveness Testing
    'python src/evaluate/run_model_tests.py --prompts data/test_balanced.jsonl --model EleutherAI/gpt-neo-1.3B --out results/model_outputs.jsonl --max 200',
    "python src/classify_outputs.py",
    
    # Interpretation
    "python src/analysis/interpret_patterns.py"
]

def run_cmd(cmd):
    print(f"\n{'='*60}")
    print(f"[RUN] {cmd}")
    print('='*60)
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[ERR] Command failed: {cmd}")
        sys.exit(res.returncode)

def main():
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("STARTING COMPLETE JAILBREAK RESEARCH PIPELINE")
    print("="*60)
    
    for i, cmd in enumerate(SCRIPTS, 1):
        print(f"\n>>> Step {i}/{len(SCRIPTS)}")
        run_cmd(cmd)
    
    print("\n" + "="*60)
    print("[OK] Pipeline completed!")
    print("="*60)
    print("\nGenerated reports:")
    print("  - results/analysis_report.txt (feature importance)")
    print("  - results/classification_report.txt (effectiveness)")
    print("  - results/interpretation_report.txt (WHY patterns work)")
    print("\nNext: Review these reports for your paper!")

if __name__ == "__main__":
    main()