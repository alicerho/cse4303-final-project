# cse4303-final-project

# LLM Jailbreak Study (safe-by-default)

This repository provides a safe, runnable scaffold for studying jailbreak prompts and building a simple prompt-risk detector.
It uses sanitized placeholders by default so you do not accidentally publish sensitive prompt text.

## Structure
- data/  -- datasets and outputs (sanitized)
- src/collect -- collect_prompts.py
- src/augment -- generate_variations.py
- src/evaluate -- run_model_tests.py
- src/analysis -- extract_features.py, analyze_features.py
- src/detector -- risk_detector.py
- src/utils -- helpers
- src/pipeline.py -- run the demo pipeline

## Quick demo (safe placeholders)
1. Install dependencies:
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python -m nltk.downloader punkt wordnet

2. Run the pipeline:
   python src/pipeline.py

3. Train a classifier (optional after extracting features):
   python src/analysis/analyze_features.py --in data/features.npz --out results/analysis_report.txt

## Important safety notes
- By default the code creates and uses sanitized placeholders (e.g. "[SANITIZED_JAILBREAK...]").
- If you decide to use real public jailbreak datasets, download them locally, keep them off any public repo,
  and follow your instructor/IRB guidance.
- Do not run evaluation against hosted APIs (OpenAI, Anthropic, etc.) for jailbreak testing.

## Next steps for full experiments
- Replace sanitized placeholders with your locally stored raw dataset (kept private).
- Create paraphrases/obfuscations locally.
- Train classifier models and evaluate robustness.
