"""
interpret_patterns.py
Connects statistical feature importance to real-world jailbreak effectiveness.
Model tested: GPT-Neo-1.3B (EleutherAI)
Usage: python src/analysis/interpret_patterns.py
"""
import json
import numpy as np
from collections import defaultdict

def load_data():
    """Load all analysis artifacts"""
    features = np.load("data/features.npz", allow_pickle=True)
    
    with open("results/categorized_outputs.json", "r") as f:
        outputs = json.load(f)
    
    return features, outputs

def analyze_pattern_effectiveness(features, outputs):
    """Cross-reference: Which features predict actual jailbreak success?"""
    X_sf = features['surface_feats']
    feat_names = list(features['feat_names'])
    labels = features['labels']
    
    results = {}
    
    for i, feat_name in enumerate(feat_names):
        feat_values = X_sf[:, i]
        
        jailbreak_avg = feat_values[labels == 1].mean()
        benign_avg = feat_values[labels == 0].mean()
        
        results[feat_name] = {
            'jailbreak_avg': jailbreak_avg,
            'benign_avg': benign_avg,
            'difference': jailbreak_avg - benign_avg
        }
    
    return results

def analyze_variant_effectiveness(outputs):
    """Which obfuscation techniques actually work?"""
    variant_stats = defaultdict(lambda: {'total': 0, 'complied': 0})
    
    for output in outputs:
        variant = output.get('variant', 'original')
        category = output.get('category', 'UNCLEAR')
        
        variant_stats[variant]['total'] += 1
        if category == 'COMPLIED':
            variant_stats[variant]['complied'] += 1
    
    for variant in variant_stats:
        total = variant_stats[variant]['total']
        complied = variant_stats[variant]['complied']
        variant_stats[variant]['success_rate'] = complied / total if total > 0 else 0
    
    return dict(variant_stats)

def generate_interpretation_report():
    """Main analysis function"""
    features, outputs = load_data()
    
    pattern_analysis = analyze_pattern_effectiveness(features, outputs)
    variant_analysis = analyze_variant_effectiveness(outputs)
    
    with open("results/interpretation_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("JAILBREAK PATTERN INTERPRETATION REPORT\n")
        f.write("Model Tested: GPT-Neo-1.3B (EleutherAI)\n")
        f.write("Connecting Statistical Importance to Real-World Effectiveness\n")
        f.write("=" * 70 + "\n\n")
        
        # Section 1: Linguistic Pattern Analysis
        f.write("SECTION 1: WHY CERTAIN LINGUISTIC FEATURES MATTER\n")
        f.write("-" * 70 + "\n\n")
        
        sorted_patterns = sorted(
            pattern_analysis.items(),
            key=lambda x: abs(x[1]['difference']),
            reverse=True
        )
        
        for feat_name, stats in sorted_patterns:
            f.write(f"{feat_name}:\n")
            f.write(f"  Jailbreak prompts avg: {stats['jailbreak_avg']:.2f}\n")
            f.write(f"  Benign prompts avg: {stats['benign_avg']:.2f}\n")
            f.write(f"  Difference: {stats['difference']:.2f}\n")
            
            # Interpret the meaning
            if 'instruction_override' in feat_name.lower():
                if stats['difference'] > 0:
                    f.write(f"  → Jailbreaks have HIGHER instruction override similarity\n")
                    f.write(f"  → WHY: Attempts to reset model context with 'ignore previous instructions'\n")
            elif 'roleplay' in feat_name.lower():
                if stats['difference'] > 0:
                    f.write(f"  → Jailbreaks have HIGHER roleplay similarity\n")
                    f.write(f"  → WHY: Persona manipulation creates distance from safety guidelines\n")
            elif 'manipulation' in feat_name.lower():
                if stats['difference'] > 0:
                    f.write(f"  → Jailbreaks have HIGHER manipulation similarity\n")
                    f.write(f"  → WHY: Deceptive framing attempts to trick the model\n")
            elif 'negation' in feat_name.lower():
                if stats['difference'] > 0:
                    f.write(f"  → Jailbreaks use MORE negation words (not, never, don't)\n")
                    f.write(f"  → WHY: Complex logical structures may confuse simple filters\n")
            elif 'subord' in feat_name.lower():
                if stats['difference'] > 0:
                    f.write(f"  → Jailbreaks use MORE subordinate clauses\n")
                    f.write(f"  → WHY: Nested conditions help bypass keyword filters\n")
            elif 'question' in feat_name.lower():
                if stats['difference'] > 0:
                    f.write(f"  → Jailbreaks are MORE often phrased as questions\n")
                elif stats['difference'] < 0:
                    f.write(f"  → Jailbreaks are LESS often questions\n")
                    f.write(f"  → WHY: Statements/commands may bypass safety checks better\n")
            
            f.write("\n")
        
        # Section 2: Obfuscation Technique Effectiveness
        f.write("\nSECTION 2: WHICH OBFUSCATION TECHNIQUES ACTUALLY WORK\n")
        f.write("-" * 70 + "\n\n")
        f.write("Tested on: GPT-Neo-1.3B (EleutherAI)\n\n")
        
        sorted_variants = sorted(
            variant_analysis.items(),
            key=lambda x: x[1]['success_rate'],
            reverse=True
        )
        
        for variant, stats in sorted_variants:
            success_pct = stats['success_rate'] * 100
            f.write(f"{variant}:\n")
            f.write(f"  Success rate: {success_pct:.1f}% ({stats['complied']}/{stats['total']})\n")
            
            if variant == 'lowercase':
                if success_pct > 20:
                    f.write(f"  → WHY IT WORKS: Bypasses exact-match keyword filters\n")
                else:
                    f.write(f"  → WHY IT FAILS: Semantic meaning preserved despite case change\n")
            elif 'Briefly' in variant:
                if success_pct > 20:
                    f.write(f"  → WHY IT WORKS: Adds legitimate-seeming instruction prefix\n")
                else:
                    f.write(f"  → WHY IT FAILS: Prefix doesn't change underlying request\n")
            elif 'Please explain' in variant:
                if success_pct > 20:
                    f.write(f"  → WHY IT WORKS: Polite suffix reduces suspicion scoring\n")
                else:
                    f.write(f"  → WHY IT FAILS: Politeness doesn't mask harmful intent\n")
            
            f.write("\n")
        
        # Section 3: Key Findings
        f.write("\nSECTION 3: KEY RESEARCH FINDINGS\n")
        f.write("-" * 70 + "\n\n")
        
        top_feature = sorted_patterns[0]
        most_effective_variant = sorted_variants[0]
        
        f.write("1. MOST PREDICTIVE LINGUISTIC FEATURE:\n")
        f.write(f"   {top_feature[0]} shows {abs(top_feature[1]['difference']):.2f} difference\n")
        f.write(f"   between jailbreak and benign prompts.\n\n")
        
        f.write("2. MOST EFFECTIVE OBFUSCATION TECHNIQUE:\n")
        f.write(f"   {most_effective_variant[0]} achieves {most_effective_variant[1]['success_rate']*100:.1f}% success rate.\n\n")
        
        f.write("3. IMPLICATIONS FOR DEFENSE:\n")
        f.write(f"   - Detectors should weight {top_feature[0]} heavily\n")
        f.write(f"   - Simple case normalization is insufficient\n")
        f.write(f"   - Semantic analysis (embeddings) is necessary\n\n")
        
        f.write("4. MODEL OBSERVATIONS (GPT-Neo-1.3B):\n")
        f.write(f"   - GPT-Neo-1.3B has no explicit safety training\n")
        f.write(f"   - Response patterns reflect capability limits, not safety refusals\n")
        f.write(f"   - Testing on safety-tuned models would show actual bypass rates\n\n")
    
    print("[OK] Interpretation report saved to results/interpretation_report.txt")

if __name__ == "__main__":
    generate_interpretation_report()