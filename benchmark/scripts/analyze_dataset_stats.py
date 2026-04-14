"""Analyze Dataset Statistics: Sentence Count, Vocabulary Size, Lengths"""
import json
from pathlib import Path
from collections import Counter

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = BASE_DIR / "benchmark"
CONTINUOUS_DATA_DIR = BENCHMARK_DIR / "continuous" / "data"
SENTENCE_DATA_DIR = BENCHMARK_DIR / "sentence_level" / "data"
OUTPUT_DIR = BENCHMARK_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_continuous_dataset():
    """Analyze continuous dataset"""
    dataset_file = CONTINUOUS_DATA_DIR / "continuous_dataset.real.json"
    
    if not dataset_file.exists():
        print(f"⚠ Continuous dataset not found: {dataset_file}")
        return None
    
    print("[STATS] Analyzing continuous dataset...")
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get('samples', [])
    num_samples = len(samples)
    
    # Collect all glosses
    all_glosses = []
    lengths = []
    
    for sample in samples:
        gloss_list = sample.get('sentence_gloss', [])
        all_glosses.extend(gloss_list)
        lengths.append(len(gloss_list))
    
    unique_glosses = set(all_glosses)
    vocab_size = len(unique_glosses)
    gloss_freq = Counter(all_glosses)
    
    stats = {
        'dataset_type': 'continuous',
        'total_samples': num_samples,
        'vocabulary_size': vocab_size,
        'total_sentence_words': len(all_glosses),
        'sentence_length': {
            'min': min(lengths) if lengths else 0,
            'max': max(lengths) if lengths else 0,
            'mean': sum(lengths) / len(lengths) if lengths else 0,
            'median': sorted(lengths)[len(lengths)//2] if lengths else 0,
        },
        'top_20_glosses': dict(gloss_freq.most_common(20)),
        'unique_glosses_sample': list(unique_glosses)[:20]  # First 20 for reference
    }
    
    print(f"  Samples: {num_samples}")
    print(f"  Vocabulary: {vocab_size} unique glosses")
    print(f"  Sentence length: min={stats['sentence_length']['min']}, "
          f"max={stats['sentence_length']['max']}, "
          f"mean={stats['sentence_length']['mean']:.1f}")
    
    return stats


def analyze_sentence_dataset():
    """Analyze sentence-level dataset"""
    dataset_file = SENTENCE_DATA_DIR / "sentence_dataset.synthetic.small.json"
    
    if not dataset_file.exists():
        print(f"⚠ Sentence dataset not found: {dataset_file}")
        return None
    
    print("[STATS] Analyzing sentence-level dataset...")
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get('samples', [])
    num_samples = len(samples)
    
    # Split counts
    splits = Counter()
    dialects = Counter()
    signers = Counter()
    lengths = []
    all_glosses = []
    
    for sample in samples:
        splits[sample.get('split', 'unknown')] += 1
        dialects[sample.get('dialect', 'unknown')] += 1
        signers[sample.get('signer_id', 'unknown')] += 1
        
        gloss_list = sample.get('sentence_gloss', [])
        all_glosses.extend(gloss_list)
        lengths.append(len(gloss_list))
    
    unique_glosses = set(all_glosses)
    vocab_size = len(unique_glosses)
    
    stats = {
        'dataset_type': 'sentence_level',
        'total_samples': num_samples,
        'vocabulary_size': vocab_size,
        'total_sentence_words': len(all_glosses),
        'split_distribution': dict(splits),
        'dialect_distribution': dict(dialects),
        'signer_count': len(signers),
        'sentence_length': {
            'min': min(lengths) if lengths else 0,
            'max': max(lengths) if lengths else 0,
            'mean': sum(lengths) / len(lengths) if lengths else 0,
            'median': sorted(lengths)[len(lengths)//2] if lengths else 0,
        }
    }
    
    print(f"  Samples: {num_samples}")
    print(f"  Splits: {dict(splits)}")
    print(f"  Dialects: {dict(dialects)}")
    print(f"  Signers: {len(signers)}")
    print(f"  Vocabulary: {vocab_size} unique glosses")
    print(f"  Sentence length: min={stats['sentence_length']['min']}, "
          f"max={stats['sentence_length']['max']}, "
          f"mean={stats['sentence_length']['mean']:.1f}")
    
    return stats


def main():
    """Analyze all datasets"""
    print("\n" + "="*60)
    print("DATASET STATISTICS ANALYSIS")
    print("="*60 + "\n")
    
    results = {
        'continuous': analyze_continuous_dataset(),
        'sentence_level': analyze_sentence_dataset()
    }
    
    # Save results
    output_file = OUTPUT_DIR / "dataset_statistics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[STATS] Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results.get('continuous'):
        cont = results['continuous']
        print(f"\n📊 CONTINUOUS DATASET")
        print(f"   Samples: {cont['total_samples']}")
        print(f"   Vocabulary: {cont['vocabulary_size']} glosses")
        print(f"   Avg sentence length: {cont['sentence_length']['mean']:.1f} words")
    
    if results.get('sentence_level'):
        sent = results['sentence_level']
        print(f"\n📊 SENTENCE-LEVEL DATASET")
        print(f"   Samples: {sent['total_samples']}")
        print(f"   Splits: {sent['split_distribution']}")
        print(f"   Dialects: {sent['dialect_distribution']}")
        print(f"   Vocabulary: {sent['vocabulary_size']} glosses")
        print(f"   Avg sentence length: {sent['sentence_length']['mean']:.1f} words")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
