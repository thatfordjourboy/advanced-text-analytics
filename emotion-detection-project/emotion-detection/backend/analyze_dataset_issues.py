#!/usr/bin/env python3
"""
Academic dataset analysis script - NO DATA MODIFICATION.
This script analyzes the ConvLab dataset to understand its characteristics,
limitations, and challenges for emotion detection without changing any labels.
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_dataset_characteristics():
    """Analyze the original dataset without modifying any labels."""
    
    print("🔍 Academic Dataset Analysis - NO MODIFICATIONS")
    print("=" * 60)
    
    # Load the ORIGINAL dataset (not modified)
    original_path = Path("data/dialogues.json")
    
    if not original_path.exists():
        print("❌ Original dataset not found.")
        return
    
    print("📊 Loading ORIGINAL dataset for analysis...")
    with open(original_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    print(f"📈 Original dataset loaded: {len(original_data)} dialogues")
    
    # Extract all utterances
    utterances = []
    for dialogue in original_data:
        turns = dialogue.get('turns', [])
        for turn in turns:
            text = turn.get('utterance', '').strip()
            emotion = turn.get('emotion', '').strip()
            if text and emotion:
                utterances.append({
                    'text': text,
                    'emotion': emotion,
                    'dialogue_id': dialogue.get('dialogue_id', ''),
                    'turn_idx': turns.index(turn)
                })
    
    print(f"📝 Total utterances: {len(utterances)}")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(utterances)
    
    # 1. Class Distribution Analysis
    print("\n📊 1. CLASS DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    emotion_counts = df['emotion'].value_counts()
    total_samples = len(df)
    
    print("Emotion Class Distribution:")
    for emotion, count in emotion_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  {emotion:15}: {count:6} samples ({percentage:5.1f}%)")
    
    # Calculate imbalance metrics
    max_count = emotion_counts.max()
    min_count = emotion_counts.min()
    imbalance_ratio = max_count / min_count
    
    print(f"\n📈 Imbalance Analysis:")
    print(f"  Most frequent class: {emotion_counts.idxmax()} ({max_count} samples)")
    print(f"  Least frequent class: {emotion_counts.idxmin()} ({min_count} samples)")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
    print(f"  This is a {'SEVERE' if imbalance_ratio > 10 else 'MODERATE' if imbalance_ratio > 5 else 'MILD'} class imbalance")
    
    # 2. Text Length Analysis
    print("\n📏 2. TEXT LENGTH ANALYSIS")
    print("-" * 40)
    
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print("Text Length Statistics:")
    print(f"  Average characters: {df['text_length'].mean():.1f}")
    print(f"  Average words: {df['word_count'].mean():.1f}")
    print(f"  Min characters: {df['text_length'].min()}")
    print(f"  Max characters: {df['text_length'].max()}")
    
    # 3. Potential Labeling Issues Analysis
    print("\n⚠️  3. POTENTIAL LABELING ISSUES ANALYSIS")
    print("-" * 40)
    
    # Look for texts that might be mislabeled (without changing them)
    print("Analyzing potential mislabeling patterns...")
    
    # Check for emotional keywords in "no emotion" texts
    emotional_keywords = {
        'sadness': ['sad', 'unhappy', 'depressed', 'heartbroken', 'devastated', 'sorry', 'regret'],
        'fear': ['scared', 'frightened', 'terrified', 'afraid', 'worried', 'anxious', 'nervous'],
        'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated'],
        'happiness': ['happy', 'joyful', 'excited', 'thrilled', 'delighted', 'pleased']
    }
    
    potential_issues = {}
    
    for emotion, keywords in emotional_keywords.items():
        # Find "no emotion" texts that contain emotional keywords
        no_emotion_texts = df[df['emotion'] == 'no emotion']
        
        for keyword in keywords:
            matches = no_emotion_texts[no_emotion_texts['text'].str.contains(keyword, case=False)]
            if len(matches) > 0:
                if emotion not in potential_issues:
                    potential_issues[emotion] = []
                potential_issues[emotion].append({
                    'keyword': keyword,
                    'count': len(matches),
                    'examples': matches['text'].head(3).tolist()
                })
    
    print("Potential Labeling Issues Found:")
    for emotion, issues in potential_issues.items():
        print(f"\n  {emotion.upper()}:")
        for issue in issues:
            print(f"    '{issue['keyword']}' found in {issue['count']} 'no emotion' texts")
            for example in issue['examples']:
                print(f"      Example: \"{example[:80]}...\"")
    
    # 4. Academic Recommendations
    print("\n🎓 4. ACADEMIC RECOMMENDATIONS")
    print("-" * 40)
    
    print("Based on this analysis, here are the academic recommendations:")
    print("\n📚 Dataset Limitations:")
    print("  1. Severe class imbalance (72:1 ratio)")
    print("  2. Potential mislabeling in 'no emotion' category")
    print("  3. Limited examples for minority emotions (fear, disgust)")
    
    print("\n🔧 Technical Solutions (No Data Modification):")
    print("  1. Use class_weight='balanced' in models")
    print("  2. Implement proper evaluation metrics (F1-score, precision, recall)")
    print("  3. Try ensemble methods and different algorithms")
    print("  4. Use cross-validation to ensure robust results")
    
    print("\n📝 Academic Reporting:")
    print("  1. Document all dataset limitations found")
    print("  2. Acknowledge the impact on model performance")
    print("  3. Discuss challenges in emotion detection research")
    print("  4. Propose future data collection improvements")
    
    # 5. Save Analysis Report
    print("\n💾 5. SAVING ANALYSIS REPORT")
    print("-" * 40)
    
    analysis_report = {
        'dataset_info': {
            'total_dialogues': len(original_data),
            'total_utterances': len(utterances),
            'emotion_classes': list(emotion_counts.index)
        },
        'class_distribution': {str(k): int(v) for k, v in emotion_counts.to_dict().items()},
        'imbalance_metrics': {
            'imbalance_ratio': float(imbalance_ratio),
            'most_frequent': str(emotion_counts.idxmax()),
            'least_frequent': str(emotion_counts.idxmin())
        },
        'text_statistics': {
            'avg_chars': float(df['text_length'].mean()),
            'avg_words': float(df['word_count'].mean()),
            'min_chars': int(df['text_length'].min()),
            'max_chars': int(df['text_length'].max())
        },
        'potential_issues': potential_issues,
        'academic_recommendations': [
            "Use class weights to handle imbalance",
            "Implement proper evaluation metrics",
            "Document all dataset limitations",
            "Acknowledge impact on performance"
        ]
    }
    
    report_path = Path("dataset_analysis_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Analysis report saved to {report_path}")
    
    print("\n🎯 NEXT STEPS (Academic Approach):")
    print("1. Use class weights in model training")
    print("2. Implement proper evaluation metrics")
    print("3. Document all findings in your report")
    print("4. Acknowledge dataset limitations")
    print("5. Discuss challenges and solutions")

if __name__ == "__main__":
    analyze_dataset_characteristics()
