#!/usr/bin/env python3
"""
Script to fix mislabeled emotion data in the ConvLab dataset.
This addresses the issue where emotional content is incorrectly labeled as 'no emotion'.
"""

import pandas as pd
import re
from pathlib import Path

def fix_emotion_labels():
    """Fix obvious mislabeled emotions in the dataset."""
    
    # Load the dataset
    data_path = Path("data/dialogues.json")
    if not data_path.exists():
        print("❌ Dialogues file not found. Run this from the backend directory.")
        return
    
    print("🔍 Loading dataset...")
    with open(data_path, 'r', encoding='utf-8') as f:
        import json
        data = json.load(f)
    
    # Extract all utterances
    utterances = []
    for dialogue in data:
        turns = dialogue.get('turns', [])
        for turn in turns:
            text = turn.get('utterance', '').strip()  # Fixed: use 'utterance' not 'text'
            emotion = turn.get('emotion', '').strip()
            if text and emotion:
                utterances.append({
                    'text': text,  # Keep as 'text' for consistency
                    'emotion': emotion,
                    'dialogue_id': dialogue.get('dialogue_id', ''),
                    'turn_idx': turns.index(turn)
                })
    
    print(f"📊 Found {len(utterances)} utterances")
    
    # Define emotion keywords and their target emotions
    emotion_keywords = {
        'sadness': [
            'sad', 'unhappy', 'depressed', 'heartbroken', 'devastated', 'tragic',
            'miserable', 'upset', 'disappointed', 'grief', 'sorrow', 'melancholy',
            'blue', 'down', 'low', 'unfortunate', 'regret', 'sorry', 'woe'
        ],
        'fear': [
            'scared', 'frightened', 'terrified', 'afraid', 'worried', 'anxious',
            'nervous', 'panicked', 'horrified', 'dread', 'alarm', 'panic',
            'threatened', 'intimidated', 'uneasy', 'apprehensive'
        ],
        'anger': [
            'angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated',
            'outraged', 'enraged', 'livid', 'fuming', 'raging', 'hostile',
            'aggressive', 'bitter', 'resentful', 'indignant'
        ],
        'disgust': [
            'disgusted', 'revolted', 'repulsed', 'sickened', 'appalled',
            'horrified', 'nauseated', 'contempt', 'aversion', 'loathing'
        ],
        'surprise': [
            'surprised', 'shocked', 'amazed', 'astonished', 'stunned',
            'bewildered', 'startled', 'dumbfounded', 'flabbergasted'
        ],
        'happiness': [
            'happy', 'joyful', 'excited', 'thrilled', 'delighted', 'ecstatic',
            'elated', 'cheerful', 'pleased', 'content', 'satisfied', 'grateful'
        ]
    }
    
    # Count fixes needed
    fixes_needed = 0
    fixed_utterances = []
    
    print("🔧 Analyzing emotion labels...")
    
    for utterance in utterances:
        text = utterance['text'].lower()
        current_emotion = utterance['emotion']
        suggested_emotion = None
        
        # Skip if already correctly labeled
        if current_emotion != 'no emotion':
            fixed_utterances.append(utterance)
            continue
        
        # Check for emotional keywords
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text for keyword in keywords):
                suggested_emotion = emotion
                break
        
        # Apply fixes for obvious cases
        if suggested_emotion and suggested_emotion != current_emotion:
            # Additional validation: check if text is actually emotional
            emotional_indicators = [
                '!' in text or '?' in text,  # Punctuation indicating emotion
                len(text.split()) > 3,  # Longer texts more likely to be emotional
                any(word in text for word in ['feel', 'felt', 'am', 'is', 'are', 'was', 'were'])  # Emotional verbs
            ]
            
            if sum(emotional_indicators) >= 2:  # At least 2 indicators
                utterance['emotion'] = suggested_emotion
                utterance['original_emotion'] = current_emotion
                fixes_needed += 1
                print(f"🔄 Fixed: '{text[:50]}...' -> {suggested_emotion}")
        
        fixed_utterances.append(utterance)
    
    print(f"\n✅ Fixed {fixes_needed} emotion labels")
    
    # Save corrected dataset
    output_path = Path("data/dialogues_corrected.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_utterances, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved corrected dataset to {output_path}")
    
    # Show new distribution
    emotion_counts = pd.DataFrame(fixed_utterances)['emotion'].value_counts()
    print("\n📊 New Emotion Distribution:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} samples")
    
    return fixed_utterances

def create_balanced_dataset():
    """Create a more balanced dataset for training."""
    
    print("\n⚖️ Creating balanced dataset...")
    
    # Load corrected data
    corrected_path = Path("data/dialogues_corrected.json")
    if not corrected_path.exists():
        print("❌ Corrected dataset not found. Run fix_emotion_labels() first.")
        return
    
    with open(corrected_path, 'r', encoding='utf-8') as f:
        import json
        corrected_data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(corrected_data)
    
    # Get emotion counts
    emotion_counts = df['emotion'].value_counts()
    min_samples = emotion_counts.min()
    max_samples = emotion_counts.max()
    
    print(f"📊 Current distribution: {min_samples} to {max_samples} samples per emotion")
    
    # Create balanced dataset (undersample majority classes)
    balanced_data = []
    target_samples = min(5000, max_samples // 2)  # Target 5000 samples per emotion
    
    for emotion in emotion_counts.index:
        emotion_data = df[df['emotion'] == emotion]
        
        if len(emotion_data) > target_samples:
            # Undersample majority classes
            sampled_data = emotion_data.sample(n=target_samples, random_state=42)
        else:
            # Keep minority classes as-is
            sampled_data = emotion_data
        
        balanced_data.extend(sampled_data.to_dict('records'))
        print(f"  {emotion}: {len(sampled_data)} samples")
    
    # Save balanced dataset
    balanced_path = Path("data/dialogues_balanced.json")
    with open(balanced_path, 'w', encoding='utf-8') as f:
        json.dump(balanced_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved balanced dataset to {balanced_path}")
    print(f"📊 Total samples: {len(balanced_data)}")
    
    return balanced_data

if __name__ == "__main__":
    print("🚀 Starting emotion label correction...")
    
    # Step 1: Fix mislabeled emotions
    fixed_data = fix_emotion_labels()
    
    # Step 2: Create balanced dataset
    balanced_data = create_balanced_dataset()
    
    print("\n🎉 Emotion label correction complete!")
    print("\nNext steps:")
    print("1. Review the corrected dataset")
    print("2. Retrain models with balanced data")
    print("3. Test accuracy improvements")
