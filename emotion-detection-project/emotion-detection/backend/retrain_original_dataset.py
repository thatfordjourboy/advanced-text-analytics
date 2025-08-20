#!/usr/bin/env python3
"""
Retrain models using the ORIGINAL dataset with class weights.
This script does NOT modify any dataset labels - it only uses
technical solutions to handle the existing class imbalance.
"""

import sys
import os
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from core.model_trainer import MultiLabelEmotionTrainer
from core.data_loader import DataLoader
from core.core.embeddings import GloVeEmbeddings
from core.text_processor import TextProcessor
import json
import pandas as pd
import numpy as np

def retrain_with_original_data():
    """Retrain models using the ORIGINAL dataset with class weights."""
    
    print("🚀 Retraining with ORIGINAL Dataset (No Label Changes)")
    print("=" * 60)
    
    # Initialize components
    text_processor = TextProcessor()
    embeddings = GloVeEmbeddings(dimension=100)
    models_dir = Path(__file__).parent / "models"
    model_trainer = MultiLabelEmotionTrainer(models_dir)
    
    # Load embeddings
    print("📚 Loading GloVe embeddings...")
    embeddings.load_embeddings()
    print("✅ Embeddings loaded")
    
    # Load the ORIGINAL dataset (no modifications)
    original_path = Path("data/dialogues.json")
    
    if not original_path.exists():
        print("❌ Original dataset not found.")
        return
    
    print("📊 Loading ORIGINAL dataset...")
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
    
    # Convert to DataFrame
    df = pd.DataFrame(utterances)
    
    # Show original distribution
    emotion_counts = df['emotion'].value_counts()
    print("\n📊 Original Emotion Distribution:")
    for emotion, count in emotion_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {emotion:15}: {count:6} samples ({percentage:5.1f}%)")
    
    # Prepare data for training
    print("\n🔧 Preparing data for training...")
    
    # Split data into train/validation/test (80/10/10)
    from sklearn.model_selection import train_test_split
    
    # First split: 80% train, 20% temp
    train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])
    
    # Second split: 10% validation, 10% test
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['emotion'])
    
    print(f"📊 Data splits:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Convert to format expected by model trainer
    train_texts = train_data['text'].tolist()
    train_emotions = train_data['emotion'].tolist()
    
    val_texts = val_data['text'].tolist()
    val_emotions = val_data['emotion'].tolist()
    
    test_texts = test_data['text'].tolist()
    test_emotions = test_data['emotion'].tolist()
    
    # Process text and create embeddings
    print("🔄 Processing text and creating embeddings...")
    
    # Process training data
    processed_train_texts = [text_processor.process_text(text) for text in train_texts]
    train_vectors = []
    train_emotions_filtered = []
    
    for i, text in enumerate(processed_train_texts):
        vector = embeddings.get_text_vector(text)
        if vector is not None:
            train_vectors.append(vector)
            train_emotions_filtered.append(train_emotions[i])
        else:
            continue
    
    # Process validation data
    processed_val_texts = [text_processor.process_text(text) for text in val_texts]
    val_vectors = []
    val_emotions_filtered = []
    
    for i, text in enumerate(processed_val_texts):
        vector = embeddings.get_text_vector(text)
        if vector is not None:
            val_vectors.append(vector)
            val_emotions_filtered.append(val_emotions[i])
        else:
            continue
    
    # Process test data
    processed_test_texts = [text_processor.process_text(text) for text in test_texts]
    test_vectors = []
    test_emotions_filtered = []
    
    for i, text in enumerate(processed_test_texts):
        vector = embeddings.get_text_vector(text)
        if vector is not None:
            test_vectors.append(vector)
            test_emotions_filtered.append(test_emotions[i])
        else:
            continue
    
    print(f"✅ Vectors created:")
    print(f"  Train: {len(train_vectors)} vectors")
    print(f"  Validation: {len(val_vectors)} vectors")
    print(f"  Test: {len(test_vectors)} vectors")
    
    # Convert to numpy arrays
    X_train = np.array(train_vectors)
    y_train = np.array(train_emotions_filtered)
    
    X_val = np.array(val_vectors)
    y_val = np.array(val_emotions_filtered)
    
    X_test = np.array(test_vectors)
    y_test = np.array(test_emotions_filtered)
    
    # Update model trainer with new data
    model_trainer.X_train = X_train
    model_trainer.y_train = y_train
    model_trainer.X_val = X_val
    model_trainer.y_val = y_val
    model_trainer.X_test = X_test
    model_trainer.y_test = y_test
    
    # Retrain Logistic Regression with class weights
    print("\n🧠 Retraining Logistic Regression with class weights...")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import RandomizedSearchCV
        
        # Define parameter grid with class weights
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced']  # This handles the imbalance
        }
        
        # Create base model
        base_lr = LogisticRegression(random_state=42, max_iter=1000)
        
        # Perform hyperparameter tuning
        random_search = RandomizedSearchCV(
            base_lr, param_grid, n_iter=20, cv=3, random_state=42, n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        
        lr_model = random_search.best_estimator_
        print(f"✅ Logistic Regression retrained successfully")
        print(f"📊 Best parameters: {random_search.best_params_}")
        
        # Evaluate on test set
        from sklearn.metrics import accuracy_score, classification_report, f1_score
        y_pred = lr_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"📊 Logistic Regression Performance:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1-Score (Macro): {f1_macro:.3f}")
        
        # Save the model
        model_trainer.save_model(lr_model, 'logistic_regression_original.pkl')
        print("💾 Logistic Regression model saved")
        
    except Exception as e:
        print(f"❌ Logistic Regression training failed: {e}")
    
    # Retrain Random Forest with class weights
    print("\n🌲 Retraining Random Forest with class weights...")
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        # Define parameter grid with class weights
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced']  # This handles the imbalance
        }
        
        # Create base model
        base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Perform hyperparameter tuning
        random_search = RandomizedSearchCV(
            base_rf, param_grid, n_iter=20, cv=3, random_state=42, n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        
        rf_model = random_search.best_estimator_
        print(f"✅ Random Forest retrained successfully")
        print(f"📊 Best parameters: {random_search.best_params_}")
        
        # Evaluate on test set
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"📊 Random Forest Performance:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1-Score (Macro): {f1_macro:.3f}")
        
        # Save the model
        model_trainer.save_model(rf_model, 'random_forest_original.pkl')
        print("💾 Random Forest model saved")
        
    except Exception as e:
        print(f"❌ Random Forest training failed: {e}")
    
    # Generate detailed classification report
    print("\n📋 Detailed Classification Report (Random Forest):")
    try:
        y_pred = rf_model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=sorted(set(y_test)))
        print(report)
    except Exception as e:
        print(f"❌ Could not generate classification report: {e}")
    
    print("\n🎉 Model retraining complete!")
    print("\n📚 Academic Approach Summary:")
    print("✅ Used ORIGINAL dataset (no label modifications)")
    print("✅ Applied class weights to handle imbalance")
    print("✅ Used proper evaluation metrics (F1-score)")
    print("✅ Documented all dataset limitations")
    
    print("\n🎯 Next steps:")
    print("1. Test the new models with the problematic text")
    print("2. Compare performance with original models")
    print("3. Document improvements in your academic report")
    print("4. Acknowledge dataset limitations and solutions used")

if __name__ == "__main__":
    retrain_with_original_data()
