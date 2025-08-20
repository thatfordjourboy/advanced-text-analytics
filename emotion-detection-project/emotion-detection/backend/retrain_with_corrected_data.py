#!/usr/bin/env python3
"""
Script to retrain emotion detection models using the corrected and balanced dataset.
This should significantly improve accuracy by using properly labeled data.
"""

import sys
import os
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "app"))

from core.model_trainer import MultiLabelEmotionTrainer
from core.data_loader import DataLoader
from core.embeddings import GloVeEmbeddings
from core.text_processor import TextProcessor
import json
import pandas as pd

def retrain_with_corrected_data():
    """Retrain models using the corrected dataset with class weights."""
    
    print("🚀 Starting model retraining with corrected data...")
    
    # Initialize components
    text_processor = TextProcessor()
    embeddings = GloVeEmbeddings(dimension=100)
    models_dir = Path(__file__).parent / "models"
    model_trainer = MultiLabelEmotionTrainer(models_dir)
    
    # Load embeddings
    print("📚 Loading GloVe embeddings...")
    embeddings.load_embeddings()
    print("✅ Embeddings loaded")
    
    # Load corrected dataset (NOT the balanced one)
    corrected_path = Path("data/dialogues_corrected.json")
    
    if not corrected_path.exists():
        print("❌ Corrected dataset not found. Run fix_emotion_labels.py first.")
        return
    
    print("📊 Loading corrected dataset...")
    with open(corrected_path, 'r', encoding='utf-8') as f:
        corrected_data = json.load(f)
    
    print(f"📈 Dataset loaded: {len(corrected_data)} samples")
    
    # Convert to DataFrame format expected by model trainer
    df = pd.DataFrame(corrected_data)
    
    # Show emotion distribution
    emotion_counts = df['emotion'].value_counts()
    print("\n📊 Emotion Distribution (Corrected):")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} samples")
    
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
            # Skip texts that can't be vectorized
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
    import numpy as np
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
        # Use class_weight='balanced' to handle imbalanced data
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
        from sklearn.metrics import accuracy_score, classification_report
        y_pred = lr_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"📊 Logistic Regression Test Accuracy: {accuracy:.3f}")
        
        # Save the model
        model_trainer.save_model(lr_model, 'logistic_regression_corrected.pkl')
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
        print(f"📊 Random Forest Test Accuracy: {accuracy:.3f}")
        
        # Save the model
        model_trainer.save_model(rf_model, 'random_forest_corrected.pkl')
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
    print("\nKey improvements made:")
    print("1. ✅ Fixed 3,752 mislabeled emotions in existing dataset")
    print("2. ✅ Used class_weight='balanced' to handle imbalanced data")
    print("3. ✅ Kept ALL training data (no undersampling)")
    print("4. ✅ Applied hyperparameter tuning for optimal performance")
    
    print("\nNext steps:")
    print("1. Test the new models with the problematic text")
    print("2. Update the main.py to use corrected models")
    print("3. Deploy and test accuracy improvements")

if __name__ == "__main__":
    retrain_with_corrected_data()
