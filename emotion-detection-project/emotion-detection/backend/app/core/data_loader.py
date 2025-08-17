import logging
import pandas as pd
import numpy as np
import json

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader for emotion detection datasets."""
    
    def __init__(self):
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.loaded = False
        
        # Initialize emotion mapping - will be populated from actual dataset
        self.emotion_mapping = {}
        
        # Load the dataset first to get actual emotions - no fallbacks
        self._load_and_setup_emotions()
        
    def load_dataset(self, force_reload=False):
        """Load the complete Daily Dialog dataset with all three splits."""
        if self.loaded and not force_reload:
            logger.info("Dataset already loaded")
            return True
        
        try:
            logger.info("Loading complete Daily Dialog dataset with all splits...")
            
            # Load the real ConvLab Daily Dialog Dataset
            all_data = self._load_complete_dataset()
            
            if not all_data:
                logger.error("❌ Failed to load ConvLab dataset")
                logger.error("Please ensure data.zip has been extracted to the backend directory")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            logger.info(f"Successfully loaded {len(df)} total utterances")
            
            # Create proper splits
            self._create_proper_splits(df)
            
            self.loaded = True
            logger.info("Dataset loading completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False


    # _load_dataset_via_api method removed - we only use the real ConvLab dataset
    
    def _load_complete_dataset(self):
        """Load the complete dataset with all three splits from the real ConvLab data."""
        try:
            import json
            import os
            
            logger.info("Loading complete dataset from ConvLab data.zip...")
            
            # Check if we have the extracted data
            dialogues_path = "data/dialogues.json"
            if not os.path.exists(dialogues_path):
                logger.error(f"Dialogues file not found at {dialogues_path}")
                logger.info("Please ensure data.zip has been extracted to the backend directory")
                return []
            
            # Load the dialogues
            with open(dialogues_path, 'r', encoding='utf-8') as f:
                dialogues = json.load(f)
            
            logger.info(f"Loaded {len(dialogues)} dialogues from ConvLab dataset")
            
            all_data = []
            
            # Process each dialogue
            for dialogue in dialogues:
                split_name = dialogue.get('data_split', 'unknown')
                turns = dialogue.get('turns', [])
                
                for turn in turns:
                    utterance = turn.get('utterance', '')
                    emotion = turn.get('emotion', 'no emotion')
                    
                    if utterance and emotion:
                        all_data.append({
                            'text': utterance,
                            'emotion': emotion,
                            'emotion_id': self.emotion_mapping.get(emotion, 0),
                            'split': split_name,  # Track which split this came from
                            'dialogue_id': dialogue.get('dialogue_id', ''),
                            'speaker': turn.get('speaker', ''),
                            'utt_idx': turn.get('utt_idx', 0)
                        })
            
            # Verify we have all three splits
            splits_found = set(item['split'] for item in all_data)
            logger.info(f"✅ ConvLab dataset loaded successfully!")
            logger.info(f"✅ Splits found: {splits_found}")
            logger.info(f"✅ Total utterances: {len(all_data)}")
            
            # Count utterances per split
            for split in ['train', 'validation', 'test']:
                count = len([item for item in all_data if item['split'] == split])
                logger.info(f"✅ {split.capitalize()}: {count} utterances")
            
            if len(splits_found) >= 3:  # All three splits present
                logger.info("✅ SUCCESS: Real ConvLab Daily Dialog Dataset loaded with proper splits!")
                return all_data
            else:
                logger.error(f"❌ Missing splits. Expected 3, found: {splits_found}")
                return []
            
        except Exception as e:
            logger.error(f"Failed to load ConvLab dataset: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _load_and_setup_emotions(self):
        """Load emotions from the actual ConvLab dataset and setup mapping."""
        try:
            import json
            import os
            
            # Check if we have the extracted data
            dialogues_path = "data/dialogues.json"
            if not os.path.exists(dialogues_path):
                logger.error("Dialogues file not found")
                raise ValueError("Cannot proceed without ConvLab dataset")
            
            # Load a sample to get emotions
            with open(dialogues_path, 'r', encoding='utf-8') as f:
                # Read just the first few lines to get emotion samples
                sample_data = []
                for i, line in enumerate(f):
                    if i >= 1000:  # Read first 1000 lines to get good emotion coverage
                        break
                    if line.strip() and line.strip() != '[' and line.strip() != ']':
                        sample_data.append(line.strip().rstrip(','))
                
                # Parse the sample data
                emotions_found = set()
                for line in sample_data:
                    if '"emotion":' in line:
                        emotion_match = line.split('"emotion": "')[1].split('"')[0]
                        emotions_found.add(emotion_match)
            
            # Create emotion mapping from actual dataset
            emotions_list = sorted(list(emotions_found))
            self.emotion_mapping = {emotion: idx for idx, emotion in enumerate(emotions_list)}
            
            logger.info(f"✅ Emotions loaded from ConvLab dataset: {emotions_list}")
            logger.info(f"✅ Emotion mapping: {self.emotion_mapping}")
            
        except Exception as e:
            logger.error(f"Failed to load emotions from dataset: {e}")
            raise ValueError("Cannot proceed without loading emotions from ConvLab dataset")

    # _setup_default_emotion_mapping method removed - no fallbacks allowed

    def _create_proper_splits(self, df):
        """Create proper train/validation/test splits from the complete dataset."""
        try:
            logger.info("Creating proper data splits...")
            
            # Check if we have split information AND if it contains multiple splits
            if 'split' in df.columns:
                unique_splits = df['split'].unique()
                if len(unique_splits) >= 3 and 'validation' in unique_splits and 'test' in unique_splits:
                    logger.info("✅ Using pre-defined splits from ConvLab dataset...")
                    self._use_predefined_splits(df)
                else:
                    logger.error(f"❌ Split column found but insufficient splits: {unique_splits}")
                    logger.error("Expected: train, validation, test")
                    raise ValueError(f"Insufficient splits found: {unique_splits}")
            else:
                logger.error("❌ No split column found in ConvLab dataset")
                raise ValueError("Split information missing from dataset")
            
        except Exception as e:
            logger.error(f"Failed to create splits: {e}")
            raise

    def _use_predefined_splits(self, df):
        """Use the pre-defined splits from the dataset."""
        try:
            # Filter by split
            train_df = df[df['split'] == 'train'].copy()
            val_df = df[df['split'] == 'validation'].copy()
            test_df = df[df['split'] == 'test'].copy()
            
            # Remove split column as it's no longer needed
            train_df = train_df.drop('split', axis=1)
            val_df = val_df.drop('split', axis=1)
            test_df = test_df.drop('split', axis=1)
            
            self.train_data = train_df
            self.val_data = val_df
            self.test_data = test_df
            
            logger.info(f"Pre-defined splits loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
            # Log emotion distribution for each split
            logger.info(f"✅ Train emotion distribution: {train_df['emotion_id'].value_counts().to_dict()}")
            logger.info(f"✅ Validation emotion distribution: {val_df['emotion_id'].value_counts().to_dict()}")
            logger.info(f"✅ Test emotion distribution: {test_df['emotion_id'].value_counts().to_dict()}")
            
        except Exception as e:
            logger.error(f"Failed to use predefined splits: {e}")
            raise

    # _create_splits_from_combined method removed - we only use predefined splits from ConvLab dataset

        # Helper methods removed - not needed with predefined splits

    def get_split_info(self):
        """Get information about all three splits."""
        if not self.loaded:
            return {"error": "Dataset not loaded"}
        
        return {
            "train": {
                "samples": len(self.train_data),
                "emotion_distribution": self.train_data['emotion_id'].value_counts().to_dict()
            },
            "validation": {
                "samples": len(self.val_data),
                "emotion_distribution": self.val_data['emotion_id'].value_counts().to_dict()
            },
            "test": {
                "samples": len(self.test_data),
                "emotion_distribution": self.test_data['emotion_id'].value_counts().to_dict()
            }
        }
    
    def get_texts_and_labels(self, split='train'):
        """Get texts and labels for a specific split."""
        if not self.loaded:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        split_map = {
            'train': self.train_data,
            'validation': self.val_data,
            'test': self.test_data
        }
        
        if split not in split_map:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'validation', or 'test'.")
        
        data = split_map[split]
        texts = data['text'].tolist()
        labels = data['emotion_id'].tolist()
        
        return texts, labels
    
    def get_emotion_distribution(self, split='train'):
        """Get emotion distribution for a specific split."""
        texts, labels = self.get_texts_and_labels(split)
        
        distribution = {}
        for label in labels:
            emotion = [k for k, v in self.emotion_mapping.items() if v == label][0]
            distribution[emotion] = distribution.get(emotion, 0) + 1
        
        return distribution
    
    def get_dataset_info(self):
        """Get dataset information and statistics."""
        if not self.loaded:
            return {}
        
        return {
            'total_samples': len(self.train_data) + len(self.val_data) + len(self.test_data),
            'train_samples': len(self.train_data),
            'validation_samples': len(self.val_data),
            'test_samples': len(self.test_data),
            'emotion_categories': list(self.emotion_mapping.keys()),
            'emotion_mapping': self.emotion_mapping,
            'train_emotion_distribution': self._convert_emotion_ids_to_names(self.train_data['emotion_id'].value_counts().to_dict()) if len(self.train_data) > 0 else {},
            'loaded': self.loaded
        }
    
    def _convert_emotion_ids_to_names(self, emotion_id_counts):
        """Convert emotion ID counts to emotion name counts."""
        emotion_name_counts = {}
        for emotion_id, count in emotion_id_counts.items():
            # Find emotion name for this ID
            emotion_name = None
            for name, id_val in self.emotion_mapping.items():
                if id_val == emotion_id:
                    emotion_name = name
                    break
            
            if emotion_name:
                emotion_name_counts[emotion_name] = count
            else:
                # Fallback: use ID as string if no name found
                emotion_name_counts[str(emotion_id)] = count
        
        return emotion_name_counts
    
    @property
    def emotion_categories(self):
        """Get list of emotion category names."""
        return list(self.emotion_mapping.keys())
