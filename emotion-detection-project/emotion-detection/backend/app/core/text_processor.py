import re
import logging
from typing import List
import pandas as pd

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Enhanced text processor for emotion detection.
    Preserves emotional context while cleaning text.
    """
    
    def __init__(self):
        """Initialize enhanced text processor."""
        logger.info("✅ Enhanced TextProcessor initialized - preserving emotional context")
        
        # Emotional intensity markers to preserve
        self.emotional_markers = {
            'very': 2.0, 'really': 2.0, 'extremely': 3.0, 'absolutely': 3.0,
            'hate': 3.0, 'love': 3.0, 'adore': 3.0, 'despise': 3.0,
            'terrible': 2.5, 'amazing': 2.5, 'awful': 2.5, 'wonderful': 2.5,
            'horrible': 2.5, 'fantastic': 2.5, 'dreadful': 2.5, 'excellent': 2.5
        }
        
        # Emotional punctuation patterns
        self.emotional_punctuation = {
            '!': 1.5, '!!': 2.0, '!!!': 2.5,
            '?': 1.2, '??': 1.5, '???': 2.0
        }
    
    def clean_text(self, text: str) -> str:
        """
        Enhanced text cleaning that preserves emotional context.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text with emotional context preserved
        """
        if not text:
            return ""
        
        # Convert to string if needed
        text = str(text)
        
        # Preserve emotional punctuation before cleaning
        emotional_score = 0
        for pattern, score in self.emotional_punctuation.items():
            if pattern in text:
                emotional_score += score * text.count(pattern)
        
        # Enhanced cleaning that preserves emotional words
        text = text.lower().strip()
        
        # Preserve emotional intensity words
        for word in self.emotional_markers:
            if word in text:
                # Add intensity marker to preserve emotional context
                text = text.replace(word, f"{word}_INTENSE")
        
        # Remove extra whitespace but preserve structure
        text = re.sub(r'\s+', ' ', text)
        
        # Add emotional context marker if high emotional content detected
        if emotional_score > 2.0:
            text = f"EMOTIONAL_CONTEXT_{text}"
        
        return text
    
    def process_text(self, text: str) -> str:
        """
        Minimal text processing.
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        return self.clean_text(text)
    
    def process_dataset(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Process text column in dataset.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            
        Returns:
            DataFrame with processed text
        """
        if text_column not in df.columns:
            logger.warning(f"Text column '{text_column}' not found in DataFrame")
            return df
        
        logger.info(f"Processing {len(df)} text samples...")
        
        # Process text column
        df[f'{text_column}_processed'] = df[text_column].apply(self.process_text)
        
        logger.info("Text processing completed")
        return df
