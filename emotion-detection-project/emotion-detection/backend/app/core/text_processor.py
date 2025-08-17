import re
import logging
from typing import List
import pandas as pd

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Minimal text processor - basic cleaning only.
    No stop word removal, just simple text normalization.
    """
    
    def __init__(self):
        """Initialize minimal text processor."""
        logger.info("✅ Minimal TextProcessor initialized - basic cleaning only")
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to string if needed
        text = str(text)
        
        # Minimal cleaning only
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        
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
