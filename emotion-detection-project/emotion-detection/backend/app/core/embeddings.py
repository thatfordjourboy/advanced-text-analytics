import numpy as np
from typing import List
from pathlib import Path
import time

class GloVeEmbeddings:
    """Simple GloVe embeddings wrapper using 2024 Stanford vectors."""
    
    def __init__(self, dimension: int = 100):
        self.dimension = dimension
        self.embeddings = {}
        self.loaded = False
        
        # File paths - vectors already in data directory
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        
        # call GloVe vectors
        if dimension == 100:
            self.txt_path = self.data_dir / "wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt"
        elif dimension == 300:
            self.txt_path = self.data_dir / "glove.2024.wikigiga.300d.txt"
        else:
            # Fallback to 6B if other dimensions needed requires download
            self.txt_path = self.data_dir / f"glove.6B.{dimension}d.txt"
    
    def load_embeddings(self):
        """Load pre-downloaded 2024 GloVe embeddings."""
        try:
            if not self.txt_path.exists():
                print(f"❌ GloVe file not found: {self.txt_path.name}")
                print("📥 Please download the 2024 vectors from Stanford:")
                print("🔗 https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.100d.zip")
                if self.dimension == 100:
                    print("📁 Download: glove.2024.wikigiga.100d.zip (555 MB)")
                elif self.dimension == 300:
                    print("📁 Download: glove.2024.wikigiga.300d.zip (1.6 GB)")
                print("📂 Extract to: backend/data/ directory")
                print("💡 Note: The startup script should handle this automatically on Render")
                return False
            
            # Load embeddings
            self._load_from_file()
            return True
            
        except Exception as e:
            print(f"Failed to load embeddings: {e}")
            return False
    
    def _load_from_file(self):
        """Load embeddings from text file."""
        print(f"Loading {self.dimension}d vectors from {self.txt_path.name}...")
        print("✅ Using 2024 GloVe vectors")
        
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    values = line.strip().split()
                    if len(values) < self.dimension + 1:
                        continue  # Skip malformed lines
                    
                    word = values[0]
                    #  2024 format has special characters
                    if word in [',', '.', '!', '?', ';', ':', '"', "'", '-', '(', ')']:
                        continue  # Skip punctuation-only words
                    
                    # Convert remaining values to float, handling any formatting issues
                    vector_values = []
                    for val in values[1:]:
                        try:
                            vector_values.append(float(val))
                        except ValueError:
                            continue  # Skip invalid values
                    
                    # Only add if we have the right number of dimensions
                    if len(vector_values) == self.dimension:
                        self.embeddings[word] = np.array(vector_values, dtype='float32')
                    
                except Exception as e:
                    continue  # Skip problematic lines
        
        self.loaded = True
        print(f"✅ Loaded {len(self.embeddings)} word vectors")
    
    def get_text_vector(self, text: str) -> np.ndarray:
        """Get vector for text."""
        if not self.loaded:
            raise ValueError("Embeddings not loaded")
        
        words = text.lower().split()
        vectors = []
        
        for word in words:
            if word in self.embeddings:
                vectors.append(self.embeddings[word])
        
        if not vectors:
            return np.zeros(self.dimension)
        
        return np.mean(vectors, axis=0)
    
    def get_batch_vectors(self, texts: List[str]) -> np.ndarray:
        """Get vectors for multiple texts using optimized batch processing."""
        if not self.loaded:
            raise ValueError("Embeddings not loaded")
        
        if not texts:
            return np.array([])
        
        # Pre-allocate result array for better memory efficiency
        result = np.zeros((len(texts), self.dimension), dtype='float32')
        
        # Process all texts in batch
        for i, text in enumerate(texts):
            if not text or not text.strip():
                result[i] = np.zeros(self.dimension)
                continue
                
            words = text.lower().split()
            if not words:
                result[i] = np.zeros(self.dimension)
                continue
            
            # Collect vectors for all words in this text
            word_vectors = []
            for word in words:
                if word in self.embeddings:
                    word_vectors.append(self.embeddings[word])
            
            # Compute mean vector for this text
            if word_vectors:
                result[i] = np.mean(word_vectors, axis=0)
            else:
                result[i] = np.zeros(self.dimension)
        
        return result
    
    def get_batch_vectors_optimized(self, texts: List[str]) -> np.ndarray:
        """Ultra-optimized batch processing using vectorized operations."""
        if not self.loaded:
            raise ValueError("Embeddings not loaded")
        
        if not texts:
            return np.array([])
        
        print(f"🚀 Processing {len(texts)} texts in optimized batches...")
        
        # Pre-allocate result array
        result = np.zeros((len(texts), self.dimension), dtype='float32')
        
        # Process in chunks for memory efficiency
        chunk_size = 1000  # Process 1000 texts at a time
        total_chunks = (len(texts) + chunk_size - 1) // chunk_size
        
        for chunk_idx, chunk_start in enumerate(range(0, len(texts), chunk_size)):
            chunk_end = min(chunk_start + chunk_size, len(texts))
            chunk_texts = texts[chunk_start:chunk_end]
            
            print(f"📦 Processing chunk {chunk_idx + 1}/{total_chunks} ({chunk_start + 1}-{chunk_end})")
            
            # Vectorized processing for this chunk
            for i, text in enumerate(chunk_texts):
                if not text or not text.strip():
                    continue
                
                words = text.lower().split()
                if not words:
                    continue
                
                # Use numpy operations for vector aggregation
                word_vectors = []
                for word in words:
                    if word in self.embeddings:
                        word_vectors.append(self.embeddings[word])
                
                if word_vectors:
                    result[chunk_start + i] = np.mean(word_vectors, axis=0)
        
        print(f"✅ Completed batch processing of {len(texts)} texts")
        return result
    
    def get_embedding_info(self) -> dict:
        """Get information about the loaded embeddings."""
        if not self.loaded:
            return {
                'loaded': False,
                'dimension': self.dimension,
                'vocabulary_size': 0,
                'file_size_mb': 0
            }
        
        # Calculate file size
        file_size_mb = 0
        if self.txt_path.exists():
            file_size_mb = round(self.txt_path.stat().st_size / (1024 * 1024), 2)
        
        return {
            'loaded': True,
            'dimension': self.dimension,
            'vocabulary_size': len(self.embeddings),
            'file_size_mb': file_size_mb,
            'file_path': str(self.txt_path)
        }

    def get_processing_stats(self) -> dict:
        """Get statistics about the embeddings processing."""
        return {
            'vocabulary_size': len(self.embeddings),
            'dimension': self.dimension,
            'memory_usage_mb': round(len(self.embeddings) * self.dimension * 4 / (1024 * 1024), 2),
            'coverage_estimate': 'High' if len(self.embeddings) > 100000 else 'Medium' if len(self.embeddings) > 50000 else 'Low'
        }

    def health_check(self) -> dict:
        """Check the health and performance of the embeddings system."""
        try:
            if not self.loaded:
                return {
                    'status': 'not_loaded',
                    'message': 'Embeddings not loaded',
                    'error': 'Call load_embeddings() first'
                }
            
            # Test processing speed with a small sample
            test_texts = ['hello world', 'test text', 'sample sentence']
            start_time = time.time()
            test_vectors = self.get_batch_vectors(test_texts)
            processing_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'loaded': True,
                'vocabulary_size': len(self.embeddings),
                'dimension': self.dimension,
                'test_processing_time_ms': round(processing_time * 1000, 2),
                'estimated_batch_speed': f"{round(1000 / processing_time)} texts/second",
                'memory_usage_mb': round(len(self.embeddings) * self.dimension * 4 / (1024 * 1024), 2)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Health check failed: {str(e)}',
                'error': str(e)
            }
