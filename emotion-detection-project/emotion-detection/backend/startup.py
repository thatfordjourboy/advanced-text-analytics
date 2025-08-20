#!/usr/bin/env python3
"""
🚀 Startup Script for Emotion Detection Backend
Downloads required data files during first startup and caches them on persistent disk.
"""

import os
import sys
import time
import logging
import requests
import zipfile
from pathlib import Path
from typing import Optional, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataFileManager:
    """Manages data file downloads and caching on Render's persistent disk."""
    
    def __init__(self):
        # Try multiple possible data directory paths
        possible_paths = [
            Path("/app/data"),  # Render persistent disk path
            Path("/opt/render/project/src/emotion-detection-project/emotion-detection/backend/data"),  # Render source path
            Path("data"),  # Local development path
            Path("/tmp/data"),  # Temporary path as fallback
        ]
        
        self.data_dir = None
        for path in possible_paths:
            try:
                path.mkdir(parents=True, exist_ok=True)
                self.data_dir = path
                logger.info(f"Using data directory: {path}")
                break
            except Exception as e:
                logger.warning(f"Failed to create/use {path}: {e}")
                continue
        
        if self.data_dir is None:
            raise Exception("Could not create data directory in any location")
        
        # Required files with their sources
        self.required_files = {
            "glove.2024.wikigiga.100d.zip": {
                "size_mb": 555,
                "sources": [
                    "https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.100d.zip"
                ],
                "description": "GloVe embeddings (555MB) - Stanford NLP 2024"
            },
            "wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_050_combined.txt": {
                "size_mb": 1600,
                "sources": [],  # Will be extracted from zip
                "description": "Extracted GloVe vectors (1.6GB)"
            },
            "dialogues.json": {
                "size_mb": 45,
                "sources": [],  # Will be created if missing
                "description": "Emotion dataset (45MB)"
            },
            "ontology.json": {
                "size_mb": 0.001,
                "sources": [],  # Will be created if missing
                "description": "Emotion labels (1.4KB)"
            }
        }
    
    def check_files_exist(self) -> Dict[str, bool]:
        """Check which required files already exist."""
        logger.info(f"Checking files in data directory: {self.data_dir}")
        existing_files = {}
        for filename in self.required_files:
            file_path = self.data_dir / filename
            exists = file_path.exists()
            existing_files[filename] = exists
            
            if exists:
                try:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    logger.info(f"{filename} exists ({size_mb:.1f}MB) at {file_path}")
                except Exception as e:
                    logger.warning(f"{filename} exists but size check failed: {e}")
            else:
                logger.info(f"{filename} missing - expected at {file_path}")
        
        logger.info(f"File existence summary: {existing_files}")
        return existing_files
    
    def download_file(self, url: str, filename: str) -> bool:
        """Download a file with progress tracking."""
        try:
            file_path = self.data_dir / filename
            logger.info(f"Downloading {filename} from {url}")
            
            # Stream download with progress
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress update every 10MB
                        if downloaded % (10 * 1024 * 1024) == 0:
                            mb_downloaded = downloaded / (1024 * 1024)
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                logger.info(f"{filename}: {mb_downloaded:.1f}MB ({progress:.1f}%)")
                            else:
                                logger.info(f"{filename}: {mb_downloaded:.1f}MB downloaded")
            
            logger.info(f"{filename} downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return False
    
    def extract_glove_vectors(self) -> bool:
        """Extract GloVe vectors from zip file."""
        try:
            zip_path = self.data_dir / "glove.2024.wikigiga.100d.zip"
            logger.info(f"Starting GloVe extraction process...")
            logger.info(f"Zip file path: {zip_path}")
            logger.info(f"Zip file exists: {zip_path.exists()}")
            
            if not zip_path.exists():
                logger.error("GloVe zip file not found")
                return False
            
            # Check zip file size before extraction
            zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
            logger.info(f"Zip file size: {zip_size_mb:.1f}MB")
            
            logger.info(f"Extracting GloVe vectors from: {zip_path}")
            logger.info(f"Extracting to directory: {self.data_dir}")
            logger.info(f"Target directory exists: {self.data_dir.exists()}")
            logger.info(f"Target directory contents before extraction:")
            for item in self.data_dir.iterdir():
                logger.info(f"  - {item.name}")
            
            # List contents of zip file first
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                logger.info(f"Zip file contains {len(file_list)} files:")
                for file in file_list:
                    logger.info(f"  - {file}")
                
                logger.info("Starting extraction...")
                zip_ref.extractall(self.data_dir)
                logger.info("Extraction completed")
            
            # List directory contents after extraction
            logger.info(f"Directory contents after extraction:")
            for item in self.data_dir.iterdir():
                if item.is_file():
                    size_mb = item.stat().st_size / (1024 * 1024)
                    logger.info(f"  - {item.name} ({size_mb:.1f}MB)")
                else:
                    logger.info(f"  - {item.name} (directory)")
            
            # Verify extraction
            extracted_file = self.data_dir / "wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_050_combined.txt"
            logger.info(f"Checking for extracted file: {extracted_file}")
            logger.info(f"Extracted file exists: {extracted_file.exists()}")
            
            if extracted_file.exists():
                size_mb = extracted_file.stat().st_size / (1024 * 1024)
                logger.info(f"GloVe vectors extracted successfully ({size_mb:.1f}MB)")
                logger.info(f"Extraction verification: PASSED")
                return True
            else:
                logger.error("Extracted file not found after extraction")
                logger.error(f"Expected file: {extracted_file}")
                logger.error(f"Available files in {self.data_dir}:")
                for item in self.data_dir.iterdir():
                    logger.error(f"  - {item.name}")
                logger.error("Extraction verification: FAILED")
                return False
                
        except Exception as e:
            logger.error(f"Failed to extract GloVe vectors: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def create_sample_files(self) -> bool:
        """Create sample data files if they don't exist."""
        try:
            # Create sample dialogues.json
            dialogues_path = self.data_dir / "dialogues.json"
            if not dialogues_path.exists():
                logger.info("Creating sample dialogues.json...")
                sample_dialogues = [
                    {
                        "dialogue_id": "sample_1",
                        "data_split": "train",
                        "turns": [
                            {
                                "utterance": "I'm feeling really happy today!",
                                "emotion": "happiness",
                                "speaker": "user",
                                "utt_idx": 0
                            }
                        ]
                    }
                ]
                
                import json
                with open(dialogues_path, 'w') as f:
                    json.dump(sample_dialogues, f, indent=2)
                logger.info("Created sample dialogues.json")
            
            # Create ontology.json
            ontology_path = self.data_dir / "ontology.json"
            if not ontology_path.exists():
                logger.info("Creating ontology.json...")
                ontology = {
                    "emotions": [
                        "anger", "disgust", "fear", "happiness", 
                        "no emotion", "sadness", "surprise"
                    ]
                }
                
                with open(ontology_path, 'w') as f:
                    json.dump(ontology, f, indent=2)
                logger.info("Created ontology.json")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create sample files: {e}")
            return False
    
    def download_required_files(self) -> bool:
        """Download all required files that don't exist."""
        existing_files = self.check_files_exist()
        
        # Download GloVe vectors if missing
        if not existing_files["glove.2024.wikigiga.100d.zip"]:
            logger.info("First time setup: Downloading GloVe vectors...")
            
            # Try multiple sources
            for source in self.required_files["glove.2024.wikigiga.100d.zip"]["sources"]:
                if self.download_file(source, "glove.2024.wikigiga.100d.zip"):
                    break
            else:
                logger.error("Failed to download GloVe vectors from all sources")
                return False
        
        # Extract GloVe vectors if zip exists but extracted file doesn't
        logger.info(f"Checking extraction condition:")
        logger.info(f"  - GloVe zip exists: {existing_files['glove.2024.wikigiga.100d.zip']}")
        logger.info(f"  - Extracted file exists: {existing_files['wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_050_combined.txt']}")
        
        if (existing_files["glove.2024.wikigiga.100d.zip"] and 
            not existing_files["wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_050_combined.txt"]):
            logger.info("GloVe zip exists but extracted file missing - extracting now...")
            logger.info("This will create the 1.6GB wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_050_combined.txt file...")
            if not self.extract_glove_vectors():
                logger.error("GloVe extraction failed!")
                return False
            logger.info("GloVe extraction completed - re-checking files...")
            # Re-check files after extraction
            existing_files = self.check_files_exist()
        else:
            logger.info("Extraction condition not met:")
            if not existing_files["glove.2024.wikigiga.100d.zip"]:
                logger.info("  - GloVe zip file is missing")
            if existing_files["wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_050_combined.txt"]:
                logger.info("  - Extracted file already exists")
            logger.info("Skipping extraction step")
        
        # Create sample files if missing
        if not existing_files["dialogues.json"] or not existing_files["ontology.json"]:
            if not self.create_sample_files():
                return False
        
        return True
    
    def verify_all_files(self) -> bool:
        """Verify all required files are present and accessible."""
        logger.info("Verifying all required files...")
        
        for filename, info in self.required_files.items():
            file_path = self.data_dir / filename
            if not file_path.exists():
                logger.error(f"{filename} still missing after setup")
                return False
            
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"{filename}: {size_mb:.1f}MB")
        
        total_size = sum((self.data_dir / f).stat().st_size for f in self.required_files)
        total_size_mb = total_size / (1024 * 1024)
        logger.info(f"Total data directory size: {total_size_mb:.1f}MB")
        
        return True

def main():
    """Main startup function."""
    logger.info("Starting Emotion Detection Backend data setup...")
    
    # Check if we're on Render (multiple possible paths)
    render_paths = ["/app", "/opt/render/project/src", "/opt/render/project/src/emotion-detection-project/emotion-detection/backend"]
    is_render = any(os.path.exists(path) for path in render_paths)
    
    if not is_render:
        logger.info("Not on Render, skipping data setup")
        return True
    
    # Initialize file manager
    file_manager = DataFileManager()
    
    # Download required files
    if not file_manager.download_required_files():
        logger.error("Failed to download required files")
        return False
    
    # Verify all files
    if not file_manager.verify_all_files():
        logger.error("File verification failed")
        return False
    
    logger.info("Data setup completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
