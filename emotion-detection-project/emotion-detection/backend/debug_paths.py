#!/usr/bin/env python3
"""
Debug script to check file paths and existence for GloVe embeddings
"""

import os
from pathlib import Path

def debug_paths():
    """Debug the file paths and existence."""
    print("=== PATH DEBUGGING ===")
    
    # Check current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Check possible data directory paths
    possible_paths = [
        Path("/app/data"),  # Render persistent disk path
        Path("/opt/render/project/src/emotion-detection-project/emotion-detection/backend/data"),  # Render source path
        Path("data"),  # Local development path
        Path("/tmp/data"),  # Temporary path as fallback
    ]
    
    print("\n=== CHECKING POSSIBLE DATA PATHS ===")
    for path in possible_paths:
        print(f"\nPath: {path}")
        print(f"  Exists: {path.exists()}")
        if path.exists():
            print(f"  Is directory: {path.is_dir()}")
            print(f"  Contents:")
            try:
                for item in path.iterdir():
                    if item.is_file():
                        size_mb = item.stat().st_size / (1024 * 1024)
                        print(f"    - {item.name} ({size_mb:.1f}MB)")
                    else:
                        print(f"    - {item.name} (directory)")
            except Exception as e:
                print(f"    Error listing contents: {e}")
    
    # Check specific GloVe files
    print("\n=== CHECKING GLOVE FILES ===")
    glove_files = [
        "glove.2024.wikigiga.100d.zip",
        "wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_050_combined.txt"
    ]
    
    for data_path in possible_paths:
        if data_path.exists():
            print(f"\nIn {data_path}:")
            for glove_file in glove_files:
                file_path = data_path / glove_file
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    print(f"  [EXISTS] {glove_file} exists ({size_mb:.1f}MB)")
                else:
                    print(f"  [MISSING] {glove_file} missing")
    
    # Check environment variables
    print("\n=== ENVIRONMENT VARIABLES ===")
    env_vars = ["PWD", "HOME", "USER", "PATH"]
    for var in env_vars:
        print(f"{var}: {os.environ.get(var, 'Not set')}")
    
    # Check if we're on Render
    print("\n=== RENDER DETECTION ===")
    render_paths = ["/app", "/opt/render/project/src", "/opt/render/project/src/emotion-detection-project/emotion-detection/backend"]
    for path in render_paths:
        exists = os.path.exists(path)
        print(f"{path}: {'[EXISTS]' if exists else '[MISSING]'}")
    
    # Check file permissions
    print("\n=== FILE PERMISSIONS ===")
    for data_path in possible_paths:
        if data_path.exists():
            try:
                stat = data_path.stat()
                print(f"{data_path}:")
                print(f"  Owner: {stat.st_uid}")
                print(f"  Group: {stat.st_gid}")
                print(f"  Permissions: {oct(stat.st_mode)}")
            except Exception as e:
                print(f"{data_path}: Error checking permissions: {e}")

if __name__ == "__main__":
    debug_paths()
