#!/usr/bin/env python3
"""
Test script to demonstrate training progress tracking.
Run this script to test the new progress tracking features.
"""

import requests
import time
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_training_progress():
    """Test the training progress tracking functionality."""
    
    print("🚀 Testing Training Progress Tracking")
    print("=" * 50)
    
    # 1. Check initial status
    print("\n1. Checking initial training status...")
    try:
        response = requests.get(f"{BASE_URL}/api/models/training/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Training status: {json.dumps(status, indent=2)}")
        else:
            print(f"❌ Failed to get status: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 2. Check if models are available
    print("\n2. Checking model availability...")
    try:
        response = requests.get(f"{BASE_URL}/api/models/status")
        if response.status_code == 200:
            models_status = response.json()
            print(f"✅ Models status: {json.dumps(models_status, indent=2)}")
        else:
            print(f"❌ Failed to get models status: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 3. Start Random Forest training
    print("\n3. Starting Random Forest training...")
    try:
        response = requests.post(f"{BASE_URL}/api/models/train/random_forest")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Training started: {json.dumps(result, indent=2)}")
        elif response.status_code == 409:
            print("⚠️  Training already in progress, monitoring existing training...")
        else:
            print(f"❌ Failed to start training: {response.status_code}")
            print(f"Response: {response.text}")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 4. Monitor training progress
    print("\n4. Monitoring training progress...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            # Get current progress
            response = requests.get(f"{BASE_URL}/api/models/training/progress")
            if response.status_code == 200:
                progress_data = response.json()
                
                if progress_data['status'] == 'training_in_progress':
                    progress = progress_data['progress']
                    
                    # Clear screen and show progress
                    print("\033[2J\033[H")  # Clear screen
                    print("🚀 Random Forest Training Progress")
                    print("=" * 50)
                    print(f"Status: {progress['status']}")
                    print(f"Current Step: {progress['current_step']}")
                    print(f"Progress: {progress['progress_percentage']:.1f}%")
                    
                    if 'elapsed_time' in progress:
                        elapsed = progress['elapsed_time']
                        print(f"Elapsed Time: {elapsed:.1f} seconds")
                    
                    if 'current_params' in progress and progress['current_params']:
                        print(f"Current Parameters: {progress['current_params']}")
                    
                    if 'best_score' in progress and progress['best_score'] > 0:
                        print(f"Best Score: {progress['best_score']:.4f}")
                    
                    print("\nRecent Messages:")
                    for msg in progress['messages'][-5:]:  # Show last 5 messages
                        print(f"  • {msg}")
                    
                    print(f"\nLast Update: {datetime.now().strftime('%H:%M:%S')}")
                    print("Press Ctrl+C to stop monitoring")
                    
                elif progress_data['status'] == 'no_training':
                    print("✅ No training in progress")
                    break
                else:
                    print(f"⚠️  Unexpected status: {progress_data['status']}")
                    break
                    
            else:
                print(f"❌ Failed to get progress: {response.status_code}")
                break
            
            # Wait before next update
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
    
    # 5. Final status check
    print("\n5. Final training status...")
    try:
        response = requests.get(f"{BASE_URL}/api/models/training/status")
        if response.status_code == 200:
            final_status = response.json()
            print(f"✅ Final status: {json.dumps(final_status, indent=2)}")
        else:
            print(f"❌ Failed to get final status: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_stop_training():
    """Test stopping training functionality."""
    print("\n🛑 Testing stop training functionality...")
    
    try:
        response = requests.post(f"{BASE_URL}/api/models/training/stop")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Stop result: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Failed to stop training: {response.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🎯 Emotion Detection Training Progress Test")
    print("Make sure your FastAPI server is running on http://localhost:8000")
    print("=" * 60)
    
    try:
        test_training_progress()
        
        # Optionally test stop functionality
        user_input = input("\nWould you like to test stop training functionality? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            test_stop_training()
            
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
    
    print("\n✨ Test completed!")
