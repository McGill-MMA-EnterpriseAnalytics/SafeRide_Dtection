#!/usr/bin/env python3
"""
Runner script to execute both training notebooks in sequence
"""
import os
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Run SafeRide helmet and plate detection training")
    parser.add_argument("--helmet-only", action="store_true", 
                        help="Train only helmet detection model")
    parser.add_argument("--plate-only", action="store_true",
                        help="Train only plate detection model")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU ID to use for training")
    
    args = parser.parse_args()
    
    # Set GPU device if specified
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)        
    # Run helmet detection training
    if not args.plate_only:
        print("\n===== Running Helmet Detection Training =====")
        result = subprocess.run([sys.executable, "/app/scripts/helmet_training_s.py"], check=False)
        if result.returncode != 0:
            print(f"Warning: Helmet training exited with return code {result.returncode}")
    
    # Run plate detection training
    if not args.helmet_only:
        print("\n===== Running License Plate Detection Training =====")
        result = subprocess.run([sys.executable, "/app/scripts/plate_training_s.py"], check=False)
        if result.returncode != 0:
            print(f"Warning: Plate training exited with return code {result.returncode}")
    
    print("\nTraining completed!")
    
    # Display model paths
    models_dir = "/app/models"
    
    if os.path.exists(os.path.join(models_dir, "Final_Helmet.pt")):
        print(f"✅ Helmet model saved to: {os.path.join(models_dir, 'Final_Helmet.pt')}")
    
    if os.path.exists(os.path.join(models_dir, "Final_Plates.pt")):
        print(f"✅ Plate model saved to: {os.path.join(models_dir, 'Final_Plates.pt')}")
    
    print("\nModels are ready for deployment in the inference Docker container.")

if __name__ == "__main__":
    main()