#!/usr/bin/env python3
import os
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Run SafeRide helmet and plate detection training")
    parser.add_argument("--helmet-only", action="store_true", help="Train only helmet detection model")
    parser.add_argument("--plate-only", action="store_true", help="Train only plate detection model")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use for training")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    if not args.plate_only:
        print("\n===== Running Helmet Detection Training =====")
        subprocess.run([sys.executable, "/app/scripts/helmet_training.py"], check=False)
    
    if not args.helmet_only:
        print("\n===== Running Plate Detection Training =====")
        subprocess.run([sys.executable, "/app/scripts/plate_training.py"], check=False)

    print("\nTraining completed!\nModels ready for deployment.")

if __name__ == "__main__":
    main()