#!/usr/bin/env python3

import argparse
import os

from src.training.train import train_model

def main():
    parser = argparse.ArgumentParser(description="Run training of MobileNetV2 model for fire detection.")
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if possible (using CPU by default)')
    parser.add_argument('--config', type=str, default='config/training_cfg.yaml',
                        help='Path to training config YAML')
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    train_model(config_path=args.config, use_gpu=args.gpu)

if __name__ == "__main__":
    main()