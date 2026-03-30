#!/bin/bash

# AutoDL Setup Script for YOLOv11 + MaxViT
# Run this script on your AutoDL instance

echo "🚀 Setting up YOLOv11 + MaxViT on AutoDL..."

# Update system packages
apt update
apt install -y git wget unzip

# Clone your repository
echo "📥 Cloning YOLOv11 + MaxViT repository..."
git clone https://github.com/TurjoRahman-afk/yolov11-maxvit.git
cd yolov11-maxvit

# Install your custom ultralytics package
echo "📦 Installing custom ultralytics package..."
pip install -e .

# Install additional dependencies for training
pip install wandb  # For experiment tracking
pip install tensorboard  # For monitoring
pip install opencv-python-headless  # Headless version for servers

# Create directories for datasets and experiments
mkdir -p datasets
mkdir -p runs/train
mkdir -p runs/val

# Download sample dataset (COCO8 for testing)
echo "📊 Downloading sample dataset..."
wget -O coco8.zip https://ultralytics.com/assets/coco8.zip
unzip coco8.zip
mv coco8 datasets/

echo "✅ Setup complete! Your YOLOv11 + MaxViT is ready for training on AutoDL"
echo "🎯 Next steps:"
echo "1. Upload your dataset or use the sample coco8 dataset"
echo "2. Run: python train_custom.py"
echo "3. Monitor training with: tensorboard --logdir runs/train"
