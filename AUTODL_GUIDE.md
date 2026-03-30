# 🚀 AutoDL Training Guide for YOLOv11 + MaxViT

Complete guide to train your YOLOv11 + MaxViT model on AutoDL platform with GPU acceleration.

## 📋 Prerequisites

1. **AutoDL Account**: Sign up at [AutoDL](https://www.autodl.com/)
2. **GitHub Access**: Your repository is ready at `https://github.com/TurjoRahman-afk/yolov11-maxvit`
3. **Dataset**: Prepare your dataset or use provided sample

## 🖥️ AutoDL Setup

### Step 1: Create AutoDL Instance

1. **Login** to AutoDL platform
2. **Create New Instance**:
   - **GPU**: RTX 4090 / A100 / V100 (recommended)
   - **Framework**: PyTorch 2.0+
   - **Storage**: 50GB+ (for datasets)
   - **Region**: Choose closest to you

### Step 2: Connect to Your Instance

```bash
# SSH into your AutoDL instance (provided by AutoDL)
ssh root@your-instance-ip
```

### Step 3: Quick Setup

Run our automated setup script:

```bash
# Download and run setup script
wget https://raw.githubusercontent.com/TurjoRahman-afk/yolov11-maxvit/main/setup_autodl.sh
chmod +x setup_autodl.sh
./setup_autodl.sh
```

**Manual Setup Alternative:**
```bash
# Clone repository
git clone https://github.com/TurjoRahman-afk/yolov11-maxvit.git
cd yolov11-maxvit

# Install package
pip install -e .

# Install additional dependencies
pip install wandb tensorboard GPUtil psutil matplotlib pandas
```

## 🎯 Training Your Model

### Quick Test Training (5 epochs)

```bash
cd yolov11-maxvit
python train_custom.py
# Choose option 1 for quick test
```

### Full Training (300 epochs)

```bash
python train_custom.py
# Choose option 2 for full training
```

### Custom Training Script

```python
from ultralytics import YOLO

# Load your custom MaxViT model
model = YOLO('ultralytics/cfg/models/11/yolov11C3TR.yaml')

# Train with custom settings
results = model.train(
    data='dataset_config.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # Use GPU
    project='runs/train',
    name='yolov11_maxvit_custom',
    # MaxViT-optimized settings
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,
    cos_lr=True,
    warmup_epochs=3
)
```

## 📊 Monitoring Training

### Real-time System Monitoring

```bash
# Start monitoring in a separate terminal
python monitor_autodl.py
```

This will show:
- 🖥️ GPU utilization and memory
- 💻 CPU and RAM usage
- 💾 Disk usage
- 📈 Training progress

### TensorBoard Monitoring

```bash
# Start TensorBoard
tensorboard --logdir runs/train --host 0.0.0.0 --port 6006

# Access via AutoDL's port forwarding
# Usually: http://your-instance-ip:6006
```

### Weights & Biases Integration

```python
import wandb

# Initialize (run this before training)
wandb.login()  # Enter your API key
wandb.init(project="yolov11-maxvit-training")

# Training will automatically log to W&B
```

## 📁 Dataset Setup

### Option 1: Upload Your Dataset

```bash
# Create dataset directory
mkdir -p datasets/my_dataset/{images,labels}/{train,val}

# Upload via AutoDL file manager or SCP
scp -r my_dataset/ root@your-instance:/root/yolov11-maxvit/datasets/
```

### Option 2: Download Public Dataset

```bash
# Example: Download COCO dataset
cd datasets
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip && unzip val2017.zip

# Convert annotations if needed
python ultralytics/data/converter.py --format coco
```

### Dataset Structure
```
datasets/
└── my_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── dataset.yaml
```

## ⚙️ Training Configuration

### Optimize for Your GPU

**RTX 4090 (24GB):**
```python
batch_size = 32
workers = 8
imgsz = 640
```

**RTX 3090 (12GB):**
```python
batch_size = 16
workers = 6
imgsz = 640
```

**V100 (16GB):**
```python
batch_size = 24
workers = 8
imgsz = 640
```

### MaxViT-Specific Settings

```yaml
# In your training config
optimizer: AdamW          # Better for transformers
lr0: 0.001               # Lower learning rate
weight_decay: 0.0005     # Regularization
warmup_epochs: 3         # Gradual warmup
cos_lr: True            # Cosine learning rate
```

## 📈 Performance Optimization

### Memory Optimization

```python
# Enable mixed precision training
model.train(
    amp=True,  # Automatic Mixed Precision
    cache=True,  # Cache images in RAM
    workers=8,   # Multi-processing
)
```

### Speed Optimization

```bash
# Set optimal PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=8
```

## 🔍 Troubleshooting

### Common Issues

**GPU Out of Memory:**
```python
# Reduce batch size
batch_size = 8
# Or reduce image size
imgsz = 512
```

**Slow Training:**
```python
# Enable optimizations
cache=True,      # Cache dataset
workers=8,       # More workers
device=[0,1],    # Multi-GPU if available
```

**Model Not Loading:**
```bash
# Check model path
ls ultralytics/cfg/models/11/yolov11C3TR.yaml
# Verify installation
python -c "from ultralytics import YOLO; print('OK')"
```

## 💾 Saving Results

### Automatic Saves
- Models saved every 10 epochs in `runs/train/yolov11_maxvit/weights/`
- Best model: `best.pt`
- Latest model: `last.pt`

### Manual Save
```python
# Save trained model
model.save('yolov11_maxvit_final.pt')

# Export for deployment
model.export(format='onnx')
```

### Download Results
```bash
# Compress results
tar -czf training_results.tar.gz runs/train/

# Download via AutoDL file manager
# Or use SCP: scp root@instance:/root/yolov11-maxvit/training_results.tar.gz ./
```

## 🚀 Advanced Features

### Multi-GPU Training
```python
# If you have multiple GPUs
model.train(device=[0,1,2,3])  # Use all 4 GPUs
```

### Resume Training
```python
# Resume from checkpoint
model = YOLO('runs/train/yolov11_maxvit/weights/last.pt')
model.train(resume=True)
```

### Custom Callbacks
```python
from ultralytics.utils.callbacks import default_callbacks

def on_epoch_end(trainer):
    # Custom logic after each epoch
    pass

# Add custom callback
model.add_callback("on_train_epoch_end", on_epoch_end)
```

## 📊 Expected Results

**YOLOv11C3TR with MaxViT:**
- **Parameters**: ~4.9M
- **mAP50**: 0.65+ (COCO dataset)
- **Inference**: ~15ms (GPU)
- **Training time**: ~12 hours (100 epochs, COCO)

## 🎉 Success Checklist

- ✅ AutoDL instance running
- ✅ Repository cloned and installed
- ✅ Dataset uploaded/configured
- ✅ Training started successfully
- ✅ Monitoring tools active
- ✅ Model saving correctly
- ✅ Results downloaded

## 📞 Support

If you encounter issues:
1. Check the monitoring logs: `cat training_monitor.log`
2. Review AutoDL documentation
3. Check GitHub issues in your repository
4. AutoDL community support

---

**🎯 Your YOLOv11 + MaxViT model is now ready for professional training on AutoDL!**
