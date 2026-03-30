from ultralytics import YOLO
import os

# Avoid multiprocessing issues
os.environ['OMP_NUM_THREADS'] = '1'

# Load model
model = YOLO('ultralytics/cfg/models/11/yolov11C3TR.yaml')

print("Starting training...")

# Train
model.train(
    data='ultralytics/cfg/datasets/coco128.yaml',     # Small dataset
    epochs=3,              # Just 3 epochs
    imgsz=320,            # Small images
    batch=1,              # Single batch
    device='cpu',         # CPU only
    workers=0,            # No multiprocessing
    name='maxvit_simple'  # Name
)

print("Training done!")