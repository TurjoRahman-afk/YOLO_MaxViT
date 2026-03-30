"""
YOLOv11-C3TR + MaxViT Training Script — COCO 2017
Research paper training configuration.

Model  : yolov11C3TR  (MaxViT backbone + C3TR transformer neck)
Dataset: COCO 2017    (118,287 train / 5,000 val / 80 classes)
"""

import os
from ultralytics import YOLO

os.environ["OMP_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
# Config  (edit here — safe at module level on Windows)
# ---------------------------------------------------------------------------
DEVICE    = "0"    # cuda:0 GPU
MODEL_CFG = "ultralytics/cfg/models/11/yolov11C3TR.yaml"
DATASET   = "ultralytics/cfg/datasets/coco20k.yaml"  # 20K subset — ~5-6 min/epoch vs 30 min full COCO
PROJECT   = "runs/research"
RUN_NAME  = "yolov11_C3TR_MaxViT_coco20k"

# Resume control: set RESUME = True to continue from a stopped run
RESUME          = True
LAST_CHECKPOINT = f"{PROJECT}/{RUN_NAME}/weights/last.pt"

EPOCHS        = 300
IMGSZ         = 640
BATCH         = 16    # increased: ~3.9 GB headroom available on RTX 5060 8 GB
WORKERS       = 4     # reduced from 8 to lower memory pressure
LR0           = 0.01
LRF           = 0.01
MOMENTUM      = 0.937
WEIGHT_DECAY  = 0.0005
WARMUP_EPOCHS = 3
PATIENCE      = 50
IOU           = 0.7
OPTIMIZER     = "SGD"
AMP           = True

# ---------------------------------------------------------------------------
# REQUIRED on Windows: all spawned dataloader workers re-import this file,
# so training code must live inside  if __name__ == '__main__'
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print(f"Using device: GPU (cuda:0)")

    # Load model (fresh start OR resume from checkpoint)
    if RESUME:
        print(f"Resuming from checkpoint: {LAST_CHECKPOINT}")
        model = YOLO(LAST_CHECKPOINT)
    else:
        model = YOLO(MODEL_CFG)

    print(f"\n{'='*60}")
    print(f"  Model  : {MODEL_CFG}")
    print(f"  Dataset: {DATASET}")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {EPOCHS}  |  imgsz: {IMGSZ}  |  batch: {BATCH}")
    print(f"{'='*60}\n")

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    results = model.train(
        data          = DATASET,
        epochs        = EPOCHS,
        imgsz         = IMGSZ,
        batch         = BATCH,
        device        = DEVICE,
        workers       = WORKERS,
        optimizer     = OPTIMIZER,
        lr0           = LR0,
        lrf           = LRF,
        momentum      = MOMENTUM,
        weight_decay  = WEIGHT_DECAY,
        warmup_epochs = WARMUP_EPOCHS,
        patience      = PATIENCE,
        iou           = IOU,
        amp           = AMP,
        cos_lr        = True,
        close_mosaic  = 10,
        augment       = True,
        plots         = True,
        save          = True,
        save_period   = 30,
        resume        = RESUME,
        project       = PROJECT,
        name          = RUN_NAME,
        exist_ok      = RESUME,
        verbose       = True,
    )

    # -----------------------------------------------------------------------
    # Final validation
    # -----------------------------------------------------------------------
    print("\nRunning final validation on COCO val2017...")
    metrics = model.val(
        data    = DATASET,
        imgsz   = IMGSZ,
        batch   = BATCH,
        device  = DEVICE,
        workers = WORKERS,
        plots   = True,
        verbose = True,
    )

    print(f"\n{'='*60}")
    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print(f"  Results saved to: {PROJECT}/{RUN_NAME}")
    print(f"{'='*60}\n")
    print("Training done!")
