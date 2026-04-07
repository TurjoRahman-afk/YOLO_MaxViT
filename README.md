# YOLO-MaxViT

**Hybrid CNN–Transformer Object Detection via MaxViT Backbone and C3TR Neck integrated into YOLOv11**

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Framework: Ultralytics](https://img.shields.io/badge/Framework-Ultralytics%20YOLOv11-darkgreen)](https://github.com/ultralytics/ultralytics)
[![Training: Complete](https://img.shields.io/badge/Training-Complete%20%E2%9C%94-brightgreen)]()

---

## Overview

YOLO-MaxViT is a research project that integrates **Multi-Axis Vision Transformer (MaxViT)** attention into the YOLOv11 detection backbone, combined with **C3TR transformer blocks** in the detection neck. The goal is to evaluate whether hybrid CNN–Transformer architectures can improve detection accuracy over vanilla YOLOv11n while remaining in a comparable parameter budget.

This repository contains the full modified [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) codebase, custom architecture definition, training scripts, and complete training results on a 20K-image COCO 2017 subset.

> ✅ **Training is complete.** All 300 epochs have finished. No further training updates will be made to this run.

---

## Table of Contents

- [Architecture](#architecture)
- [Key Files](#key-files)
- [Getting Started](#getting-started)
- [Results](#results)
- [Comparison with YOLO11 Family](#comparison-with-yolo11-family)
- [Training Environment](#training-environment)
- [References](#references)
- [License](#license)

---

## Architecture

### Model: `yolov11C3TR`

The architecture replaces standard convolutions at deeper backbone stages with `MaxViTCNNBlock` (multi-axis window + grid attention), and uses `C3TR` transformer blocks in the neck at the two largest feature scales.

```
Backbone
────────────────────────────────────────────────────────────────
Conv → Conv → C2f → Conv → C2f
  → Conv → MaxViTCNNBlock  (512ch,  window 16×16)  ← Local + Global Attention
  → Conv → MaxViTCNNBlock  (1024ch, window 8×8)    ← Local + Global Attention
  → SPPF → C2PSA

Detection Neck
────────────────────────────────────────────────────────────────
Upsample → Concat → C3k2  (256ch,  P3 small scale)   ← CNN
Upsample → Concat → C3k2  (512ch,  P4 mid scale)     ← CNN
Downsample → Concat → C3TR (512ch,  P4 large scale)   ← Transformer
Downsample → Concat → C3TR (1024ch, P5 largest scale) ← Transformer

Detection Head: [P3, P4, P5]
```

**Design rationale**: Small-scale feature maps (P3) use efficient local CNN (`C3k2`) since fine-grained spatial detail matters more than global context. Large-scale feature maps (P4, P5) use `C3TR` to leverage global semantic context for detecting larger objects.

### Model Properties

| Property | Value |
|---|---|
| Parameters | 4.89M |
| GFLOPs | 170.5 |
| Input size | 640 × 640 |
| Classes (COCO) | 80 |
| Architecture layers | 303 |

> **Note on GFLOPs**: The high compute cost (170.5 GFLOPs) relative to parameter count (4.9M) is characteristic of transformer attention — the multi-axis attention passes in `MaxViTCNNBlock` are computationally intensive despite adding few learnable weights.

---

## Key Files

| File | Description |
|---|---|
| `ultralytics/nn/MaxViT.py` | `MaxViTCNNBlock` implementation — multi-axis local window + grid attention |
| `ultralytics/cfg/models/11/yolov11C3TR.yaml` | Custom model architecture YAML |
| `ultralytics/cfg/datasets/coco20k.yaml` | COCO 2017 20K-image subset dataset config |
| `train_custom.py` | Training entry point with all hyperparameters |
| `plot_results.py` | Generates training curve plots from `results.csv` |
| `results/` | Training plots committed to repo |

---

## Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/TurjoRahman-afk/YOLO_MaxViT.git
cd YOLO_MaxViT
pip install -e .
```

### 2. Prepare Dataset

This project was trained on a 20K-image subset of **COCO 2017**.

```
datasets/coco/images/train2017/   ← training images
datasets/coco/labels/train2017/   ← training labels
datasets/coco/images/val2017/     ← validation images
```

The `train20k.txt` file (not included — dataset is gitignored) lists 20,000 image paths sampled from `train2017`. To recreate:

```powershell
# PowerShell — take first 20K lines from train2017.txt
Get-Content datasets/coco/train2017.txt | Select-Object -First 20000 | Set-Content datasets/coco/train20k.txt
```

To train on the **full COCO 2017** dataset instead, change `DATASET` in `train_custom.py` to `"ultralytics/cfg/datasets/coco.yaml"`.

### 3. Train from Scratch

```bash
python train_custom.py
```

Key hyperparameters in `train_custom.py`:

```python
DATASET    = "ultralytics/cfg/datasets/coco20k.yaml"
EPOCHS     = 300
IMGSZ      = 640
BATCH      = 16
OPTIMIZER  = "SGD"
LR0        = 0.01
AMP        = True    # Mixed precision (FP16)
COS_LR     = True    # Cosine LR schedule
RESUME     = False
```

### 4. Resume an Interrupted Run

Set `RESUME = True` in `train_custom.py` — it will continue from `weights/last.pt` automatically.

### 5. Run Inference

```python
from ultralytics import YOLO

model = YOLO("runs/research/yolov11_C3TR_MaxViT_coco20k/weights/best.pt")
results = model("path/to/image.jpg")
results[0].show()
```

---

## Results

> ✅ **Training complete** — 300/300 epochs on COCO 2017 20K subset. No further updates to this run.

### Final Checkpoint

| Model | Dataset | Epochs | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| **YOLO-MaxViT (ours)** | COCO 20K (20,000 imgs) | 300 | **0.405** | **0.263** |

### Training Progress

| Epoch | mAP@0.5 | mAP@0.5:0.95 | Box Loss | Cls Loss |
|---|---|---|---|---|
| 1 | 0.001 | 0.000 | 3.363 | 4.999 |
| 30 | 0.274 | 0.173 | 1.423 | 2.150 |
| 60 | 0.340 | 0.219 | 1.343 | 1.878 |
| 90 | 0.358 | 0.232 | 1.307 | 1.767 |
| 120 | 0.372 | 0.243 | 1.282 | 1.691 |
| 150 | 0.385 | 0.253 | 1.257 | 1.619 |
| 180 | 0.394 | 0.259 | 1.228 | 1.538 |
| 210 | 0.401 | 0.263 | 1.200 | 1.460 |
| 240 | 0.404 | 0.264 | 1.172 | 1.385 |
| 270 | 0.405 | 0.263 | 1.125 | 1.298 |
| **300** | **0.405** | **0.263** | — | — |

The model converged cleanly — per-30-epoch mAP gains dropped below 0.001 after epoch 240, indicating full convergence on this dataset size.

### Training Curves

<table>
  <tr>
    <td><img src="results/training_dashboard.png" width="100%"/></td>
    <td><img src="results/map_curve.png" width="100%"/></td>
    <td><img src="results/loss_curve.png" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><b>Dashboard</b></td>
    <td align="center"><b>mAP Curve</b></td>
    <td align="center"><b>Loss Curve</b></td>
  </tr>
</table>

### Saved Checkpoints

Checkpoints are saved to `runs/research/yolov11_C3TR_MaxViT_coco20k/weights/`:

| File | Description |
|---|---|
| `best.pt` | Highest mAP@0.5:0.95 checkpoint across all epochs |
| `last.pt` | Final epoch (300) checkpoint |
| `epoch30.pt`, `epoch60.pt`, ... | Periodic saves every 30 epochs |

---

## Comparison with YOLO11 Family

> ⚠️ **This is not a fair direct comparison.** Official YOLO11 models are pretrained on the **full COCO 2017 train set (118,287 images)**. YOLO-MaxViT was trained on a **20K-image subset (~17% of full COCO)**. The table below is provided for scale reference only.

| Model | Train Images | mAP@0.5:0.95 | Params (M) | GFLOPs |
|---|---|---|---|---|
| YOLO11n | 118K | 0.395 | 2.6 | 6.5 |
| YOLO11s | 118K | 0.470 | 9.4 | 21.5 |
| YOLO11m | 118K | 0.515 | 20.1 | 68.0 |
| YOLO11l | 118K | 0.534 | 25.3 | 86.9 |
| YOLO11x | 118K | 0.547 | 56.9 | 194.9 |
| **YOLO-MaxViT (ours)** | **20K** | **0.263** | **4.9** | **170.5** |

**Key observations:**

- With only 17% of the training data, YOLO-MaxViT reaches mAP@0.5:0.95 = 0.263 — within ~0.13 of YOLO11n trained on 6× more images
- By **parameter count** (4.9M), it sits between YOLO11n (2.6M) and YOLO11s (9.4M) — a nano-to-small scale model
- By **compute cost** (170.5 GFLOPs), it is comparable to YOLO11l/x — a known transformer trade-off
- Training on the full COCO 118K dataset is expected to close the mAP gap significantly, potentially reaching **mAP@0.5:0.95 ≈ 0.38–0.42**

---

## Training Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 5060 8 GB |
| VRAM utilization | ~6–7 GB (batch=16, AMP enabled) |
| OS | Windows 11 |
| Framework | Ultralytics YOLOv11 (custom fork) |
| Python | 3.10+ |
| Approx. epoch time | ~5–6 min (20K subset) |
| Total training time | ~30 hours |

---

## References

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) — base framework
- [MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/abs/2204.01697) — Tu et al., NeurIPS 2022
- [Swin Transformer](https://arxiv.org/abs/2103.14030) — Liu et al., ICCV 2021
- [COCO Dataset](https://cocodataset.org) — Lin et al., ECCV 2014

---

## License

This project is built on top of [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics), which is licensed under **AGPL-3.0**. All custom modifications in this repository fall under the same license. See [LICENSE](LICENSE) for full details.

---

<p align="center">
  <i>Built as a research experiment in hybrid CNN–Transformer object detection.</i>
</p>

---


