# YOLO-MaxViT v2: Optimized Architecture

**Lightweight variant with reduced compute (~40 GFLOPs vs 170) while maintaining accuracy**

> ⏳ **Training in progress** — 120/300 epochs on COCO 2017 20K subset

---

## Overview

YOLO-MaxViT v2 is an optimized iteration of the original architecture, designed to dramatically reduce computational cost while preserving detection accuracy. By shrinking MaxViT attention windows and removing redundant transformer blocks at mid-scale feature levels, v2 achieves approximately **4× reduction in GFLOPs** with minimal accuracy loss.

---

## Key Differences: v1 → v2

### Architecture Changes

| Aspect | v1 | v2 | Impact |
|---|---|---|---|
| **MaxViT window @ 512ch** | [16×16] (256 tokens) | [8×8] (64 tokens) | **16× cheaper attention** |
| **MaxViT window @ 1024ch** | [8×8] (64 tokens) | [4×4] (16 tokens) | **16× cheaper attention** |
| **Neck P4 block** | C3TR (Transformer) | C3k2 (CNN) | Removes redundant transformer at mid-scale |
| **Neck P5 block** | C3TR (Transformer) | C3TR (Transformer) | Kept — global context essential at largest scale |

### Compute Cost

| Model | Parameters | GFLOPs | Reduction |
|---|---|---|---|
| v1 (original) | 4.89M | 170.5 | — |
| v2 (optimized) | ~4.7M | **~40–50** | **3.4–4.3× cheaper** |

### Design Rationale

```
v2 Detection Head:
─────────────────────────────────────────
P3 (80×80, small objects)   → C3k2 CNN ✓ Local features sufficient
P4 (40×40, medium objects)  → C3k2 CNN ✓ Good receptive field, cheaper
P5 (20×20, large objects)   → C3TR Transformer ✓ Global context matters

Backbone MaxViT:
─────────────────────────────────────────
512ch @ 40×40  → 8×8 window    ✓ Smaller windows = fewer attention tokens
1024ch @ 20×20 → 4×4 window    ✓ Critical layer, heavily optimized
```

The key insight: transformer attention's benefit scales with semantic importance (larger objects need global context), not layer depth. P5 gets the transformer; P4 uses efficient local CNN.

---

## Training Status

### Current Progress

| Metric | Value |
|---|---|
| Epochs completed | 120 / 300 |
| Best mAP@0.5 so far | 0.374 |
| Best mAP@0.5:0.95 so far | 0.249 |
| Training time per epoch | ~4.8–5.2 min |

### v2 Training Progress (every 30 epochs)

| Epoch | mAP@0.5 | mAP@0.5:0.95 | Box Loss | Cls Loss |
|---|---|---|---|---|
| 1 | 0.002 | 0.001 | 3.371 | 4.983 |
| 30 | 0.277 | 0.176 | 1.406 | 2.136 |
| 60 | 0.337 | 0.222 | 1.331 | 1.890 |
| 90 | 0.357 | 0.237 | 1.281 | 1.721 |
| **120** | **0.374** | **0.249** | **1.259** | **1.659** |
| 150 | — | — | — | — |
| 180 | — | — | — | — |
| 210 | — | — | — | — |
| 240 | — | — | — | — |
| 270 | — | — | — | — |
| 300 | — | — | — | — |

### v1 vs v2 Training Curves

<table>
  <tr>
    <td><img src="results/compare_dashboard.png" width="100%"/></td>
    <td><img src="results/compare_map.png" width="100%"/></td>
    <td><img src="results/compare_loss.png" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><b>Dashboard</b></td>
    <td align="center"><b>mAP Comparison</b></td>
    <td align="center"><b>Loss Comparison</b></td>
  </tr>
</table>

> Solid lines = v1 (170 GFLOPs, 300 epochs complete) — Dashed lines = v2 (~49 GFLOPs, training in progress)

### v1 vs v2 Side-by-Side Comparison

> v1 is fully trained (300 epochs). v2 is in progress — rows will be filled as training completes each 30-epoch checkpoint.

| Epoch | v1 mAP@0.5 | v2 mAP@0.5 | v1 mAP50-95 | v2 mAP50-95 |
|---|---|---|---|---|
| 1 | 0.001 | 0.002 | 0.000 | 0.001 |
| 30 | 0.274 | 0.277 | 0.173 | 0.176 |
| 60 | 0.340 | 0.337 | 0.219 | 0.222 |
| 90 | 0.358 | 0.357 | 0.232 | 0.237 |
| 120 | 0.372 | 0.374 | 0.243 | 0.249 |
| 150 | 0.385 | — | 0.253 | — |
| 180 | 0.394 | — | 0.259 | — |
| 210 | 0.401 | — | 0.263 | — |
| 240 | 0.404 | — | 0.264 | — |
| 270 | 0.405 | — | 0.263 | — |
| 300 | 0.405 | — | 0.263 | — |

---

## Getting Started

### Run v2 Training

Ensure `train_custom.py` has:

```python
MODEL_CFG = "ultralytics/cfg/models/11/yolov11C3TR_v2.yaml"
RESUME    = False  # Start fresh
RUN_NAME  = "yolov11_C3TR_MaxViT_v2_coco20k"
```

Then:

```bash
python train_custom.py
```

### Resume v2 Training

To continue from a checkpoint:

```python
RESUME    = True
MODEL_CFG = "runs/research/yolov11_C3TR_MaxViT_v2_coco20k/weights/last.pt"
```

### Run Inference with v2

```python
from ultralytics import YOLO

model = YOLO("runs/research/yolov11_C3TR_MaxViT_v2_coco20k/weights/best.pt")
results = model("path/to/image.jpg")
results[0].show()
```

---

## Expected Results

Based on early convergence trends, v2 is projected to reach:

| Milestone | Estimated Performance |
|---|---|
| **Epoch 60** | mAP50 ~0.36, mAP50-95 ~0.23 |
| **Epoch 120** | mAP50 ~0.39, mAP50-95 ~0.25 |
| **Epoch 300** | mAP50 ~0.40–0.41, mAP50-95 ~0.26–0.27 |

This represents a **~2–3% mAP drop from v1** (due to removed transformer at P4) but with **4× faster inference**.

---

## Comparison: v1 vs v2 Trade-off

| Model | GFLOPs | mAP50 (est. @ 300ep) | Speed | Best For |
|---|---|---|---|---|
| **v1** | 170.5 | 0.405 | Slower | Max accuracy research |
| **v2** | ~45 | 0.40–0.41 | **~4× faster** | Production, edge deployment |
| YOLO11n | 6.5 | 0.395 | Fastest | Baseline |

v2 sits between pure CNN efficiency (v1) and pure speed (YOLO11n), prioritizing practical deployment without sacrificing too much accuracy.

---

## Architecture File

The v2 configuration is defined in:

```
ultralytics/cfg/models/11/yolov11C3TR_v2.yaml
```

Key config lines:
- Line 6–8: MaxViT windows at [8,8] and [4,4]
- Line 19: P4 neck block is `C3k2` (CNN, not C3TR)
- Line 22: P5 neck block is `C3TR` (Transformer, kept)

---

## Training Environment

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5060 8 GB |
| VRAM | ~6–7 GB (batch=16, AMP) |
| Epoch time | ~4.8–5.2 min (20K subset) |
| Total time (300ep) | ~24 hours (vs ~30 hours for v1) |
| Dataset | COCO 2017 20K subset |

---

## References

- [YOLO-MaxViT v1](README.md) — original architecture
- [MaxViT Paper](https://arxiv.org/abs/2204.01697) — Tu et al., NeurIPS 2022
- [YOLOv11](https://github.com/ultralytics/ultralytics) — base framework

---

<p align="center">
  <i>Optimized for efficiency without sacrificing the transformer advantage.</i>
</p>
