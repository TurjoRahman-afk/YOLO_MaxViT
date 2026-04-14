"""
plot_eval.py — YOLO-MaxViT v2 Evaluation Dashboard Generator
Composites evaluation curves, confusion matrices, and sample predictions
into clean dashboard PNGs saved to results/
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

RUN_DIR = Path("runs/research/yolov11_C3TR_MaxViT_v2_coco20k")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def load(name):
    return mpimg.imread(RUN_DIR / name)

def hide_axes(ax):
    ax.axis("off")

# ── 1. Evaluation Curves Dashboard ───────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("YOLO-MaxViT v2 — Evaluation Curves", fontsize=18,
             fontweight="bold", color="white", y=0.98)

titles = ["Precision–Recall Curve", "F1–Confidence Curve",
          "Precision–Confidence Curve", "Recall–Confidence Curve"]
files  = ["PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png"]

for ax, fname, title in zip(axes.flat, files, titles):
    ax.imshow(load(fname))
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
    hide_axes(ax)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = OUT_DIR / "eval_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out}")

# ── 2. Confusion Matrix Dashboard ────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("YOLO-MaxViT v2 — Confusion Matrix", fontsize=18,
             fontweight="bold", color="white", y=1.01)

cm_files  = ["confusion_matrix.png", "confusion_matrix_normalized.png"]
cm_titles = ["Raw Counts", "Normalized"]

for ax, fname, title in zip(axes, cm_files, cm_titles):
    ax.imshow(load(fname))
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=8)
    hide_axes(ax)

plt.tight_layout()
out = OUT_DIR / "eval_confusion.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out}")

# ── 3. Sample Predictions (3 batches × GT + Pred) ────────────────────────────

fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("YOLO-MaxViT v2 — Sample Predictions vs Ground Truth",
             fontsize=18, fontweight="bold", color="white", y=0.99)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.08, wspace=0.04)

for row in range(3):
    ax_gt   = fig.add_subplot(gs[row, 0])
    ax_pred = fig.add_subplot(gs[row, 1])

    ax_gt.imshow(load(f"val_batch{row}_labels.jpg"))
    ax_pred.imshow(load(f"val_batch{row}_pred.jpg"))

    hide_axes(ax_gt)
    hide_axes(ax_pred)

    if row == 0:
        ax_gt.set_title("Ground Truth", color="white", fontsize=13,
                        fontweight="bold", pad=8)
        ax_pred.set_title("Predictions", color="white", fontsize=13,
                          fontweight="bold", pad=8)

    ax_gt.set_ylabel(f"Batch {row}", color="#aaaaaa", fontsize=10,
                     rotation=90, labelpad=6)
    ax_gt.axis("on")
    ax_gt.tick_params(left=False, bottom=False,
                      labelleft=False, labelbottom=False)
    for spine in ax_gt.spines.values():
        spine.set_visible(False)

out = OUT_DIR / "eval_predictions.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out}")

# ── 4. Full Eval Dashboard (all in one) ──────────────────────────────────────

fig = plt.figure(figsize=(24, 28))
fig.patch.set_facecolor("#0d1117")
fig.suptitle("YOLO-MaxViT v2 — Full Evaluation Dashboard\n"
             "mAP@0.5: 0.4129  |  mAP@0.5:0.95: 0.2756  |  "
             "Precision: 0.5362  |  Recall: 0.4019  |  F1: 0.459  |  GFLOPs: 49.3",
             fontsize=14, fontweight="bold", color="white", y=0.995)

gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.12, wspace=0.06)

# Row 0: 4 eval curves
curve_files  = ["PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png"]
curve_titles = ["PR Curve", "F1 Curve", "Precision Curve", "Recall Curve"]
for col, (fname, title) in enumerate(zip(curve_files, curve_titles)):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow(load(fname))
    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=5)
    hide_axes(ax)

# Row 1: 2 confusion matrices (span 2 cols each)
ax_cm1 = fig.add_subplot(gs[1, 0:2])
ax_cm2 = fig.add_subplot(gs[1, 2:4])
ax_cm1.imshow(load("confusion_matrix.png"))
ax_cm2.imshow(load("confusion_matrix_normalized.png"))
ax_cm1.set_title("Confusion Matrix (Raw)", color="white", fontsize=10, fontweight="bold", pad=5)
ax_cm2.set_title("Confusion Matrix (Normalized)", color="white", fontsize=10, fontweight="bold", pad=5)
hide_axes(ax_cm1)
hide_axes(ax_cm2)

# Rows 2–3: 3 batches × GT + Pred (span 2 cols each across 2 rows, 3 batches)
pred_gs = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[2:4, :],
                                           hspace=0.06, wspace=0.04)
for row in range(3):
    ax_gt   = fig.add_subplot(pred_gs[row, 0])
    ax_pred = fig.add_subplot(pred_gs[row, 1])
    ax_gt.imshow(load(f"val_batch{row}_labels.jpg"))
    ax_pred.imshow(load(f"val_batch{row}_pred.jpg"))
    if row == 0:
        ax_gt.set_title("Ground Truth", color="white", fontsize=10,
                        fontweight="bold", pad=5)
        ax_pred.set_title("Predictions", color="white", fontsize=10,
                          fontweight="bold", pad=5)
    hide_axes(ax_gt)
    hide_axes(ax_pred)

out = OUT_DIR / "eval_dashboard.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {out}")

print("\nAll evaluation plots saved to results/")
