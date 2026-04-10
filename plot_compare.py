"""
Generate v1 vs v2 comparison plots.
Saves PNGs to results/ for GitHub display.
"""

import csv
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

V1_CSV  = "runs/research/yolov11_C3TR_MaxViT_coco20k/results.csv"
V2_CSV  = "runs/research/yolov11_C3TR_MaxViT_v2_coco20k/results.csv"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load CSV helper
# ---------------------------------------------------------------------------
def load_csv(path):
    epochs, map50, map95, box_loss, cls_loss = [], [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            map50.append(float(row["metrics/mAP50(B)"]))
            map95.append(float(row["metrics/mAP50-95(B)"]))
            box_loss.append(float(row["val/box_loss"]))
            cls_loss.append(float(row["val/cls_loss"]))
    return epochs, map50, map95, box_loss, cls_loss

v1 = load_csv(V1_CSV)
v2 = load_csv(V2_CSV)

# ---------------------------------------------------------------------------
# Plot 1 — mAP@0.5 comparison
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(v1[0], v1[1], color="#2196F3", linewidth=2, label="v1 mAP@0.5  (170 GFLOPs)")
ax.plot(v2[0], v2[1], color="#2196F3", linewidth=2, linestyle="--", label="v2 mAP@0.5  (~49 GFLOPs)")
ax.plot(v1[0], v1[2], color="#FF5722", linewidth=2, label="v1 mAP@0.5:0.95")
ax.plot(v2[0], v2[2], color="#FF5722", linewidth=2, linestyle="--", label="v2 mAP@0.5:0.95")

ax.set_title("YOLO-MaxViT v1 vs v2 — mAP Comparison (COCO 20K)", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("mAP", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

# Annotate last v2 values
ax.annotate(f" v2: {v2[1][-1]:.3f}", xy=(v2[0][-1], v2[1][-1]), fontsize=9, color="#2196F3")
ax.annotate(f" v2: {v2[2][-1]:.3f}", xy=(v2[0][-1], v2[2][-1]), fontsize=9, color="#FF5722")

plt.tight_layout()
out = os.path.join(OUT_DIR, "compare_map.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 2 — Loss comparison
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(v1[0], v1[3], color="#4CAF50", linewidth=2, label="v1 Box Loss")
ax.plot(v2[0], v2[3], color="#4CAF50", linewidth=2, linestyle="--", label="v2 Box Loss")
ax.plot(v1[0], v1[4], color="#9C27B0", linewidth=2, label="v1 Cls Loss")
ax.plot(v2[0], v2[4], color="#9C27B0", linewidth=2, linestyle="--", label="v2 Cls Loss")

ax.set_title("YOLO-MaxViT v1 vs v2 — Validation Loss Comparison (COCO 20K)", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(OUT_DIR, "compare_loss.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ---------------------------------------------------------------------------
# Plot 3 — 2x2 dashboard comparison
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("YOLO-MaxViT v1 vs v2 — Training Dashboard (COCO 20K)", fontsize=15, fontweight="bold")

# mAP@0.5
axes[0,0].plot(v1[0], v1[1], color="#2196F3", linewidth=2, label="v1")
axes[0,0].plot(v2[0], v2[1], color="#2196F3", linewidth=2, linestyle="--", label="v2")
axes[0,0].set_title("mAP@0.5"); axes[0,0].set_xlabel("Epoch")
axes[0,0].legend(fontsize=9); axes[0,0].grid(True, alpha=0.3)

# mAP@0.5:0.95
axes[0,1].plot(v1[0], v1[2], color="#FF5722", linewidth=2, label="v1")
axes[0,1].plot(v2[0], v2[2], color="#FF5722", linewidth=2, linestyle="--", label="v2")
axes[0,1].set_title("mAP@0.5:0.95"); axes[0,1].set_xlabel("Epoch")
axes[0,1].legend(fontsize=9); axes[0,1].grid(True, alpha=0.3)

# Box Loss
axes[1,0].plot(v1[0], v1[3], color="#4CAF50", linewidth=2, label="v1")
axes[1,0].plot(v2[0], v2[3], color="#4CAF50", linewidth=2, linestyle="--", label="v2")
axes[1,0].set_title("Val Box Loss"); axes[1,0].set_xlabel("Epoch")
axes[1,0].legend(fontsize=9); axes[1,0].grid(True, alpha=0.3)

# Cls Loss
axes[1,1].plot(v1[0], v1[4], color="#9C27B0", linewidth=2, label="v1")
axes[1,1].plot(v2[0], v2[4], color="#9C27B0", linewidth=2, linestyle="--", label="v2")
axes[1,1].set_title("Val Cls Loss"); axes[1,1].set_xlabel("Epoch")
axes[1,1].legend(fontsize=9); axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(OUT_DIR, "compare_dashboard.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

print("\nAll comparison plots saved to results/")
