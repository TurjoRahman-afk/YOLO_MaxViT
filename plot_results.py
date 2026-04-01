"""
Generate training progress plots from results.csv
and save them to runs/research/yolov11_C3TR_MaxViT_coco20k/
"""

import csv
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CSV_PATH = "runs/research/yolov11_C3TR_MaxViT_coco20k/results.csv"
OUT_DIR  = "runs/research/yolov11_C3TR_MaxViT_coco20k"

# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------
epochs, map50, map95, box_loss, cls_loss, dfl_loss = [], [], [], [], [], []

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        map50.append(float(row["metrics/mAP50(B)"]))
        map95.append(float(row["metrics/mAP50-95(B)"]))
        box_loss.append(float(row["val/box_loss"]))
        cls_loss.append(float(row["val/cls_loss"]))
        dfl_loss.append(float(row["val/dfl_loss"]))

# ---------------------------------------------------------------------------
# Plot 1 — mAP Curves
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, map50,  color="#2196F3", linewidth=2, label="mAP@0.5")
ax.plot(epochs, map95,  color="#FF5722", linewidth=2, label="mAP@0.5:0.95")

ax.set_title("YOLO-MaxViT — mAP Training Curve (COCO 20K)", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("mAP", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

# Annotate last values
ax.annotate(f"  {map50[-1]:.3f}", xy=(epochs[-1], map50[-1]), fontsize=10, color="#2196F3", fontweight="bold")
ax.annotate(f"  {map95[-1]:.3f}", xy=(epochs[-1], map95[-1]), fontsize=10, color="#FF5722", fontweight="bold")

plt.tight_layout()
out_map = os.path.join(OUT_DIR, "map_curve.png")
plt.savefig(out_map, dpi=150)
plt.close()
print(f"Saved: {out_map}")

# ---------------------------------------------------------------------------
# Plot 2 — Loss Curves
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, box_loss, color="#4CAF50", linewidth=2, label="Val Box Loss")
ax.plot(epochs, cls_loss, color="#9C27B0", linewidth=2, label="Val Cls Loss")
ax.plot(epochs, dfl_loss, color="#FF9800", linewidth=2, label="Val DFL Loss")

ax.set_title("YOLO-MaxViT — Validation Loss Curves (COCO 20K)", fontsize=14, fontweight="bold")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_loss = os.path.join(OUT_DIR, "loss_curve.png")
plt.savefig(out_loss, dpi=150)
plt.close()
print(f"Saved: {out_loss}")

# ---------------------------------------------------------------------------
# Plot 3 — Combined 2x2 dashboard
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("YOLO-MaxViT Training Dashboard — COCO 20K", fontsize=15, fontweight="bold")

# mAP@0.5
axes[0,0].plot(epochs, map50, color="#2196F3", linewidth=2)
axes[0,0].set_title("mAP@0.5")
axes[0,0].set_xlabel("Epoch"); axes[0,0].grid(True, alpha=0.3)
axes[0,0].annotate(f"Latest: {map50[-1]:.3f}", xy=(0.05, 0.88), xycoords="axes fraction", fontsize=10, color="#2196F3")

# mAP@0.5:0.95
axes[0,1].plot(epochs, map95, color="#FF5722", linewidth=2)
axes[0,1].set_title("mAP@0.5:0.95")
axes[0,1].set_xlabel("Epoch"); axes[0,1].grid(True, alpha=0.3)
axes[0,1].annotate(f"Latest: {map95[-1]:.3f}", xy=(0.05, 0.88), xycoords="axes fraction", fontsize=10, color="#FF5722")

# Box Loss
axes[1,0].plot(epochs, box_loss, color="#4CAF50", linewidth=2)
axes[1,0].set_title("Val Box Loss")
axes[1,0].set_xlabel("Epoch"); axes[1,0].grid(True, alpha=0.3)

# Cls Loss
axes[1,1].plot(epochs, cls_loss, color="#9C27B0", linewidth=2)
axes[1,1].set_title("Val Cls Loss")
axes[1,1].set_xlabel("Epoch"); axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
out_dash = os.path.join(OUT_DIR, "training_dashboard.png")
plt.savefig(out_dash, dpi=150)
plt.close()
print(f"Saved: {out_dash}")

print("\nAll plots saved successfully!")
