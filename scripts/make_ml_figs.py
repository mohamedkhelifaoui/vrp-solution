# scripts/make_ml_figs.py
# Generates illustrative ML figures so your LaTeX compiles.
# If you later have real numbers, just replace the arrays below.

import os
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = "data/figures"
os.makedirs(OUTDIR, exist_ok=True)

# 1) Feature importance (horizontal bars)
features = [
    "window_width_mean", "window_width_p25", "nn_distance_mean",
    "coord_var", "service_share", "demand_total",
    "family_RC", "family_R", "horizon_1", "depot_window_len"
]
importances = np.array([0.18, 0.12, 0.14, 0.11, 0.10, 0.08, 0.09, 0.07, 0.06, 0.05])

order = np.argsort(importances)
plt.figure(figsize=(8, 5))
plt.barh(np.array(features)[order], importances[order])
plt.xlabel("Relative importance")
plt.title("Selector feature importance (illustrative)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "ml_feature_importance.png"), dpi=200, bbox_inches="tight")
plt.close()

# 2) Uplift by family (bars for SLA coverage uplift, markers for distance delta)
families = ["C", "R", "RC"]
sla_uplift = np.array([3.0, 8.5, 10.2])   # percentage points (pp)
dist_delta = np.array([-0.4, 0.6, 0.8])   # percent vs fixed baseline
x = np.arange(len(families))

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.bar(x, sla_uplift, width=0.6)
ax.set_xticks(x)
ax.set_xticklabels(families)
ax.set_ylabel("SLA coverage uplift (pp)")
ax.set_title("Selector uplift by family (illustrative)")

ax2 = ax.twinx()
ax2.plot(x, dist_delta, marker="o", linestyle="None")
ax2.set_ylabel("Distance delta (%)")

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "ml_uplift_by_family.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# 3) Confusion matrix (normalized)
methods = ["Q120", "SAA32-b0p5", "Gamma1-q1p645", "DET"]
cm = np.array([
    [28,  3,  2,  1],
    [ 4, 24,  5,  2],
    [ 3,  4, 22,  3],
    [ 2,  2,  3, 16]
], dtype=float)
cm = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6,5))
plt.imshow(cm, aspect="equal")
plt.xticks(np.arange(len(methods)), methods, rotation=0)
plt.yticks(np.arange(len(methods)), methods)
plt.xlabel("Predicted")
plt.ylabel("Best SLA-feasible (label)")
plt.title("Confusion matrix (normalized, illustrative)")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "ml_confusion_matrix.png"), dpi=200, bbox_inches="tight")
plt.close()

# 4) Regret histogram (distance % above best SLA-feasible)
np.random.seed(7)
regret = np.clip(np.random.normal(loc=0.5, scale=0.6, size=120), 0, 3.0)  # percent
plt.figure(figsize=(7,5))
plt.hist(regret, bins=15)
plt.xlabel("Distance regret (%)")
plt.ylabel("Count")
plt.title("Distribution of distance regret (illustrative)")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "ml_regret_hist.png"), dpi=200, bbox_inches="tight")
plt.close()

print("Wrote figures to", OUTDIR)
