"""Figures + consolidated results for the report."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

PROC, REP = Path("data/processed"), Path("reports")
FIG = REP / "figures"; FIG.mkdir(exist_ok=True)
ATT = ["DoS", "Fuzzy", "Gear", "RPM"]
t_l = json.load(open("models/lstm_threshold.json"))["threshold"]
t_i = json.load(open("models/iforest_v2_threshold.json"))["threshold"]
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3})

# ---- 1. F1 comparison across detector versions
v1 = pd.read_csv(REP / "results_iforest.csv").set_index("attack").f1
hy = pd.read_csv(REP / "results_hybrid.csv")
v2 = hy[hy.detector == "forest_only"].set_index("attack").f1
hyb = hy[hy.detector == "hybrid"].set_index("attack").f1
ls = pd.read_csv(REP / "results_lstm.csv")
lstm = ls[ls.detector == "lstm+rule"].set_index("attack").f1
f1 = pd.DataFrame({"IF v1 (naive)": v1, "IF v2 (directional)": v2,
                   "IF v2 + ID rule": hyb, "LSTM + ID rule*": lstm}).loc[ATT]
ax = f1.plot.bar(figsize=(9, 4.5), rot=0, width=.8)
ax.set_ylabel("F1 (message level)"); ax.set_ylim(0, 1.05)
ax.set_title("Per-attack F1 by detector version")
ax.legend(loc="lower left", fontsize=8)
plt.figtext(.01, .005, "*LSTM evaluated on a 1M-row slice per file; others on full files.",
            fontsize=7)
plt.tight_layout(); plt.savefig(FIG / "f1_by_detector.png"); plt.close()
f1.round(4).to_csv(REP / "f1_by_detector.csv")

# ---- slice-based analyses
fig_r, axr = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
fig_c, axc = plt.subplots(1, 4, figsize=(15, 3.4))
dist_rows, quiet_rows = [], []
bins = [(0, 20), (20, 200), (200, 2000), (2000, 10**9)]
for k, name in enumerate(ATT):
    z = np.load(PROC / f"eval_slice_{name}.npz")
    y, lstm_s, if_s, unk = z["y"], z["lstm"], z["iforest"], z["unknown"]
    # ROC
    for lab, sc in [("LSTM + rule", lstm_s + 20 * unk), ("IF + rule", if_s + 10 * unk)]:
        fpr, tpr, _ = roc_curve(y, sc)
        axr[k].plot(fpr, tpr, label=f"{lab} (AUC {roc_auc_score(y, sc):.3f})")
    axr[k].plot([0, 1], [0, 1], "k:", lw=.8)
    axr[k].set_title(name); axr[k].set_xlabel("FPR")
    axr[k].legend(fontsize=7, loc="lower right")
    # hybrid confusion matrix (row-normalised)
    pred = (if_s > t_i) | unk
    cm = np.array([[((y == a) & (pred == b)).sum() for b in (0, 1)] for a in (0, 1)])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    axc[k].imshow(cmn, cmap="Blues", vmin=0, vmax=1); axc[k].grid(False)
    for a in range(2):
        for b in range(2):
            axc[k].text(b, a, f"{cmn[a, b]:.3f}\n({cm[a, b]:,})", ha="center",
                        va="center", fontsize=8, color="black" if cmn[a, b] < .6 else "white")
    axc[k].set_xticks([0, 1], ["pred normal", "pred attack"], fontsize=8)
    axc[k].set_yticks([0, 1], ["normal", "attack"], fontsize=8); axc[k].set_title(name)
    # distance to nearest injection
    att, pos = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    j = np.searchsorted(att, pos)
    left = np.where(j > 0, pos - att[np.clip(j - 1, 0, None)], 10**9)
    right = np.where(j < len(att), att[np.clip(j, None, len(att) - 1)] - pos, 10**9)
    dist = np.minimum(left, right)
    fp_l, fp_i = lstm_s[pos] > t_l, ((if_s > t_i) | unk)[pos]
    for lo, hi in bins:
        m = (dist > lo) & (dist <= hi)
        dist_rows.append(dict(attack=name, bin=f"{lo+1}-{hi if hi < 10**9 else 'inf'}",
                              lstm=fp_l[m].mean(), iforest_hybrid=fp_i[m].mean()))
    q = dist > 200
    quiet_rows.append(dict(attack=name, quiet_share=q.mean(),
                           lstm_quiet_fpr=fp_l[q].mean(), if_hybrid_quiet_fpr=fp_i[q].mean(),
                           if_hybrid_overall_fpr=fp_i.mean(),
                           if_hybrid_burst_fpr=fp_i[~q].mean()))
axr[0].set_ylabel("TPR")
fig_r.suptitle("ROC: LSTM vs Isolation Forest (+ whitelist rule), 1M-row slices")
fig_r.tight_layout(); fig_r.savefig(FIG / "roc_comparison.png"); plt.close(fig_r)
fig_c.suptitle("Hybrid detector (IF v2 + ID rule): row-normalised confusion matrices")
fig_c.tight_layout(); fig_c.savefig(FIG / "confusion_hybrid.png"); plt.close(fig_c)

d = pd.DataFrame(dist_rows)
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
for k, name in enumerate(ATT):
    s = d[d.attack == name].set_index("bin")[["lstm", "iforest_hybrid"]]
    s.plot.bar(ax=axs[k], rot=0, legend=(k == 0)); axs[k].set_title(name)
    axs[k].set_xlabel("messages from nearest injection")
axs[0].set_ylabel("false-positive rate")
fig.suptitle("False-positive rate vs distance to nearest injection")
fig.tight_layout(); fig.savefig(FIG / "fpr_vs_distance.png"); plt.close(fig)

qt = pd.DataFrame(quiet_rows).round(4)
qt.to_csv(REP / "quiet_period_fpr.csv", index=False)
print("Per-attack F1 by detector version:\n", f1.round(4).to_string())
print("\nFPR in quiet periods (>200 msgs from any injection) vs bursts:\n",
      qt.to_string(index=False))
print("\nFigures saved to", FIG)
