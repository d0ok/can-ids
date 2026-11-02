"""Threshold-free comparison + context-contamination test for the LSTM."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROC = Path("data/processed")
W = 20
thr = json.load(open("models/lstm_threshold.json"))["threshold"]
rows, ctx = [], []
for name in ["DoS", "Fuzzy", "Gear", "RPM"]:
    z = np.load(PROC / f"eval_slice_{name}.npz")
    y, lstm, iff = z["y"], z["lstm"], z["iforest"]
    # (1) recall at fixed FPR (threshold set on the labeled-normal rows themselves,
    #     so this is an ROC-curve comparison, not a deployable threshold)
    for fpr in [0.005, 0.01, 0.05]:
        row = dict(attack=name, fpr_target=fpr)
        for det, s in [("lstm", lstm), ("iforest", iff)]:
            t = np.quantile(s[y == 0], 1 - fpr)
            row[f"{det}_recall"] = (s[y == 1] > t).mean()
        rows.append(row)
    # (2) LSTM false-positive rate: clean context vs attack-contaminated context
    cs = np.r_[0, np.cumsum(y.astype(np.int64))]
    idx = np.arange(W, len(y))
    contaminated = (cs[idx] - cs[idx - W]) > 0
    fp = lstm[idx] > thr
    norm_ = y[idx] == 0
    ctx.append(dict(attack=name,
                    fpr_clean_ctx=fp[norm_ & ~contaminated].mean(),
                    fpr_contaminated_ctx=fp[norm_ & contaminated].mean(),
                    share_normal_contaminated=(contaminated[norm_]).mean()))
print("Recall at matched FPR (higher is better):")
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("\nLSTM false-positive rate by context:")
print(pd.DataFrame(ctx).round(4).to_string(index=False))
