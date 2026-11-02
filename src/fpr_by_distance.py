"""False-positive rate vs distance (in messages) to nearest injected message."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROC = Path("data/processed")
t_l = json.load(open("models/lstm_threshold.json"))["threshold"]
t_i = json.load(open("models/iforest_v2_threshold.json"))["threshold"]
bins = [(0, 20), (20, 200), (200, 2000), (2000, 10**9)]   # ~10ms, 100ms, 1s
rows = []
for name in ["DoS", "Fuzzy", "Gear", "RPM"]:
    z = np.load(PROC / f"eval_slice_{name}.npz")
    y, lstm, iff, unk = z["y"], z["lstm"], z["iforest"], z["unknown"]
    att = np.flatnonzero(y == 1)
    pos = np.flatnonzero(y == 0)
    j = np.searchsorted(att, pos)
    left = np.where(j > 0, pos - att[np.clip(j - 1, 0, None)], 10**9)
    right = np.where(j < len(att), att[np.clip(j, None, len(att) - 1)] - pos, 10**9)
    dist = np.minimum(left, right)
    fp_l, fp_i = lstm[pos] > t_l, (iff[pos] > t_i) | unk[pos]
    for lo, hi in bins:
        m = (dist > lo) & (dist <= hi)
        rows.append(dict(attack=name, dist_msgs=f"{lo+1}-{hi if hi < 10**9 else 'inf'}",
                         n=int(m.sum()), lstm_fpr=fp_l[m].mean() if m.any() else np.nan,
                         iforest_hybrid_fpr=fp_i[m].mean() if m.any() else np.nan))
print(pd.DataFrame(rows).round(4).to_string(index=False))
