"""Isolation Forest v2: directional per-ID features, normal-only training."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

PROC, MODELS, REPORTS = Path("data/processed"), Path("models"), Path("reports")
N, CAP, CALIB_Q = 100, 100.0, 0.995
COLS = ["can_id", "dlc", "id_freq", "dt_id_rel"]
F2 = ["log_freq_ratio", "fast_rel", "dlc_mismatch"]

norm = pd.read_parquet(PROC / "features_Normal.parquet", columns=COLS + ["split"])
tr = norm[norm.split == "train"]
share = tr.can_id.value_counts(normalize=True)                 # per-ID traffic share
dlc_mode = tr.groupby("can_id").dlc.agg(lambda x: x.mode().iloc[0])


def make(df):
    known = df.can_id.isin(share.index).to_numpy()
    expected = (N * df.can_id.map(share)).clip(lower=1.0)      # NaN for unknown IDs
    ratio = np.where(known, df.id_freq * N / expected, CAP).clip(max=CAP)
    mism = (df.dlc != df.can_id.map(dlc_mode)).to_numpy() & known
    return pd.DataFrame({"log_freq_ratio": np.log1p(ratio).astype("float32"),
                         "fast_rel": df.dt_id_rel.clip(upper=1.0).astype("float32"),
                         "dlc_mismatch": mism.astype("float32")})


def score(model, X, chunk=500_000):
    out = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = -model.score_samples(X.iloc[i:i + chunk].to_numpy())
    return out


Xtr, Xcal = make(tr), make(norm[norm.split == "calib"])
model = IsolationForest(n_estimators=200, max_samples=256, random_state=42, n_jobs=2)
model.fit(Xtr.sample(200_000, random_state=42).to_numpy())
joblib.dump(model, MODELS / "iforest_v2.joblib")
thr = float(np.quantile(score(model, Xcal), CALIB_Q))
print(f"threshold={thr:.4f} | train feature means:\n{Xtr.mean().round(3).to_string()}\n",
      flush=True)

rows = []
for name in ["DoS", "Fuzzy", "Gear", "RPM"]:
    df = pd.read_parquet(PROC / f"features_{name}.parquet", columns=COLS + ["label"])
    y, X = df.label.to_numpy(), make(df)
    s = score(model, X)
    np.save(PROC / f"scores_if2_{name}.npy", s)
    pred = (s > thr).astype("uint8")
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary",
                                                  zero_division=0)
    rows.append(dict(attack=name, precision=p, recall=r, f1=f1,
                     roc_auc=roc_auc_score(y, s), fpr=pred[y == 0].mean()))
    print(rows[-1], flush=True)
res = pd.DataFrame(rows).round(4)
res.to_csv(REPORTS / "results_iforest_v2.csv", index=False)
print("\n", res.to_string(index=False))
