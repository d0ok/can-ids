"""Isolation Forest on normal traffic only; per-attack-type evaluation."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from features import FEATURES

PROC, MODELS, REPORTS = Path("data/processed"), Path("models"), Path("reports")
REPORTS.mkdir(exist_ok=True)
CALIB_QUANTILE = 0.995     # target ~0.5% FPR on clean held-out normal traffic


def score(model, X, chunk=500_000):
    """Higher = more anomalous."""
    out = np.empty(len(X), dtype=np.float32)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = -model.score_samples(X.iloc[i:i + chunk].to_numpy())
    return out


norm = pd.read_parquet(PROC / "features_Normal.parquet",
                       columns=FEATURES + ["split"])
train = norm[norm.split == "train"][FEATURES]
calib = norm[norm.split == "calib"][FEATURES]

model = IsolationForest(n_estimators=200, max_samples=256, contamination="auto",
                        random_state=42, n_jobs=2)
model.fit(train.sample(200_000, random_state=42).to_numpy())
joblib.dump(model, MODELS / "iforest.joblib")

thr = float(np.quantile(score(model, calib), CALIB_QUANTILE))
(MODELS / "iforest_threshold.json").write_text(json.dumps({"threshold": thr}))
print(f"Trained on normal only | threshold={thr:.4f} "
      f"(q={CALIB_QUANTILE} of calib scores)\n", flush=True)

rows = []
for name in ["DoS", "Fuzzy", "Gear", "RPM"]:
    df = pd.read_parquet(PROC / f"features_{name}.parquet",
                         columns=FEATURES + ["label"])
    y = df.label.to_numpy()
    s = score(model, df[FEATURES])
    np.save(PROC / f"scores_if_{name}.npy", s)
    pred = (s > thr).astype("uint8")
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    rows.append(dict(attack=name, precision=p, recall=r, f1=f1,
                     roc_auc=roc_auc_score(y, s), fpr=pred[y == 0].mean(),
                     n_attack=int(y.sum()), n_normal=int((y == 0).sum())))
    print(rows[-1], flush=True)
    del df, s

res = pd.DataFrame(rows).round(4)
res.to_csv(REPORTS / "results_iforest.csv", index=False)
print("\n", res.to_string(index=False))
