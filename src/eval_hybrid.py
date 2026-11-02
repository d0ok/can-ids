"""Hybrid IDS: ID-whitelist rule OR Isolation Forest v2. Per-attack evaluation."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

PROC, MODELS, REPORTS = Path("data/processed"), Path("models"), Path("reports")
N, CAP, CALIB_Q = 100, 100.0, 0.995
COLS = ["can_id", "dlc", "id_freq", "dt_id_rel"]

norm = pd.read_parquet(PROC / "features_Normal.parquet", columns=COLS + ["split"])
tr = norm[norm.split == "train"]
share = tr.can_id.value_counts(normalize=True)
dlc_mode = tr.groupby("can_id").dlc.agg(lambda x: x.mode().iloc[0])
whitelist = set(share.index.tolist())
print(f"Whitelist: {len(whitelist)} IDs from normal training data\n", flush=True)


def make(df):
    known = df.can_id.isin(share.index).to_numpy()
    expected = (N * df.can_id.map(share)).clip(lower=1.0)
    ratio = np.where(known, df.id_freq * N / expected, CAP).clip(max=CAP)
    mism = (df.dlc != df.can_id.map(dlc_mode)).to_numpy() & known
    return pd.DataFrame({"log_freq_ratio": np.log1p(ratio).astype("float32"),
                         "fast_rel": df.dt_id_rel.clip(upper=1.0).astype("float32"),
                         "dlc_mismatch": mism.astype("float32")})


model = joblib.load(MODELS / "iforest_v2.joblib")
cal = make(norm[norm.split == "calib"])
thr = float(np.quantile(-model.score_samples(cal.to_numpy()), CALIB_Q))
(MODELS / "iforest_v2_threshold.json").write_text(json.dumps({"threshold": thr}))
cal_unknown = (~norm[norm.split == "calib"].can_id.isin(whitelist)).mean()
print(f"threshold={thr:.4f} | whitelist FPR on clean calib traffic: "
      f"{cal_unknown:.4%}\n", flush=True)

rows = []
for name in ["DoS", "Fuzzy", "Gear", "RPM"]:
    df = pd.read_parquet(PROC / f"features_{name}.parquet", columns=["can_id", "label"])
    y = df.label.to_numpy()
    s = np.load(PROC / f"scores_if2_{name}.npy")
    unknown = (~df.can_id.isin(whitelist)).to_numpy()
    for mode, pred, sc in [
        ("forest_only", s > thr, s),
        ("rule_only", unknown, unknown.astype(float)),
        ("hybrid", (s > thr) | unknown, s + 10.0 * unknown),
    ]:
        p, r, f1, _ = precision_recall_fscore_support(
            y, pred.astype("uint8"), average="binary", zero_division=0)
        rows.append(dict(attack=name, detector=mode, precision=p, recall=r, f1=f1,
                         roc_auc=roc_auc_score(y, sc), fpr=pred[y == 0].mean()))
    print(pd.DataFrame(rows[-3:]).round(4).to_string(index=False), "\n", flush=True)

res = pd.DataFrame(rows).round(4)
res.to_csv(REPORTS / "results_hybrid.csv", index=False)
print(res[res.detector == "hybrid"].to_string(index=False))
