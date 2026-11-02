import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from features import FEATURES
P = "data/processed/"
thr = json.load(open("models/iforest_threshold.json"))["threshold"]
for name in ["DoS", "Gear"]:
    df = pd.read_parquet(P + f"features_{name}.parquet", columns=FEATURES + ["label"])
    s = np.load(P + f"scores_if_{name}.npy")
    y = df.label.to_numpy()
    idx = np.random.default_rng(0).choice(len(df), 600_000, replace=False)
    print(f"\n=== {name} ===")
    print("single-feature AUC (0.5 = useless, 1.0 or 0.0 = strong):")
    for f in FEATURES:
        a = roc_auc_score(y[idx], df[f].to_numpy()[idx]) if df[f].nunique() > 1 else float("nan")
        print(f"  {f:13s} {a:.3f}")
    print(f"mean score  attack={s[y==1].mean():.4f}  normal={s[y==0].mean():.4f}  threshold={thr:.4f}")
    fl = s > thr
    print(f"flagged rows: {fl.sum():,} | of which truly attack: {(y[fl]==1).mean():.2%}")
    print("mean features of flagged rows vs injected rows:")
    print(pd.DataFrame({"flagged": df.loc[fl, FEATURES].mean(),
                        "injected": df.loc[y == 1, FEATURES].mean()}).round(4).to_string())
