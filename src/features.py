"""Causal feature engineering for CAN streams (inter-arrival, ID freq, DLC stats)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROC = Path("data/processed")
MODELS = Path("models"); MODELS.mkdir(exist_ok=True)
N = 100            # window size in messages
DT_CAP = 1.0       # seconds: caps capture gaps / first-seen-ID intervals
TRAIN_FRAC = 0.7   # chronological split of the NORMAL run
FEATURES = ["dt", "dt_id", "dt_id_rel", "id_known", "dlc", "id_freq",
            "win_dt_mean", "win_dt_std", "win_dlc_mean", "win_dlc_std"]


def window_id_count(can_id, n):
    """Messages with the same ID among the last n messages (incl. current)."""
    m = len(can_id)
    order = np.argsort(can_id, kind="stable")      # positions ascending per ID
    sid = can_id[order]
    bounds = np.flatnonzero(np.diff(sid)) + 1
    starts, ends = np.r_[0, bounds], np.r_[bounds, m]
    out = np.empty(m, dtype=np.float32)
    for s, e in zip(starts, ends):
        pos = order[s:e]
        k = np.arange(1, len(pos) + 1)
        out[pos] = k - np.searchsorted(pos, pos - n, side="right")
    return out


def fit_period(df):
    """Median inter-arrival time per CAN ID (from normal training data)."""
    dt = df.ts.groupby(df.can_id).diff()
    per = dt.dropna().groupby(df.can_id[dt.notna()]).median().clip(lower=1e-4)
    return {int(k): float(v) for k, v in per.items()}


def build_features(df, period):
    ts, cid = df.ts.to_numpy(), df.can_id.to_numpy()
    raw_dt = np.diff(ts, prepend=ts[0])
    dt = raw_dt.clip(0, DT_CAP)
    dt_id = (pd.Series(ts).groupby(cid).diff()
             .fillna(DT_CAP).clip(0, DT_CAP).to_numpy())
    per = pd.Series(cid).map(period)
    known = per.notna().to_numpy()
    rel = (dt_id / per.to_numpy()).clip(0, 100)
    rel = np.where(known, rel, 1.0)
    dlc = df.dlc.to_numpy().astype(np.float32)

    f = pd.DataFrame({
        "dt": dt, "dt_id": dt_id, "dt_id_rel": rel,
        "id_known": known.astype("uint8"), "dlc": dlc,
        "id_freq": window_id_count(cid, N) / N,
    })
    r = pd.Series(dt).rolling(N, min_periods=1)
    f["win_dt_mean"], f["win_dt_std"] = r.mean(), r.std().fillna(0)
    d = pd.Series(dlc).rolling(N, min_periods=1)
    f["win_dlc_mean"], f["win_dlc_std"] = d.mean(), d.std().fillna(0)
    f = f.astype({c: "float32" for c in FEATURES if c != "id_known"})
    f["ts"], f["can_id"], f["label"] = ts, cid, df.label.to_numpy()
    return f, int((raw_dt > DT_CAP).sum())


if __name__ == "__main__":
    cols = ["ts", "can_id", "dlc", "label"]
    normal = pd.read_parquet(PROC / "Normal.parquet", columns=cols)
    cut = int(len(normal) * TRAIN_FRAC)
    period = fit_period(normal.iloc[:cut])
    (MODELS / "id_period.json").write_text(json.dumps(period))
    print(f"Fitted periods for {len(period)} IDs on {cut:,} training rows\n")

    for name in ["Normal", "DoS", "Fuzzy", "Gear", "RPM"]:
        df = normal if name == "Normal" else pd.read_parquet(
            PROC / f"{name}.parquet", columns=cols)
        f, gaps = build_features(df, period)
        if name == "Normal":
            f["split"] = np.where(np.arange(len(f)) < cut, "train", "calib")
        f.to_parquet(PROC / f"features_{name}.parquet")
        print(f"=== {name}: {len(f):,} rows | capture gaps >{DT_CAP}s: {gaps}")
        print(f.groupby("label")[["dt_id", "dt_id_rel", "id_known", "id_freq"]]
              .mean().round(4).to_string(), "\n", flush=True)
        del df, f
