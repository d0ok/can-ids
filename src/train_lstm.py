"""LSTM next-ID predictor (normal-only training); surprise = anomaly score."""
import os, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

PROC, MODELS, REPORTS = Path("data/processed"), Path("models"), Path("reports")
W, TRAIN_N, EPOCHS, CALIB_Q = 20, 400_000, 6, 0.995
EVAL_ROWS = 1_000_000          # contiguous slice per attack file; None = full file
keras.utils.set_random_seed(42)

norm = pd.read_parquet(PROC / "features_Normal.parquet",
                       columns=["can_id", "dlc", "dt", "split"])
tr = norm[norm.split == "train"].reset_index(drop=True)
cal = norm[norm.split == "calib"].reset_index(drop=True)
id2idx = {int(c): i + 1 for i, c in enumerate(sorted(tr.can_id.unique()))}  # 0 = UNK
V = len(id2idx) + 1


def prep(df):
    ids = df.can_id.map(id2idx).fillna(0).to_numpy().astype("int32")
    num = np.stack([np.log1p(df.dt.to_numpy() * 1000) / 7.0,
                    df.dlc.to_numpy() / 8.0], axis=1).astype("float32")
    return ids, num


def windows(ids, num, pos):
    offs = pos[:, None] - W + np.arange(W)          # W previous messages (causal)
    return ids[offs], num[offs], ids[pos]


def nll(model, ids, num, pos, batch=8192):
    out = np.empty(len(pos), dtype="float32")
    for i in range(0, len(pos), batch):
        xi, xn, y = windows(ids, num, pos[i:i + batch])
        p = model.predict_on_batch([xi, xn])
        out[i:i + batch] = -np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1.0))
    return out


i_in = keras.Input((W,), dtype="int32")
n_in = keras.Input((W, 2))
x = keras.layers.Concatenate()([keras.layers.Embedding(V, 8)(i_in), n_in])
x = keras.layers.LSTM(64)(x)
model = keras.Model([i_in, n_in], keras.layers.Dense(V, activation="softmax")(x))
model.compile(keras.optimizers.Adam(2e-3), loss="sparse_categorical_crossentropy")

rng = np.random.default_rng(42)
tids, tnum = prep(tr)
cids, cnum = prep(cal)
tpos = rng.choice(np.arange(W, len(tids)), TRAIN_N, replace=False)
vpos = rng.choice(np.arange(W, len(cids)), 100_000, replace=False)
Xi, Xn, Y = windows(tids, tnum, tpos)
Vi, Vn, VY = windows(cids, cnum, vpos)
model.fit([Xi, Xn], Y, validation_data=([Vi, Vn], VY), epochs=EPOCHS, batch_size=512,
          callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
          verbose=2)
model.save(MODELS / "lstm.keras")

thr = float(np.quantile(nll(model, cids, cnum, np.arange(W, len(cids))), CALIB_Q))
thr_if = json.load(open(MODELS / "iforest_v2_threshold.json"))["threshold"]
(MODELS / "lstm_threshold.json").write_text(json.dumps({"threshold": thr}))
print(f"\nLSTM threshold={thr:.4f}\n", flush=True)


def metrics(attack, det, y, pred, sc):
    p, r, f1, _ = precision_recall_fscore_support(y, pred.astype("uint8"),
                                                  average="binary", zero_division=0)
    return dict(attack=attack, detector=det, precision=p, recall=r, f1=f1,
                roc_auc=roc_auc_score(y, sc), fpr=pred[y == 0].mean())


rows = []
for name in ["DoS", "Fuzzy", "Gear", "RPM"]:
    df = pd.read_parquet(PROC / f"features_{name}.parquet",
                         columns=["can_id", "dlc", "dt", "label"])
    n = len(df)
    start = 0 if EVAL_ROWS is None else (n - EVAL_ROWS) // 2
    end = n if EVAL_ROWS is None else start + EVAL_ROWS
    d = df.iloc[start:end].reset_index(drop=True)
    ids, num = prep(d)
    s = nll(model, ids, num, np.arange(W, len(d)))
    y = d.label.to_numpy()[W:]
    unknown = ids[W:] == 0
    s_if = np.load(PROC / f"scores_if2_{name}.npy")[start:end][W:]
    print(f"{name}: slice rows={len(y):,}, attack share={y.mean():.1%}", flush=True)
    rows += [
        metrics(name, "lstm", y, s > thr, s),
        metrics(name, "lstm+rule", y, (s > thr) | unknown, s + 20.0 * unknown),
        metrics(name, "iforest_hybrid", y, (s_if > thr_if) | unknown, s_if + 10.0 * unknown),
    ]
    np.savez_compressed(PROC / f"eval_slice_{name}.npz",
                        y=y, lstm=s, iforest=s_if, unknown=unknown)
    print(pd.DataFrame(rows[-3:]).round(4).to_string(index=False), "\n", flush=True)

res = pd.DataFrame(rows).round(4)
res.to_csv(REPORTS / "results_lstm.csv", index=False)
print(res.to_string(index=False))
