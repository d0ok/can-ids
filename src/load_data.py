"""Load the HCRL Car-Hacking dataset into compact parquet files."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

DATA_COLS = [f"D{i}" for i in range(8)]
ATTACK_FILES = {
    "DoS": "DoS_dataset.csv",
    "Fuzzy": "Fuzzy_dataset.csv",
    "Gear": "gear_dataset.csv",
    "RPM": "RPM_dataset.csv",
}
HEX = {f"{i:02x}": i for i in range(256)}        # data-byte lookup
HEX16 = {f"{i:04x}": i for i in range(65536)}    # CAN ID lookup


def load_normal_txt(path):
    """Normal run: 'Timestamp: .. ID: .. 000 DLC: 8 b0 b1 ... b7'."""
    s = pd.Series(Path(path).read_text().splitlines(), dtype="string")
    pat = r"Timestamp:\s*([\d.]+)\s+ID:\s*(\w+)\s+\S+\s+DLC:\s*(\d)\s*(.*)"
    ex = s.str.extract(pat).dropna()
    out = pd.DataFrame({
        "ts": ex[0].astype("float64"),
        "can_id": ex[1].map(lambda x: int(x, 16)).astype("uint16"),
        "dlc": ex[2].astype("uint8"),
    })
    bytes_ = ex[3].str.split(expand=True).reindex(columns=range(8))
    for i, c in enumerate(DATA_COLS):
        out[c] = bytes_[i].map(HEX).fillna(0).astype("uint8")
    out["label"] = np.uint8(0)
    return out


def convert_attack_csv(path, out_path, chunksize=200_000):
    """Stream CSV -> parquet in small chunks; memory stays flat."""
    names = ["ts", "can_id", "dlc"] + DATA_COLS + ["flag"]
    tmp = Path(str(out_path) + ".tmp")
    writer, total, inj = None, 0, 0
    for i, ch in enumerate(pd.read_csv(path, header=None, names=names,
                                       dtype=str, chunksize=chunksize)):
        # 'T' sits in the flag column, or in a data column for DLC<8 rows
        label = (ch[DATA_COLS + ["flag"]] == "T").any(axis=1).astype("uint8")
        out = pd.DataFrame({
            "ts": ch["ts"].astype("float64"),
            "can_id": ch["can_id"].map(HEX16).astype("uint16"),
            "dlc": ch["dlc"].astype("uint8"),
        })
        for c in DATA_COLS:   # 'R'/'T'/NaN are not in HEX -> NaN -> 0 (padding)
            out[c] = ch[c].map(HEX).fillna(0).astype("uint8")
        out["label"] = label
        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema)
        writer.write_table(table)
        total += len(out)
        inj += int(label.sum())
        del ch, out, table
        if i % 5 == 0:
            print(f"    {total:,} rows...", flush=True)
    writer.close()
    tmp.rename(out_path)
    return total, inj


if __name__ == "__main__":
    if not (OUT / "Normal.parquet").exists():
        normal = load_normal_txt(RAW / "normal_run_data.txt")
        normal.to_parquet(OUT / "Normal.parquet")
        print(f"Normal: {len(normal):,} rows", flush=True)
    for name, fn in ATTACK_FILES.items():
        out_path = OUT / f"{name}.parquet"
        if out_path.exists():
            print(f"{name}: already done, skipping", flush=True)
            continue
        print(f"{name}: converting...", flush=True)
        total, inj = convert_attack_csv(RAW / fn, out_path)
        print(f"{name}: {total:,} rows | injected: {inj:,} ({inj/total:.1%})",
              flush=True)
