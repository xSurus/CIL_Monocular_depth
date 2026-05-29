from pathlib import Path
import numpy as np
import pandas as pd
import base64
import zlib
import argparse
import re

def encode_depth(depth: np.ndarray) -> str:
    depth = np.asarray(depth, dtype=np.float16)
    compressed = zlib.compress(depth.tobytes(), level=9)
    encoded = base64.b64encode(compressed).decode("utf-8")
    return encoded

p = argparse.ArgumentParser()
p.add_argument('--pred-dir', required=True)
p.add_argument('--out', required=True)
args = p.parse_args()

pred_dir = Path(args.pred_dir)
out_csv = Path(args.out)

rows = []
pred_files = sorted(
    path for path in pred_dir.glob("test_*.npy")
    if re.fullmatch(r"test_\d+\.npy", path.name)
)

if not pred_files:
    raise RuntimeError(f"No depth prediction files found in {pred_dir}")

for pred_path in pred_files:
    depth = np.load(pred_path)

    idx = pred_path.stem.split("_")[-1]
    img_id = f"test_{idx}_depth"

    encoded_depth = encode_depth(depth)

    rows.append({
        "id": img_id,
        "Depths": encoded_depth,
    })

df = pd.DataFrame(rows, columns=["id", "Depths"])
if df["id"].duplicated().any():
    dupes = df[df["id"].duplicated(keep=False)]["id"].tolist()
    raise RuntimeError(f"Duplicate submission IDs found: {dupes[:10]}")
df.to_csv(out_csv, index=False)

print(f"Saved submission to {out_csv}")
print(f"Number of predictions: {len(df)}")
