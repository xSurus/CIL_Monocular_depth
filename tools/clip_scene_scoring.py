import shutil
from pathlib import Path

import pandas as pd

from tools.classify_aerial import classify as _classify

AERIAL_POS = [
    "an aerial photograph taken from above",
    "a drone photo looking straight down",
    "a bird's eye view photo",
    "a top-down aerial image",
]
AERIAL_NEG = [
    "a photo taken from ground level",
    "a street level photograph",
    "a photo taken by a person standing",
    "an eye-level photograph",
]
WATER_POS = [
    "a photo containing a lake",
    "a photo of a river",
    "an image with ocean",
    "a photo with a body of water",
]
WATER_NEG = [
    "a photo with no water",
    "a dry landscape",
    "an indoor photo",
]
VEG_POS = [
    "a photo of a forest with trees",
    "a woodland or park with dense vegetation",
    "a landscape with trees bushes or grass",
    "a green scene with plants and foliage",
]
VEG_NEG = [
    "a photo with no trees or plants",
    "an urban scene with buildings and roads",
    "a photo of water with no vegetation",
    "a bare or paved surface",
]


def _run_clip_classifier(data_dir, out_csv, pos_label, pos_prompts, neg_prompts):
    if out_csv.exists():
        return
    _classify(str(data_dir), str(out_csv), pos_label, pos_prompts, neg_prompts)


def _merge_categories(aerial_csv, water_csv, veg_csv, out_csv):
    if out_csv.exists():
        return
    aerial = pd.read_csv(aerial_csv)[['file', 'label']].rename(columns={'label': 'aerial_label'})
    water  = pd.read_csv(water_csv)[['file', 'label']].rename(columns={'label': 'water_label'})
    veg    = pd.read_csv(veg_csv)[['file', 'label']].rename(columns={'label': 'vegetation_label'})
    df = aerial.merge(water, on='file').merge(veg, on='file')

    def _assign(row):
        if row['aerial_label'] == 'aerial':         return 'aerial'
        if row['water_label'] == 'water':           return 'water'
        if row['vegetation_label'] == 'vegetation': return 'vegetation'
        return 'other'

    df['category'] = df.apply(_assign, axis=1)
    df[['file', 'category']].to_csv(out_csv, index=False)

    counts = df['category'].value_counts()
    total  = len(df)
    for cat, n in counts.items():
        print(f'  {cat:<12} {n:>6}  ({100*n/total:.1f}%)')
    print(f'  {"TOTAL":<12} {total:>6}')


def run_clip_scene_scoring(
    train_dir: str,
    test_dir: str,
    clip_local: str,
    clip_drive: str,
    force_recompute_veg: bool = False,
) -> None:
    """Restore, compute, and preview CLIP-based scene scoring files."""
    train_dir_p  = Path(train_dir)
    test_dir_p   = Path(test_dir)
    clip_local_p = Path(clip_local)
    clip_drive_p = Path(clip_drive)
    clip_local_p.mkdir(parents=True, exist_ok=True)
    clip_drive_p.mkdir(parents=True, exist_ok=True)

    for src in clip_drive_p.iterdir():
        dst = clip_local_p / src.name
        if not dst.exists():
            shutil.copy(src, dst)
            print(f"Restored from Drive: {src.name}")

    if force_recompute_veg:
        for fname in [
            "train_vegetation.csv",
            "test_vegetation.csv",
            "train_categories.csv",
            "test_categories.csv",
        ]:
            for root in (clip_local_p, clip_drive_p):
                p = root / fname
                if p.exists():
                    p.unlink()
                    print(f"Removed cached: {fname} from {root}")

    categories = {
        "aerial":     (AERIAL_POS, AERIAL_NEG),
        "water":      (WATER_POS,  WATER_NEG),
        "vegetation": (VEG_POS,    VEG_NEG),
    }
    splits = [("train", train_dir_p), ("test", test_dir_p)]

    for label, (pos, neg) in categories.items():
        for split, data_dir_p in splits:
            out_csv = clip_local_p / f"{split}_{label}.csv"
            _run_clip_classifier(data_dir_p, out_csv, label, pos, neg)
            shutil.copy(out_csv, clip_drive_p / out_csv.name)
            print(f"  {label} ({split}) done.")

    for split, _ in splits:
        out_csv = clip_local_p / f"{split}_categories.csv"
        _merge_categories(
            clip_local_p / f"{split}_aerial.csv",
            clip_local_p / f"{split}_water.csv",
            clip_local_p / f"{split}_vegetation.csv",
            out_csv,
        )
        shutil.copy(out_csv, clip_drive_p / out_csv.name)

    print()
    for tag in ["aerial", "water", "vegetation"]:
        for split in ["train", "test"]:
            df  = pd.read_csv(clip_local_p / f"{split}_{tag}.csv")
            pos = (df["label"] == tag).sum()
            print(f"  {split:5s} {tag:11s}: {pos:5d}/{len(df)} ({100*pos/len(df):.1f}%)")

    print("\nMulti-class categories:")
    for split in ["train", "test"]:
        df     = pd.read_csv(clip_local_p / f"{split}_categories.csv")
        counts = df["category"].value_counts()
        total  = len(df)
        print(f"  {split}:")
        for cat, count in counts.items():
            print(f"    {cat:<12} {count:>6}  ({100*count/total:.1f}%)")
