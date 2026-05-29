# helper function for the SAM Processing notebook in colab_sam_preprocessing. contains the reusable sam preprocessing workflow

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm


# maps the model name to the official checkpoint download
SAM_CKPT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
# maps the model name to the local checkpoint filename after downloading 
SAM_CKPT_NAMES = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}

# all parameters for one sam preprocessing run
@dataclass
class SamPrepV2Config:
    data_root: str  # extracted dataset root
    train_sam_dir: str  # where training sam files are written
    test_sam_dir: str  # where test sam files are written
    sam_checkpoint_path: str  # local sam checkpoint file
    sam_model_type: str  # which sam backbone to use
    seed: int = 42  # sampling seed for subsets, 42 is fixed everywhere in the project
    max_trainval_images: Optional[int] = None  # opt. caps the training image pool
    max_test_images: Optional[int] = None  # opt. caps the test image pool
    subset_policy: str = "random"  # chooses whether the capped subsets are random or ordered
    part1_stats_csv: Optional[str] = None  # opt. merges per image metadata into train records
    force_regenerate: bool = False  # forces regeneration even when npz files already exist
    points_per_side: int = 24  # mask proposal density for sam -> please see SAM paper for more details 
    pred_iou_thresh: float = 0.86  # filters raw sam masks by predicted iou
    stability_score_thresh: float = 0.92  # filters raw sam masks by stability
    crop_n_layers: int = 0  # this controls multiscale crop refinement in sam
    crop_n_points_downscale_factor: int = 2  # scales point density on deeper crop layers
    min_mask_region_area: int = 400  # drops very small mask regions (400 pixel threshold)
    mask_score_min: float = 0.0  # filters for the combined iou*stability score after generation
    io_threads: int = 2  # threads


# helper to install a package only when the current runtime does not have it
def ensure_package(module_name: str, pip_name: str | None = None) -> None:
    try:
        __import__(module_name)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pip_name or module_name]
        )

# helper that makes sure the chosen sam checkpoint exists on drive and local disk
def ensure_sam_checkpoint(
    model_type: str,
    drive_dir: str | Path,
    local_dir: str | Path = "/content",
) -> tuple[Path, str]:
    if model_type not in SAM_CKPT_URLS:
        raise ValueError(f"Unsupported SAM model type: {model_type}")

    ensure_package("segment_anything", "git+https://github.com/facebookresearch/segment-anything.git")

    drive_dir = Path(drive_dir)
    local_dir = Path(local_dir)
    drive_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = SAM_CKPT_NAMES[model_type]
    drive_ckpt = drive_dir / ckpt_name
    local_ckpt = local_dir / ckpt_name

    if not drive_ckpt.exists():
        print(f"Downloading {ckpt_name} to Drive ...")
        urllib.request.urlretrieve(SAM_CKPT_URLS[model_type], str(drive_ckpt))
        print("Done.")

    if not local_ckpt.exists():
        print(f"Copying SAM checkpoint to local disk ({local_ckpt}) ...")
        shutil.copy2(drive_ckpt, local_ckpt)

    sam_device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("SAM_CHECKPOINT_PATH:", local_ckpt)
    print("SAM_MODEL_TYPE:", model_type, "| SAM device:", sam_device)
    return local_ckpt, sam_device

# helper that resolves the dataset root even when the zip extracts into one nested folder
def resolve_dataset_root(base: Path) -> Path:
    base = Path(base)
    if (base / "train").exists() and (base / "test").exists():
        return base
    for child in [path for path in base.iterdir() if path.is_dir()]:
        if (child / "train").exists() and (child / "test").exists():
            return child
    raise FileNotFoundError(f"Could not find train/test under {base}")

# lhelper that loads all rgb image paths in a stable sorted order
def sorted_rgb_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*_rgb.png"))
    if not files:
        raise RuntimeError(f"No *_rgb.png under {directory}")
    return files

# if chosen inside the config, this optionally reduces the subset of of dataset split to a deterministic subset
def choose_subset(files: list[Path], max_images: Optional[int], seed: int, policy: str) -> list[Path]:
    if max_images is None or max_images >= len(files):
        return list(files)
    if policy == "first":
        return list(files[:max_images])
    if policy != "random":
        raise ValueError(f"subset_policy must be 'random' or 'first', got {policy!r}")
    rng = random.Random(seed)
    chosen = list(files)
    rng.shuffle(chosen)
    return sorted(chosen[:max_images])

# loads one rgb image in the shape sam expects
def load_rgb_uint8(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != (560, 560):
        raise ValueError(f"Expected 560x560, got {rgb.shape[:2]} for {path.name}")
    return rgb

# converts raw sam masks outputs into top one and top two segment assignments
def masks_to_top2_partition(
    masks: list[dict],
    image_hw: tuple[int, int] = (560, 560),
    mask_score_min: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    h, w = image_hw
    s1 = np.zeros((h, w), dtype=np.float32)
    s2 = np.zeros((h, w), dtype=np.float32)
    id1 = np.zeros((h, w), dtype=np.int32)
    id2 = np.zeros((h, w), dtype=np.int32)
    coverage = np.zeros((h, w), dtype=np.int32)
    if not masks:
        return id1, id2, s1, s2, coverage, []

    def mask_score(mask: dict) -> float:
        iou = max(0.0, float(mask.get("predicted_iou", 0.5)))
        stability = max(0.0, float(mask.get("stability_score", 0.5)))
        return iou * stability

    scored = [(mask, mask_score(mask)) for mask in masks]
    scored = [(mask, score) for mask, score in scored if score >= mask_score_min]
    if not scored:
        return id1, id2, s1, s2, coverage, []

    scored.sort(key=lambda item: item[1], reverse=True)
    kept, next_id = [], 1

    for mask, score in scored:
        segmentation = mask["segmentation"].astype(bool)
        if segmentation.shape != (h, w):
            raise ValueError(f"SAM mask shape mismatch: {segmentation.shape}")
        if not segmentation.any():
            continue

        set_top1 = segmentation & (s1 == 0)
        s1[set_top1] = score
        id1[set_top1] = next_id

        set_top2 = segmentation & (s1 > 0) & (s2 == 0) & (id1 != next_id)
        s2[set_top2] = score
        id2[set_top2] = next_id

        coverage[segmentation] += 1
        kept.append(
            {
                "segment_id": next_id,
                "sam_area": int(mask.get("area", int(segmentation.sum()))),
                "sam_predicted_iou": float(mask.get("predicted_iou", np.nan)),
                "sam_stability_score": float(mask.get("stability_score", np.nan)),
                "mask_score": float(score),
                "pixels_claimed_as_top1": int(set_top1.sum()),
            }
        )
        next_id += 1

    return id1, id2, s1, s2, coverage, kept

# converts the two best mask scores taken from SAM into proxies of confidences and pixel uncertainty
def confidences_and_uncertainty(
    s1: np.ndarray,
    s2: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    denom = s1 + s2 + eps
    p1 = s1 / denom
    p2 = s2 / denom
    uncertainty = 1.0 - np.abs(p1 - p2)

    no_cover = s1 == 0
    p1[no_cover] = 0.0
    p2[no_cover] = 0.0
    uncertainty[no_cover] = 1.0
    return p1.astype(np.float32), p2.astype(np.float32), uncertainty.astype(np.float32)

# this derives a boundary map and boundary uncertainty from the top one partition
def boundary_and_boundary_uncertainty(id1: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = id1.shape
    diff_x = id1[:, 1:] != id1[:, :-1]
    diff_y = id1[1:, :] != id1[:-1, :]

    sam_boundary = np.zeros((h, w), dtype=bool)
    sam_boundary[:, 1:] |= diff_x
    sam_boundary[:, :-1] |= diff_x
    sam_boundary[1:, :] |= diff_y
    sam_boundary[:-1, :] |= diff_y

    bu_x = 1.0 - p1[:, 1:] * p1[:, :-1]
    bu_y = 1.0 - p1[1:, :] * p1[:-1, :]

    bx_pad = np.zeros((h, w), dtype=np.float32)
    bx_pad[:, 1:] = np.where(diff_x, bu_x, 0.0)
    bx_pad[:, :-1] = np.maximum(bx_pad[:, :-1], np.where(diff_x, bu_x, 0.0))

    by_pad = np.zeros((h, w), dtype=np.float32)
    by_pad[1:, :] = np.where(diff_y, bu_y, 0.0)
    by_pad[:-1, :] = np.maximum(by_pad[:-1, :], np.where(diff_y, bu_y, 0.0))

    boundary_uncertainty = np.maximum(bx_pad, by_pad)
    boundary_uncertainty[~sam_boundary] = 0.0
    return sam_boundary, boundary_uncertainty.astype(np.float32)

# helper that builds the output npz path for one rgb image
def sam_v2_path(rgb_path: Path, out_dir: Path) -> Path:
    return out_dir / f"{rgb_path.stem}_sam_v2.npz"

# writes a complete sam v2 npz artifact to disk for efficiency later on 
def save_sam_v2(
    out_path: Path,
    id1: np.ndarray,
    id2: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    uncertainty: np.ndarray,
    coverage: np.ndarray,
    sam_boundary: np.ndarray,
    boundary_uncertainty: np.ndarray,
    metadata: dict,
) -> None:
    np.savez_compressed(
        out_path,
        label_top1=id1.astype(np.int32),
        label_top2=id2.astype(np.int32),
        conf_top1=p1.astype(np.float16),
        conf_top2=p2.astype(np.float16),
        pixel_uncertainty=uncertainty.astype(np.float16),
        coverage=np.clip(coverage, 0, 255).astype(np.uint8),
        sam_boundary=sam_boundary.astype(bool),
        boundary_uncertainty=boundary_uncertainty.astype(np.float16),
        metadata=json.dumps(metadata),
    )

# this constructs the sam model and automatic mask generator
def build_mask_generator(cfg: SamPrepV2Config, sam_device: str):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[cfg.sam_model_type](checkpoint=str(cfg.sam_checkpoint_path))
    sam.to(device=sam_device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=cfg.points_per_side,
        pred_iou_thresh=cfg.pred_iou_thresh,
        stability_score_thresh=cfg.stability_score_thresh,
        crop_n_layers=cfg.crop_n_layers,
        crop_n_points_downscale_factor=cfg.crop_n_points_downscale_factor,
        min_mask_region_area=cfg.min_mask_region_area,
    )
    return sam, mask_generator


# this runs sam on one rgb image and prepares everything needed for saving downstream
@torch.inference_mode()
def generate_sam_v2_for_rgb(
    rgb_path: Path,
    out_dir: Path,
    mask_generator,
    cfg: SamPrepV2Config,
    sam_device: str,
) -> tuple[tuple | Path, bool]:
    out_path = sam_v2_path(rgb_path, out_dir)
    if out_path.exists() and not cfg.force_regenerate:
        return out_path, False

    rgb = load_rgb_uint8(rgb_path)
    masks = mask_generator.generate(rgb)
    id1, id2, s1, s2, coverage, kept = masks_to_top2_partition(
        masks,
        image_hw=(560, 560),
        mask_score_min=cfg.mask_score_min,
    )
    p1, p2, uncertainty = confidences_and_uncertainty(s1, s2)
    sam_boundary, boundary_uncertainty = boundary_and_boundary_uncertainty(id1, p1)
    metadata = {
        "rgb_path": str(rgb_path),
        "image_hw": [560, 560],
        "sam_model_type": cfg.sam_model_type,
        "sam_device": sam_device,
        "num_raw_masks": len(masks),
        "num_partition_segments_top1": int(id1.max()),
        "kept_segments": kept,
    }
    return (
        out_path,
        id1,
        id2,
        p1,
        p2,
        uncertainty,
        coverage,
        sam_boundary,
        boundary_uncertainty,
        metadata,
    ), True

# runs sam generation over one split and writes all new files
def ensure_sam_v2_for_files(
    rgb_files: list[Path],
    out_dir: Path,
    mask_generator,
    split: str,
    cfg: SamPrepV2Config,
    sam_device: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    created, skipped, failed = 0, 0, []
    futures = []

    with ThreadPoolExecutor(max_workers=max(1, cfg.io_threads)) as pool:
        for rgb_path in tqdm(rgb_files, desc=f"SAM v2 {split}"):
            try:
                result, was_created = generate_sam_v2_for_rgb(
                    rgb_path=rgb_path,
                    out_dir=out_dir,
                    mask_generator=mask_generator,
                    cfg=cfg,
                    sam_device=sam_device,
                )
                if was_created:
                    futures.append(pool.submit(save_sam_v2, *result))
                    created += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed.append((str(rgb_path), repr(exc)))
        for future in futures:
            future.result()

    print(
        f"{split}: requested={len(rgb_files)} | created={created} | "
        f"skipped={skipped} | failed={len(failed)}"
    )
    return {
        "split": split,
        "n_requested": len(rgb_files),
        "created": created,
        "skipped_existing": skipped,
        "failed": len(failed),
        "failures": failed,
        "sam_dir": str(out_dir),
    }

# helper function that builds a record table for one split
def build_records_v2(
    rgb_files: list[Path],
    sam_dir: Path,
    split_name: str,
    part1_csv: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    for rgb in rgb_files:
        row = {
            "split": split_name,
            "name": rgb.name,
            "rgb_path": str(rgb),
            "sam_npz_path": str(sam_v2_path(rgb, sam_dir)),
            "sam_npz_exists": sam_v2_path(rgb, sam_dir).exists(),
        }
        if split_name == "train":
            depth_path = Path(str(rgb).replace("_rgb.png", "_depth.npy"))
            row["depth_path"] = str(depth_path)
            row["depth_exists"] = depth_path.exists()
        rows.append(row)

    df = pd.DataFrame(rows)
    if split_name == "train":
        df = df[df["depth_exists"]].copy()
    if part1_csv is not None and Path(part1_csv).exists():
        part1 = pd.read_csv(part1_csv)
        if "name" in part1.columns:
            keep_cols = [col for col in part1.columns if col not in {"rgb_path", "depth_path"}]
            df = df.merge(part1[keep_cols], on="name", how="left")
    return df.reset_index(drop=True)


# this runs the full train and test preprocessing pass
def prepare_sam_v2(cfg: SamPrepV2Config, sam_device: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data_root = resolve_dataset_root(Path(cfg.data_root))
    all_train = sorted_rgb_files(data_root / "train")
    all_test = sorted_rgb_files(data_root / "test")

    selected_train = choose_subset(all_train, cfg.max_trainval_images, cfg.seed, cfg.subset_policy)
    selected_test = choose_subset(all_test, cfg.max_test_images, cfg.seed + 1000, cfg.subset_policy)
    print(f"Train: {len(selected_train)} | Test: {len(selected_test)}")

    sam_model, mask_generator = build_mask_generator(cfg, sam_device=sam_device)
    train_summary = ensure_sam_v2_for_files(selected_train, Path(cfg.train_sam_dir), mask_generator, "train", cfg, sam_device)
    test_summary = ensure_sam_v2_for_files(selected_test, Path(cfg.test_sam_dir), mask_generator, "test", cfg, sam_device)

    train_records = build_records_v2(selected_train, Path(cfg.train_sam_dir), "train", cfg.part1_stats_csv)
    test_records = build_records_v2(selected_test, Path(cfg.test_sam_dir), "test", None)

    train_expected = len(train_records)
    test_expected = len(test_records)
    train_actual = len(list(Path(cfg.train_sam_dir).glob("*_sam_v2.npz")))
    test_actual = len(list(Path(cfg.test_sam_dir).glob("*_sam_v2.npz")))

    assert train_records["sam_npz_exists"].all(), "missing sam npz in train records"
    assert test_records["sam_npz_exists"].all(), "missing sam npz in test records"
    assert train_actual == train_expected, (
        f"train sam count mismatch expected {train_expected} got {train_actual}"
    )
    assert test_actual == test_expected, (
        f"test sam count mismatch expected {test_expected} got {test_actual}"
    )

    del sam_model, mask_generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "train_size": len(train_records),
        "test_size": len(test_records),
        "expected_train": train_expected,
        "expected_test": test_expected,
        "train_npz_count_on_disk": train_actual,
        "test_npz_count_on_disk": test_actual,
        "train_summary": train_summary,
        "test_summary": test_summary,
        "train_sam_dir": str(cfg.train_sam_dir),
        "test_sam_dir": str(cfg.test_sam_dir),
    }
    return train_records, test_records, summary


# helper that reads one saved npz file and reports stored arrays
def inspect_sam_npz(npz_path: str | Path) -> dict:
    npz = np.load(npz_path, allow_pickle=True)
    info = {key: getattr(npz[key], "shape", None) for key in npz.files if key != "metadata"}
    info["keys"] = list(npz.files)
    if "metadata" in npz.files:
        try:
            metadata = json.loads(str(npz["metadata"]))
            info["metadata_keys"] = sorted(list(metadata.keys()))
        except Exception as exc:
            info["metadata_parse_error"] = repr(exc)
    return info

# writes the records and summary files for one completed run
def write_sam_run_artifacts(
    cfg: SamPrepV2Config,
    train_records: pd.DataFrame,
    test_records: pd.DataFrame,
    summary: dict,
    output_root: str | Path,
    overwrite: bool = True,
) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_csv = output_root / "train_records.csv"
    test_csv = output_root / "test_records.csv"
    config_json = output_root / "sam_prep_v2_config.json"
    summary_json = output_root / "sam_prep_v2_summary.json"

    train_records.to_csv(train_csv, index=False)
    test_records.to_csv(test_csv, index=False)
    config_json.write_text(json.dumps(asdict(cfg), indent=2, default=str))
    summary_json.write_text(json.dumps(summary, indent=2, default=str))

    train_sam_src = Path(cfg.train_sam_dir)
    test_sam_src = Path(cfg.test_sam_dir)
    train_sam_dst = output_root / "train_sam_v2"
    test_sam_dst = output_root / "test_sam_v2"

    if overwrite and train_sam_dst.exists():
        shutil.rmtree(train_sam_dst)
    if overwrite and test_sam_dst.exists():
        shutil.rmtree(test_sam_dst)
    if not train_sam_dst.exists():
        shutil.copytree(train_sam_src, train_sam_dst)
    if not test_sam_dst.exists():
        shutil.copytree(test_sam_src, test_sam_dst)

    reusable_paths = {
        "train_records_csv": str(train_csv),
        "test_records_csv": str(test_csv),
        "train_sam_dir": str(train_sam_dst),
        "test_sam_dir": str(test_sam_dst),
        "config_json": str(config_json),
        "summary_json": str(summary_json),
    }
    (output_root / "reusable_paths.json").write_text(json.dumps(reusable_paths, indent=2))
    return reusable_paths

# this zips one sam folder with its folder name preserved
def write_sam_zip(source_dir: str | Path, zip_path: str | Path) -> Path:
    source_dir = Path(source_dir)
    zip_path = Path(zip_path)

    if not source_dir.exists():
        raise FileNotFoundError(f"sam source dir not found: {source_dir}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_base = zip_path.with_suffix("")
    archive_path = Path(
        shutil.make_archive(
            base_name=str(tmp_base),
            format="zip",
            root_dir=str(source_dir.parent),
            base_dir=source_dir.name,
        )
    )
    if archive_path != zip_path:
        if zip_path.exists():
            zip_path.unlink()
        shutil.move(str(archive_path), str(zip_path))
    return zip_path

# this runs the full workflow and returns paths and dataframes for the notebook
def run_sam_preprocessing_workflow(
    cfg: SamPrepV2Config,
    local_artifact_root: str | Path,
    sam_device: str,
    drive_export_root: str | Path | None = None,
    train_zip_path: str | Path | None = None,
    test_zip_path: str | Path | None = None,
) -> dict:
    train_records, test_records, summary = prepare_sam_v2(cfg, sam_device=sam_device)

    local_paths = write_sam_run_artifacts(
        cfg=cfg,
        train_records=train_records,
        test_records=test_records,
        summary=summary,
        output_root=local_artifact_root,
        overwrite=False,
    )

    export_paths = None
    if drive_export_root is not None:
        export_paths = write_sam_run_artifacts(
            cfg=cfg,
            train_records=train_records,
            test_records=test_records,
            summary=summary,
            output_root=drive_export_root,
            overwrite=True,
        )

    zip_paths = None
    if train_zip_path is not None or test_zip_path is not None:
        zip_paths = {}
        if train_zip_path is not None:
            zip_paths["train_sam_zip"] = str(write_sam_zip(cfg.train_sam_dir, train_zip_path))
        if test_zip_path is not None:
            zip_paths["test_sam_zip"] = str(write_sam_zip(cfg.test_sam_dir, test_zip_path))

    return {
        "train_records": train_records,
        "test_records": test_records,
        "summary": summary,
        "local_paths": local_paths,
        "drive_paths": export_paths,
        "zip_paths": zip_paths,
    }
