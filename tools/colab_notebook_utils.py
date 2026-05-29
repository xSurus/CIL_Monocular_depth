from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_str = str(REPO_ROOT)
    if env.get("PYTHONPATH"):
        if repo_str not in env["PYTHONPATH"].split(":"):
            env["PYTHONPATH"] = f"{repo_str}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = repo_str
    return env


def _run_python(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], check=True, cwd=str(REPO_ROOT), env=_repo_env())


def _load_cfg(config_path: str) -> dict:
    return yaml.safe_load(_resolve_repo_path(config_path).read_text())


def _run_name_from_best_path(best_path: Path) -> str:
    return best_path.name.removesuffix("_best.pt")


def _resolve_saved_run(base_name: str, require_epoch: int | None = None) -> tuple[str, Path, Path | None]:
    checkpoints_dir = REPO_ROOT / "checkpoints"

    latest_alias = checkpoints_dir / f"{base_name}_latest.txt"
    if latest_alias.exists():
        resolved_name = latest_alias.read_text().strip()
        if resolved_name:
            best_path = checkpoints_dir / f"{resolved_name}_best.pt"
            epoch_path = checkpoints_dir / f"{resolved_name}_epoch{require_epoch}.pt" if require_epoch is not None else None
            if best_path.exists() and (epoch_path is None or epoch_path.exists()):
                if resolved_name != base_name:
                    print(f"Resolved saved run: {base_name} -> {resolved_name}")
                return resolved_name, best_path, epoch_path

    pattern = re.compile(rf"^{re.escape(base_name)}(?:_v\d+)?_best\.pt$")
    candidates = []
    for best_path in checkpoints_dir.glob("*_best.pt"):
        if not pattern.match(best_path.name):
            continue
        run_name = _run_name_from_best_path(best_path)
        epoch_path = checkpoints_dir / f"{run_name}_epoch{require_epoch}.pt" if require_epoch is not None else None
        if require_epoch is not None and not epoch_path.exists():
            continue
        mtime = best_path.stat().st_mtime if epoch_path is None else max(best_path.stat().st_mtime, epoch_path.stat().st_mtime)
        candidates.append((mtime, run_name, best_path, epoch_path))

    if not candidates:
        suffix = f" with epoch{require_epoch}" if require_epoch is not None else ""
        raise FileNotFoundError(
            f"No saved checkpoint run found for config name '{base_name}'{suffix}. "
            f"Looked in {checkpoints_dir} for {base_name}[_vN]_best.pt."
        )

    _, run_name, best_path, epoch_path = max(candidates, key=lambda x: x[0])
    if run_name != base_name:
        print(f"Resolved saved run: {base_name} -> {run_name}")
    return run_name, best_path, epoch_path


def resolve_base_run(config_path: str, base_checkpoint: str | None = None) -> tuple[str, Path]:
    cfg = _load_cfg(config_path)
    if base_checkpoint:
        weights_path = _resolve_repo_path(base_checkpoint)
        if not weights_path.exists():
            raise FileNotFoundError(f"BASE_CHECKPOINT not found: {weights_path}")
        run_name = _run_name_from_best_path(weights_path) if weights_path.name.endswith("_best.pt") else cfg["name"]
        return run_name, weights_path

    run_name, best_path, _ = _resolve_saved_run(cfg["name"])
    return run_name, best_path


def resolve_swap_epoch(config_path: str) -> int:
    cfg = _load_cfg(config_path)
    if cfg.get("save_epoch_checkpoint") is not None:
        return int(cfg["save_epoch_checkpoint"])
    raise ValueError(
        "Could not derive vegetation swap epoch from config. "
        "Add save_epoch_checkpoint: N."
    )


def resolve_swap_dirs(config_path: str, base: str, vggt_pred_dir: str) -> tuple[str | None, str | None]:
    cfg  = _load_cfg(config_path)
    swap = cfg.get('swap')
    if swap == 'vggt':
        return vggt_pred_dir, f'{base}_vggt_swap'
    if swap == 'epoch':
        epoch = resolve_swap_epoch(config_path)
        return f'{base}_epoch{epoch}', f'{base}_epoch{epoch}_veg_swap'
    if swap == 'refinement':
        return f'{base}_refined', f'{base}_refinement_swap'
    return None, None


def resolve_epoch_run(config_path: str, swap_epoch: int | None = None) -> tuple[str, Path, Path]:
    cfg = _load_cfg(config_path)
    swap_epoch = resolve_swap_epoch(config_path) if swap_epoch is None else int(swap_epoch)
    run_name, best_path, epoch_path = _resolve_saved_run(cfg["name"], require_epoch=swap_epoch)
    assert epoch_path is not None
    return run_name, best_path, epoch_path


def _prediction_dir_has_npy(pred_dir: Path) -> bool:
    if not pred_dir.exists():
        return False
    return any(re.fullmatch(r"test_\d+\.npy", path.name) for path in pred_dir.glob("test_*.npy"))


def _count_rgb_files(root_dir: str | Path) -> int:
    root_path = Path(root_dir)
    if not root_path.is_dir():
        return 0
    return sum(1 for path in root_path.iterdir() if path.name.endswith("_rgb.png"))


def _flatten_nested_rgb_dir(root_dir: str | Path) -> None:
    root_path = Path(root_dir)
    if _count_rgb_files(root_path) > 0:
        return
    child_dirs = [
        path for path in root_path.iterdir()
        if path.is_dir() and path.name not in {"_MACOSX", "__MACOSX"} and not path.name.startswith(".")
    ]
    rgb_children = [path for path in child_dirs if _count_rgb_files(path) > 0]
    if len(rgb_children) == 1:
        nested = rgb_children[0]
        print(f"Flattening nested dataset dir: {nested} -> {root_path}")
        for path in nested.iterdir():
            shutil.move(str(path), str(root_path / path.name))
        shutil.rmtree(nested)


def pull_and_extract_drive_archive(
    drive_zip: str,
    local_dir: str,
    check_path: str,
    expect_rgb_files: bool = False,
) -> Path:
    drive_zip_path = Path(drive_zip)
    local_dir_path = Path(local_dir)
    check_path_path = Path(check_path)

    if not drive_zip_path.exists():
        raise FileNotFoundError(f"Drive archive not found: {drive_zip_path}")

    if check_path_path.exists():
        if expect_rgb_files:
            _flatten_nested_rgb_dir(check_path_path)
            n_rgb = _count_rgb_files(check_path_path)
            if n_rgb > 0:
                print(f"Already extracted: {check_path_path} ({n_rgb} RGB files)")
                return check_path_path
            print(f"Existing path has no RGB files, re-extracting: {check_path_path}")
            if check_path_path.is_dir():
                shutil.rmtree(check_path_path)
            else:
                check_path_path.unlink()
        else:
            print(f"Already extracted: {check_path_path}")
            return check_path_path

    local_dir_path.parent.mkdir(parents=True, exist_ok=True)
    local_zip = Path("/content") / drive_zip_path.name
    print(f"Copying {drive_zip_path.name} to local SSD...")
    shutil.copy2(drive_zip_path, local_zip)
    print("Extracting...")
    subprocess.run(["unzip", "-q", str(local_zip), "-d", str(local_dir_path)], check=True)
    local_zip.unlink()

    if expect_rgb_files:
        _flatten_nested_rgb_dir(check_path_path)
        n_rgb = _count_rgb_files(check_path_path)
        if n_rgb == 0:
            raise RuntimeError(f"Expected RGB files in {check_path_path}, but found none. Check the zip structure.")
        print(f"Done. {n_rgb} RGB files available in {check_path_path}")
    else:
        print("Done.")
    return check_path_path


def ensure_test_predictions(config_path: str, weights: str, out_dir: str) -> Path:
    out_dir_p = Path(out_dir)
    if _prediction_dir_has_npy(out_dir_p):
        print(f"Predictions already exist: {out_dir_p}")
        return out_dir_p
    weights_path = _resolve_repo_path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Prediction weights not found: {weights_path}")
    out_dir_p.mkdir(parents=True, exist_ok=True)
    _run_python(
        [
            str(REPO_ROOT / "src" / "predict.py"),
            "--config",
            config_path,
            "--weights",
            str(weights_path),
            "--out",
            str(out_dir_p),
        ]
    )
    return out_dir_p


def swap_selected_predictions(
    base_dir: str,
    replacement_dir: str,
    filelist: str,
    out_dir: str,
    replacement_name: str = "replacement",
) -> Path:
    """Swap a subset of predictions using a file list of test image stems."""
    out_dir_p = Path(out_dir)
    if not Path(base_dir).exists():
        raise FileNotFoundError(f"Base prediction directory not found: {base_dir}")
    if not Path(replacement_dir).exists():
        raise FileNotFoundError(f"Replacement prediction directory not found: {replacement_dir}")
    _run_python(
        [
            str(REPO_ROOT / "tools" / "swap_predictions.py"),
            "--base",
            base_dir,
            "--replacement",
            replacement_dir,
            "--filelist",
            filelist,
            "--out",
            str(out_dir_p),
            "--replacement-name",
            replacement_name,
        ]
    )
    return out_dir_p


def build_submission_csv(
    config_path: str,
    drive_submissions: str,
    submit_variant: str = "auto",
    pred_root: str = "/content/test_preds",
    base_checkpoint: str | None = None,
) -> Path:
    run_name, best_weights = resolve_base_run(config_path, base_checkpoint=base_checkpoint)
    try:
        epoch_swap_epoch = resolve_swap_epoch(config_path)
    except ValueError:
        epoch_swap_epoch = None
    pred_root_p = Path(pred_root)
    base_dir    = pred_root_p / run_name
    vggt_dir    = pred_root_p / f"{run_name}_vggt_swap"
    epoch_dir   = pred_root_p / f"{run_name}_epoch{epoch_swap_epoch}_veg_swap" if epoch_swap_epoch is not None else None
    refined_dir = pred_root_p / f"{run_name}_refinement_swap"

    ensure_test_predictions(
        config_path=config_path,
        weights=str(best_weights),
        out_dir=str(base_dir),
    )

    variant = submit_variant.removesuffix("_swap")
    if variant == "base":
        pred_dir, pred_tag = base_dir, run_name
    elif variant == "vggt":
        pred_dir, pred_tag = vggt_dir, f"{run_name}_vggt_swap"
    elif variant == "epoch":
        if epoch_dir is None:
            raise ValueError("submit_variant='epoch' requires save_epoch_checkpoint: N in the config.")
        pred_dir, pred_tag = epoch_dir, f"{run_name}_epoch{epoch_swap_epoch}_veg_swap"
    elif variant == "refinement":
        pred_dir, pred_tag = refined_dir, f"{run_name}_refinement_swap"
    else:  # auto
        candidates = [
            (refined_dir, f"{run_name}_refinement_swap"),
            (vggt_dir,    f"{run_name}_vggt_swap"),
            *([(epoch_dir, f"{run_name}_epoch{epoch_swap_epoch}_veg_swap")] if epoch_dir else []),
            (base_dir,    run_name),
        ]
        pred_dir, pred_tag = next((d, t) for d, t in candidates if d.exists())

    if not pred_dir.exists():
        raise FileNotFoundError(f'Submit variant "{submit_variant}" expects predictions at: {pred_dir}')

    print(f"Building CSV from: {pred_dir}")
    sub_csv = Path(drive_submissions) / f"{pred_tag}.csv"
    sub_csv.parent.mkdir(parents=True, exist_ok=True)

    _run_python(
        [
            str(REPO_ROOT / "tools" / "create_submission.py"),
            "--pred-dir",
            str(pred_dir),
            "--out",
            str(sub_csv),
        ]
    )
    print(f"CSV saved to: {sub_csv}")
    return sub_csv


def kaggle_submit(sub_csv: str | Path, kaggle_token: str) -> None:
    import shutil
    sub_csv = Path(sub_csv)
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = kaggle_token
    kaggle_bin = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    subprocess.run(
        [kaggle_bin, "competitions", "submit",
         "-c", "ethz-cil-monocular-depth-estimation-2026",
         "-f", str(sub_csv), "-m", sub_csv.stem],
        check=True, env=env,
    )
    print(f"Submitted: {sub_csv.name}")
