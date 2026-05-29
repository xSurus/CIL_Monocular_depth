import os, yaml, argparse
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.dataset          import CILDepthDataset
from src.pipeline.refiner import build_refiner
from src.losses           import si_log_loss
from src.train            import USE_AMP, AMP_DTYPE


def _load_sam(npz_path):
    d = np.load(str(npz_path))
    return np.stack([
        d['conf_top1'].astype(np.float32),
        d['conf_top2'].astype(np.float32),
        d['pixel_uncertainty'].astype(np.float32),
        np.clip(d['coverage'].astype(np.float32) / 10.0, 0.0, 1.0),
        d['sam_boundary'].astype(np.float32),
        d['boundary_uncertainty'].astype(np.float32),
        d['label_top1'].astype(np.float32),
        d['label_top2'].astype(np.float32),
    ], axis=0)  # (8, H, W)


class RefinementDataset(Dataset):
    def __init__(self, base_ds, depth_dir, sam_dir, indices):
        self.base_ds   = getattr(base_ds, 'dataset', base_ds)
        self.depth_dir = Path(depth_dir)
        self.sam_dir   = Path(sam_dir)
        self.indices   = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx        = self.indices[i]
        depth_pred = torch.from_numpy(np.load(self.depth_dir / f'train_{idx:06d}_depth.npy').astype(np.float32))
        sam        = torch.from_numpy(_load_sam(self.sam_dir / f'train_{idx:06d}_rgb_sam_v2.npz'))
        fname      = self.base_ds.rgb_files[idx]
        depth_gt   = np.load(os.path.join(self.base_ds.data_dir, fname.replace('_rgb.png', '_depth.npy'))).astype(np.float32)
        return depth_pred, sam, torch.from_numpy(depth_gt).unsqueeze(0)


def compute_val_rmse(refiner, val_dl, device):
    refiner.eval()
    sum_d, sum_d2, count = 0.0, 0.0, 0
    eps = 1e-6
    with torch.no_grad():
        for depth_pred, sam, depths in val_dl:
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                preds = refiner(depth_pred[:, :1].to(device, non_blocking=True).clamp(1e-3, 80.0),
                                depth_pred[:, 1:].to(device, non_blocking=True),
                                sam.to(device, non_blocking=True)).float()
            depths = depths.to(device, non_blocking=True)
            mask  = (depths > eps) & (preds > eps) & torch.isfinite(preds)
            d     = (torch.log(preds[mask]) - torch.log(depths[mask])).clamp(-100.0, 100.0)
            if d.numel() > 0:
                sum_d  += d.sum().item()
                sum_d2 += d.pow(2).sum().item()
                count  += d.numel()
    if count == 0:
        return float('nan')
    mean_d   = sum_d / count
    variance = sum_d2 / count - mean_d ** 2
    return max(0.0, variance) ** 0.5


def train(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ref    = cfg['refiner']

    depth_dir, sam_dir = cfg.get('depth_dir'), cfg.get('sam_dir')
    assert depth_dir and Path(depth_dir).exists(), f'depth_dir missing: {depth_dir!r}'
    assert sam_dir   and Path(sam_dir).exists(),   f'sam_dir missing: {sam_dir!r}'

    full  = CILDepthDataset(cfg['data_dir'], split='train')
    n_val = int(ref['val_frac'] * len(full))
    train_sub, val_sub = random_split(full, [len(full) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(42))

    dl_kwargs = dict(num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    train_dl  = DataLoader(RefinementDataset(train_sub, depth_dir, sam_dir, list(train_sub.indices)),
                           batch_size=ref['batch_size'], shuffle=True,  drop_last=True, **dl_kwargs)
    val_dl    = DataLoader(RefinementDataset(val_sub,   depth_dir, sam_dir, list(val_sub.indices)),
                           batch_size=ref['batch_size'], shuffle=False, **dl_kwargs)

    refiner   = build_refiner(cfg).to(device)
    optimizer = AdamW(refiner.parameters(), lr=ref['lr'], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=ref['epochs'])
    scaler    = torch.amp.GradScaler('cuda', enabled=(USE_AMP and AMP_DTYPE == torch.float16))

    os.makedirs('checkpoints', exist_ok=True)
    best_val, name = float('inf'), ref['name']
    print(f'device={device} | epochs={ref["epochs"]} | bs={ref["batch_size"]} | '
          f'train={len(train_dl.dataset)} | val={len(val_dl.dataset)}', flush=True)

    for epoch in range(ref['epochs']):
        refiner.train()
        total_loss, n_batches = 0.0, 0
        for depth_pred, sam, depths in train_dl:
            depth_pred, sam, depths = depth_pred.to(device), sam.to(device), depths.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                refined = refiner(depth_pred[:, :1].clamp(1e-3, 80.0), depth_pred[:, 1:], sam)
            loss = si_log_loss(refined.float(), depths)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        val_rmse = compute_val_rmse(refiner, val_dl, device)
        torch.cuda.empty_cache()

        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(refiner.state_dict(), f'checkpoints/{name}_best.pt')

        print(f'Epoch {epoch+1:3d}/{ref["epochs"]} | '
              f'train_loss={total_loss/max(n_batches,1):.4f} | val_si_rmse={val_rmse:.4f}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',    required=True)
    p.add_argument('--depth_dir', default=None)
    p.add_argument('--sam_dir',   default=None)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.depth_dir:
        cfg['depth_dir'] = args.depth_dir
    if args.sam_dir:
        cfg['sam_dir'] = args.sam_dir
    train(cfg)
