import argparse
import os
import random

import torch
import torchvision.transforms as T
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler

from src.dataset import CILDepthDataset
from src.losses import build_loss
from src.model import DepthModel
from src.repro import get_seed, make_torch_generator, seed_everything, seed_worker


USE_AMP = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16 if (USE_AMP and torch.cuda.is_bf16_supported()) else torch.float16
torch.set_float32_matmul_precision('high')


class AugSubset(Dataset):
    """Wraps a dataset subset with augmentation. Spatial transforms (hflip) hit img+depth, color jitter only img."""

    def __init__(self, inner, aug_cfg):
        self.inner        = inner
        self.hflip_p      = aug_cfg.get('hflip', 0.0)
        cj                = aug_cfg.get('color_jitter')
        self.color_jitter = T.ColorJitter(*cj) if cj else None

    @property
    def indices(self):
        return self.inner.indices if hasattr(self.inner, 'indices') else list(range(len(self.inner)))

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        item  = self.inner[i]
        img   = item[0]
        depth = item[1]
        if random.random() < self.hflip_p:
            img   = torch.flip(img,   dims=[-1])
            depth = torch.flip(depth, dims=[-1])
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        if len(item) == 2:
            return img, depth
        return (img, depth) + tuple(item[2:])


def compute_si_rmse(model, val_dl, device):
    """Global siRMSE matching Kaggle: pool all pixel log-diffs, compute sqrt(Var(d))."""
    model.eval()
    sum_d, sum_d2, count = 0.0, 0.0, 0
    eps = 1e-6
    with torch.no_grad():
        for imgs, depths in val_dl:
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                preds = model(imgs.to(device, non_blocking=True))
            depths = depths.to(device, non_blocking=True)
            preds = preds[:, :1].float().clamp(1e-3, 80.0)
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


def build_dataloaders(cfg):
    import pandas as pd

    full      = CILDepthDataset(cfg['data_dir'], split='train')
    seed = get_seed(cfg)
    loader_cfg = cfg.get('dataloader', {})
    num_workers_cfg = loader_cfg.get('num_workers', 'auto')
    if num_workers_cfg == 'auto':
        n_workers = max(1, (os.cpu_count() or 4) // 2)
    else:
        n_workers = max(0, int(num_workers_cfg))
    pin_memory = bool(loader_cfg.get('pin_memory', True))
    persistent_workers = bool(loader_cfg.get('persistent_workers', True)) and n_workers > 0
    prefetch_factor = int(loader_cfg.get('prefetch_factor', 4))

    # aerial filter
    if cfg.get('filter_aerial_csv'):
        aerial_df   = pd.read_csv(cfg['filter_aerial_csv'])
        aerial_set  = set(aerial_df[aerial_df['label'] == 'aerial']['file'])
        keep        = [i for i, f in enumerate(full.rgb_files) if f not in aerial_set]
        print(f'Aerial filter: removed {len(full) - len(keep)}, kept {len(keep)}', flush=True)
    else:
        keep = list(range(len(full)))

    # train/val split
    n_val = int(cfg['val_frac'] * len(keep))

    if cfg.get('water_csv'):
        water_df   = pd.read_csv(cfg['water_csv'])
        water_set  = set(water_df[water_df['label'] == 'water']['file'])
        water_keep    = [i for i in keep if full.rgb_files[i] in water_set]
        nonwater_keep = [i for i in keep if full.rgb_files[i] not in water_set]

        val_water_frac  = cfg.get('val_water_frac', 0.30)
        n_val_water     = min(int(val_water_frac * n_val), len(water_keep))
        n_val_nonwater  = n_val - n_val_water

        rng = random.Random(seed)
        rng.shuffle(water_keep)
        rng.shuffle(nonwater_keep)

        val_indices   = water_keep[:n_val_water]   + nonwater_keep[:n_val_nonwater]
        train_indices = water_keep[n_val_water:]   + nonwater_keep[n_val_nonwater:]

        actual_water_pct = 100 * n_val_water / len(val_indices) if val_indices else 0
        print(f'Stratified split | train={len(train_indices)} val={len(val_indices)} '
              f'({actual_water_pct:.1f}% water in val)', flush=True)
    else:
        rng  = random.Random(seed)
        keep = keep[:]
        rng.shuffle(keep)
        val_indices   = keep[:n_val]
        train_indices = keep[n_val:]
        print(f'Random split | train={len(train_indices)} val={len(val_indices)}', flush=True)

    train_ds = Subset(full, train_indices)
    val_base = Subset(full, val_indices)

    # augmentation
    aug_cfg = cfg.get('augmentation', {})
    if aug_cfg:
        train_ds = AugSubset(train_ds, aug_cfg)
        print(f'Augmentation: hflip={aug_cfg.get("hflip", 0)} color_jitter={aug_cfg.get("color_jitter")}', flush=True)

    # extra training data (e.g. overgrown)
    extra_train_dir = cfg.get('extra_train_dir')
    n_extra = 0
    if extra_train_dir:
        extra_ds = CILDepthDataset(extra_train_dir, split=None, img_size=cfg.get('img_size', 560))
        n_extra = len(extra_ds)
        if n_extra == 0:
            raise ValueError(
                f'extra_train_dir is configured but contains 0 training samples: {extra_train_dir}. '
                'Expected paired *_rgb.png and *_depth.npy files directly in that directory.'
            )
        if aug_cfg:
            extra_ds = AugSubset(extra_ds, aug_cfg)
        train_ds = ConcatDataset([train_ds, extra_ds])
        print(f'Extra train data: {n_extra} samples from {extra_train_dir}', flush=True)

    # dataloaders
    train_dl_kwargs = {
        'batch_size': cfg['batch_size'],
        'num_workers': n_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers,
        'drop_last': True,
        'generator': make_torch_generator(seed + 1),
    }
    eval_dl_kwargs = {
        'batch_size': cfg['batch_size'],
        'shuffle': False,
        'num_workers': n_workers,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers,
        'generator': make_torch_generator(seed + 2),
    }
    if n_workers > 0:
        train_dl_kwargs['prefetch_factor'] = prefetch_factor
        eval_dl_kwargs['prefetch_factor'] = prefetch_factor
        train_dl_kwargs['worker_init_fn'] = seed_worker
        eval_dl_kwargs['worker_init_fn'] = seed_worker

    # Optional water upsampling via WeightedRandomSampler
    water_upsample = cfg.get('water_upsample', 1.0)
    if water_upsample != 1.0 and cfg.get('water_csv'):
        weights = [water_upsample if full.rgb_files[i] in water_set else 1.0
                   for i in train_indices]
        weights += [1.0] * n_extra
        sampler  = WeightedRandomSampler(
            weights,
            num_samples=len(train_indices) + n_extra,
            replacement=True,
            generator=make_torch_generator(seed),
        )
        train_dl = DataLoader(train_ds, sampler=sampler, **train_dl_kwargs)
        print(f'Water upsampling: {water_upsample}x', flush=True)
    else:
        train_dl = DataLoader(train_ds, shuffle=True, **train_dl_kwargs)

    val_dl = DataLoader(val_base, **eval_dl_kwargs)
    print(f'DataLoader: workers={n_workers} pin_memory={pin_memory} '
          f'persistent_workers={persistent_workers} '
          + (f'prefetch_factor={prefetch_factor}' if n_workers > 0 else 'prefetch_factor=n/a'),
          flush=True)
    return train_dl, val_dl


def build_optimizer(model, cfg):
    return AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=cfg['lr'], weight_decay=1e-4)


def parse_epoch_checkpoint(cfg):
    raw = cfg.get('save_epoch_checkpoint')
    if raw is None:
        return None
    epoch = int(raw)
    if epoch < 1:
        raise ValueError(f'save_epoch_checkpoint must be a positive epoch, got {epoch}')
    if epoch > int(cfg['epochs']):
        raise ValueError(f'save_epoch_checkpoint exceeds total epochs={cfg["epochs"]}: {epoch}')
    return epoch


def run_train_epoch(model, dl, optimizer, scaler, loss_fn, device, epoch):
    model.train()
    total_loss = 0.0
    n_valid    = 0
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    for i, (imgs, depths) in enumerate(dl):
        imgs   = imgs.to(device, non_blocking=True)
        depths = depths.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
            out = model(imgs)
        out     = out.float()
        depth_f = out[:, :1].clamp(1e-3, 80.0)
        preds_f = torch.cat([depth_f, out[:, 1:]], dim=1) if out.shape[1] > 1 else depth_f
        loss    = loss_fn(preds_f, depths)
        if not torch.isfinite(loss):
            print(f'  WARNING: non-finite loss at batch {i+1}, skipping', flush=True)
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n_valid    += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(dl):
            print(f'  E{epoch+1} [{i+1}/{len(dl)}] loss={loss.item():.4f}', flush=True)
    avg_loss = total_loss / n_valid if n_valid > 0 else float('nan')
    print(f'  Epoch {epoch+1} done | avg_loss={avg_loss:.4f}', flush=True)
    return avg_loss


def save_checkpoint(model, cfg, val_rmse, best_val):
    os.makedirs(os.path.realpath('checkpoints'), exist_ok=True)
    name = cfg['name']
    if val_rmse < best_val:
        best_val = val_rmse
        torch.save(model.state_dict(), f'checkpoints/{name}_best.pt')
        print(f'  -> Saved new best (siRMSE={val_rmse:.4f})', flush=True)
    return best_val


def save_epoch_checkpoint(model, cfg, epoch_num):
    path = f'checkpoints/{cfg["name"]}_epoch{epoch_num}.pt'
    torch.save(model.state_dict(), path)
    print(f'  -> Saved requested epoch checkpoint: {path}', flush=True)


def write_run_aliases(base_name, resolved_name):
    os.makedirs(os.path.realpath('checkpoints'), exist_ok=True)
    with open('checkpoints/last_run_name.txt', 'w') as f:
        f.write(resolved_name + '\n')
    with open(f'checkpoints/{base_name}_latest.txt', 'w') as f:
        f.write(resolved_name + '\n')


def train(cfg):
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_epochs = cfg['epochs']

    cfg = dict(cfg)
    seed = seed_everything(get_seed(cfg))
    epoch_checkpoint = parse_epoch_checkpoint(cfg)
    base = cfg['name']
    if os.path.exists(f'checkpoints/{base}_best.pt'):
        v = 1
        while os.path.exists(f'checkpoints/{base}_v{v}_best.pt'):
            v += 1
        cfg['name'] = f'{base}_v{v}'
        print(f'Checkpoint exists — saving as: {cfg["name"]}', flush=True)
    write_run_aliases(base, cfg['name'])

    print(f'device={device} | epochs={n_epochs} | bs={cfg["batch_size"]}'
          + f' | name={cfg["name"]} | seed={seed}', flush=True)
    if epoch_checkpoint is not None:
        print(f'Extra epoch checkpoint: {epoch_checkpoint}', flush=True)

    model = DepthModel(cfg).to(device)
    train_dl, val_dl = build_dataloaders(cfg)
    loss_fn               = build_loss(cfg)
    optimizer             = build_optimizer(model, cfg)
    scheduler             = CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler                = torch.amp.GradScaler('cuda', enabled=(USE_AMP and AMP_DTYPE == torch.float16))
    best_val = float('inf')

    for epoch in range(n_epochs):
        train_loss = run_train_epoch(model, train_dl, optimizer, scaler, loss_fn, device, epoch)
        scheduler.step()
        val_rmse   = compute_si_rmse(model, val_dl, device)
        torch.cuda.empty_cache()
        best_val = save_checkpoint(model, cfg, val_rmse, best_val)
        if epoch_checkpoint is not None and (epoch + 1) == epoch_checkpoint:
            save_epoch_checkpoint(model, cfg, epoch + 1)
        print(f'Epoch {epoch+1:3d}/{n_epochs} | train_loss={train_loss:.4f} | val_si_rmse={val_rmse:.4f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',          required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
