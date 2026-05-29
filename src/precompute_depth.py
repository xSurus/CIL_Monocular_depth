import os, yaml, argparse
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.dataset import CILDepthDataset
from src.model   import DepthModel

USE_AMP   = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16 if (USE_AMP and torch.cuda.is_bf16_supported()) else torch.float16


def precompute(cfg, weights, out_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DepthModel(cfg)
    ckpt  = torch.load(weights, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    assert cfg.get('head', {}).get('use_conf'), \
        "use_conf must be true in config — refiner requires depth + log_conf output"

    ds = CILDepthDataset(cfg['data_dir'], split='train')
    dl = DataLoader(ds, batch_size=cfg.get('batch_size', 32), shuffle=False,
                    num_workers=max(1, (os.cpu_count() or 4) - 2),
                    pin_memory=device.type == 'cuda')

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    with torch.no_grad():
        for imgs, _ in dl:
            imgs = imgs.to(device)
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                preds = model(imgs).float()   # (B, 2, 560, 560)
            for pred in preds:
                np.save(out_dir / f'train_{n:06d}_depth.npy', pred.cpu().numpy())
                n += 1
            if n % 1000 < cfg.get('batch_size', 32):
                print(f'  {n}/{len(ds)}', flush=True)

    print(f'Saved {n} predictions to {out_dir}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',  required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--out',     required=True)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    precompute(cfg, args.weights, args.out)
