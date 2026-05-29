import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import CILDepthDataset
from src.model import DepthModel

USE_AMP = torch.cuda.is_available()
AMP_DTYPE = torch.bfloat16 if (USE_AMP and torch.cuda.is_bf16_supported()) else torch.float16


def predict(cfg, weights, out_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DepthModel(cfg)
    ckpt  = torch.load(weights, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()

    test_ds = CILDepthDataset(cfg['data_dir'], split='test')

    test_dl = DataLoader(test_ds, batch_size=cfg.get('batch_size', 8), shuffle=False,
                         num_workers=max(1, (os.cpu_count() or 4) - 1),
                         pin_memory=device.type == 'cuda')

    os.makedirs(out_dir, exist_ok=True)
    use_conf = cfg.get('head', {}).get('use_conf', False)
    n = 0
    with torch.no_grad():
        for feats, fnames in test_dl:
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                out = model(feats.to(device)).float()

            preds = out[:, 0]
            confs = out[:, 1].exp() if (use_conf and out.shape[1] > 1) else None

            for j, (pred, fname) in enumerate(zip(preds, fnames)):
                depth = np.clip(pred.cpu().numpy(), 1e-3, 80.0)
                idx   = fname.split('_')[1]
                np.save(os.path.join(out_dir, f'test_{idx}.npy'), depth)
                if confs is not None:
                    np.save(os.path.join(out_dir, f'test_{idx}_conf.npy'), confs[j].cpu().numpy())
                n += 1

    print(f'Saved {n} predictions to {out_dir}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',  required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--out',     required=True)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    predict(cfg, args.weights, args.out)
