import os, yaml, argparse
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from src.dataset          import CILDepthDataset
from src.model            import DepthModel
from src.pipeline.refiner import build_refiner
from src.train            import USE_AMP, AMP_DTYPE
from src.train_refiner    import _load_sam


def _load_model(model, path):
    ckpt  = torch.load(path, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in state.items()})
    return model


class _TestDataset(Dataset):
    def __init__(self, base_ds, sam_dir):
        self.base    = base_ds
        self.sam_dir = Path(sam_dir)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, fname = self.base[i]
        sam        = torch.from_numpy(_load_sam(self.sam_dir / fname.replace('_rgb.png', '_rgb_sam_v2.npz')))
        return img, fname, sam


def predict(cfg, base_weights, refiner_weights, out_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert cfg.get('head', {}).get('use_conf'), \
        "use_conf must be true in config — refiner requires depth + log_conf"

    test_sam_dir = cfg.get('test_sam_dir')
    assert test_sam_dir and Path(test_sam_dir).exists(), f'test_sam_dir missing: {test_sam_dir!r}'

    base_model = _load_model(DepthModel(cfg).to(device).eval(), base_weights)
    refiner    = _load_model(build_refiner(cfg).to(device).eval(), refiner_weights)
    base_model.requires_grad_(False)

    dl = DataLoader(
        _TestDataset(CILDepthDataset(cfg['data_dir'], split='test'), test_sam_dir),
        batch_size=cfg.get('refiner', {}).get('batch_size', 8),
        shuffle=False,
        num_workers=max(1, (os.cpu_count() or 4) - 2),
        pin_memory=device.type == 'cuda',
    )

    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with torch.no_grad():
        for imgs, fnames, sam in dl:
            imgs, sam = imgs.to(device), sam.to(device)
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                base_out = base_model(imgs).float()
            depth_b, log_conf = base_out[:, :1].clamp(1e-3, 80.0), base_out[:, 1:]
            with torch.autocast(device_type='cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                refined = refiner(depth_b, log_conf, sam).float()
            for pred, fname in zip(refined, fnames):
                np.save(os.path.join(out_dir, f'test_{fname.split("_")[1]}.npy'),
                        np.clip(pred[0].cpu().numpy(), 1e-3, 80.0))
                n += 1

    print(f'Saved {n} refined predictions to {out_dir}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config',          required=True)
    p.add_argument('--base_weights',    required=True)
    p.add_argument('--refiner_weights', required=True)
    p.add_argument('--out',             required=True)
    p.add_argument('--test_sam_dir',    default=None)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.test_sam_dir:
        cfg['test_sam_dir'] = args.test_sam_dir
    predict(cfg, args.base_weights, args.refiner_weights, args.out)
