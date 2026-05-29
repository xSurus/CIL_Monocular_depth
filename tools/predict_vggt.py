import os, argparse
import numpy as np
import torch
import torch.nn.functional as F

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
except ImportError:
    raise SystemExit(
        'vggt not installed.\n'
        'Run: pip install git+https://github.com/facebookresearch/vggt.git'
    )

IMG_SIZE = 560


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--out',      required=True)
    p.add_argument('--model',    default='facebook/VGGT-1B')
    p.add_argument('--split',    default='test', choices=['test', 'train'])
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.bfloat16 if (device.type == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

    model = VGGT.from_pretrained(args.model).to(device).eval()
    print('Model loaded.', flush=True)

    src_dir = os.path.join(args.data_dir, args.split)
    os.makedirs(args.out, exist_ok=True)

    rgb_files = sorted(f for f in os.listdir(src_dir) if f.endswith('_rgb.png'))
    print(f'Predicting {len(rgb_files)} images...', flush=True)

    for i, fname in enumerate(rgb_files):
        img_path = os.path.join(src_dir, fname)

        images = load_and_preprocess_images([img_path], mode='pad').to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype):
                predictions = model(images)

        # depth: (1, 1, H, W, 1) → (1, 1, H, W)
        depth = predictions['depth'][0, 0, :, :, 0]

        depth = F.interpolate(
            depth.float().unsqueeze(0).unsqueeze(0),
            size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=True
        ).squeeze().cpu().numpy()

        depth = np.clip(depth, 1e-3, 80.0).astype(np.float32)

        idx = fname.split('_')[1]
        np.save(os.path.join(args.out, f'test_{idx}.npy'), depth)

        if (i + 1) % 50 == 0 or (i + 1) == len(rgb_files):
            print(f'  {i + 1}/{len(rgb_files)}', flush=True)

    print(f'Done. Saved to {args.out}', flush=True)


if __name__ == '__main__':
    main()
