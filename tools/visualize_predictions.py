import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image as PILImage


def visualize(pred_dir, out_path, test_dir=None, veglist=None, n=12, title=None):
    pred_dir = Path(pred_dir)

    if veglist:
        fnames = [
            line.strip().split('.', 1)[1].strip().split()[0]
            for line in Path(veglist).read_text().splitlines() if line.strip()
        ]
        files = [pred_dir / (f.replace('_rgb.png', '') + '.npy') for f in fnames]
        files = [f for f in files if f.exists()]
    else:
        files = sorted(f for f in pred_dir.glob('test_*.npy') if '_conf' not in f.name)

    n       = min(n, len(files))
    sample  = files[::max(1, len(files) // n)][:n]
    n_cols  = 3 if test_dir is None else 6
    n_rows  = (n + (n_cols // (1 if test_dir is None else 2)) - 1) // (n_cols // (1 if test_dir is None else 2))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    axes = np.array(axes).reshape(n_rows, n_cols)

    per_row = n_cols if test_dir is None else n_cols // 2
    for i, f in enumerate(sample):
        row, col = divmod(i, per_row)
        depth = np.load(f)

        if test_dir:
            rgb_f  = Path(test_dir) / (f.stem + '_rgb.png')
            ax_rgb = axes[row, col * 2]
            ax_rgb.imshow(PILImage.open(rgb_f)) if rgb_f.exists() else ax_rgb.set_facecolor('k')
            ax_rgb.set_title(f.stem, fontsize=6)
            ax_rgb.axis('off')
            ax_dep = axes[row, col * 2 + 1]
        else:
            ax_dep = axes[row, col]
            ax_dep.set_title(f.stem, fontsize=6)

        ax_dep.imshow(depth, cmap='magma_r')
        ax_dep.set_title(f'[{depth.min():.1f}, {depth.max():.1f}] m', fontsize=6)
        ax_dep.axis('off')

    for ax in axes.flat[len(sample) * (2 if test_dir else 1):]:
        ax.axis('off')

    plt.suptitle(title or pred_dir.name, fontsize=9)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path} ({n} images)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pred-dir',  required=True)
    p.add_argument('--out',       required=True)
    p.add_argument('--test-dir',  default=None)
    p.add_argument('--veglist',   default=None)
    p.add_argument('--n',         type=int, default=12)
    p.add_argument('--title',     default=None)
    args = p.parse_args()
    visualize(args.pred_dir, args.out, args.test_dir, args.veglist, args.n, args.title)
