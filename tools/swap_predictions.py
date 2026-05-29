import os, re, shutil, argparse


def parse_filelist(path):
    stems = set()
    with open(path) as f:
        for line in f:
            m = re.search(r'(test_\d+)_rgb\.png', line)
            if m:
                stems.add(m.group(1))
    return stems


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base',     required=True)
    p.add_argument('--replacement', default=None,
                   help='directory containing replacement .npy predictions')
    p.add_argument('--vggt', default=None,
                   help='backward-compatible alias for --replacement')
    p.add_argument('--filelist', required=True)
    p.add_argument('--out',      required=True)
    p.add_argument('--replacement-name', default='replacement',
                   help='label used in logs for the replacement predictor')
    args = p.parse_args()
    replacement_dir = args.replacement or args.vggt
    if not replacement_dir:
        p.error('one of --replacement or --vggt is required')

    swap_stems = parse_filelist(args.filelist)
    print(f'Images to swap: {len(swap_stems)}', flush=True)

    os.makedirs(args.out, exist_ok=True)

    n_copied = n_swapped = n_missing = 0
    for fname in os.listdir(args.base):
        if not fname.endswith('.npy'):
            continue
        stem = fname[:-4]
        src  = os.path.join(args.base, fname)
        dst  = os.path.join(args.out, fname)
        if stem in swap_stems:
            replacement_src = os.path.join(replacement_dir, fname)
            if os.path.exists(replacement_src):
                shutil.copy2(replacement_src, dst)
                n_swapped += 1
            else:
                print(f'  WARNING: {args.replacement_name} pred missing for {fname}, keeping base', flush=True)
                shutil.copy2(src, dst)
                n_missing += 1
        else:
            shutil.copy2(src, dst)
            n_copied += 1

    print(f'Kept base: {n_copied} | Swapped to {args.replacement_name}: {n_swapped} | Missing {args.replacement_name}: {n_missing}',
          flush=True)
    print(f'Output: {args.out}', flush=True)


if __name__ == '__main__':
    main()
