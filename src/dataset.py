import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CILDepthDataset(Dataset):
    # train: returns (img, depth), test: returns (img, filename)
    # no normalization here — encoder handles that

    def __init__(self, data_dir, split='train', img_size=560):
        self.data_dir  = data_dir if split is None else os.path.join(data_dir, split)
        self.split     = split if split is not None else 'train'
        self.img_size  = img_size
        self.rgb_files = sorted(f for f in os.listdir(self.data_dir) if f.endswith('_rgb.png'))

        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_fname = self.rgb_files[idx]
        img       = Image.open(os.path.join(self.data_dir, rgb_fname)).convert('RGB')
        img       = self.img_transform(img)

        if self.split == 'test':
            return img, rgb_fname

        depth_fname = rgb_fname.replace('_rgb.png', '_depth.npy')
        depth       = np.load(os.path.join(self.data_dir, depth_fname)).astype(np.float32)
        depth       = torch.from_numpy(depth).unsqueeze(0)  # (H,W) -> (1,H,W)

        return img, depth
