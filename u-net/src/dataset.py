from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class SimpleDepthDataset(Dataset):
    def __init__(self, root: Path, img_size=128, max_samples=None):
        self.root = Path(root)
        self.img_size = img_size
        
        self.rgb_files = sorted(self.root.glob("*_rgb.png"))
        if max_samples is not None:
            self.rgb_files = self.rgb_files[:max_samples]
        
        assert len(self.rgb_files) > 0, f"No *_rgb.png files found in {self.root}"

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        depth_path = Path(str(rgb_path).replace("_rgb.png", "_depth.npy"))
        
        # load rgb
        rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        
        # load depth
        depth = np.load(depth_path).astype(np.float32)
        
        # resize rgb
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)   # [1,3,H,W]
        rgb_t = F.interpolate(rgb_t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        rgb_t = rgb_t.squeeze(0)  # [3,H,W]
        
        # resize depth
        depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
        depth_t = F.interpolate(depth_t, size=(self.img_size, self.img_size), mode="nearest")
        depth_t = depth_t.squeeze(0)  # [1,H,W]
        
        # valid mask: depth > 0
        valid_mask = (depth_t > 0).float()
        
        # optional normalization of valid depth values
        # keeps the target range smaller and easier for the toy model
        depth_t = torch.clamp(depth_t, min=0.0, max=80.0)
        depth_t = depth_t / 80.0
        
        return {
            "image": rgb_t,
            "depth": depth_t,
            "mask": valid_mask,
            "name": rgb_path.name
        }