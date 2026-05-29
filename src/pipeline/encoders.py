import contextlib
import torch
import torch.nn as nn
import timm

class DINOv2Encoder(nn.Module):
    # returns 1-tuple of (B, 1600, 1024) from block 23
    embed_dim = 1024
    grid_size = 40  # 560px / 14px = 40 patches per side

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.model = timm.create_model(
            'vit_large_patch14_dinov2.lvd142m',
            pretrained=True,
            img_size=560,
            dynamic_img_size=True,
            num_classes=0,
        )
        print('Loaded DINOv2 ViT-L/14', flush=True)

    def forward(self, x):
        x = (x - self.mean) / self.std
        ctx = torch.no_grad() if self._frozen else contextlib.nullcontext()
        with ctx:
            return self.model.get_intermediate_layers(x, n=[23])


def build_encoder(cfg):
    encoder = DINOv2Encoder()
    frozen = cfg.get('freeze', True)
    encoder._frozen = frozen
    if frozen:
        for p in encoder.parameters():
            p.requires_grad = False
    else:
        encoder.model.set_grad_checkpointing(enable=True)
    return encoder
