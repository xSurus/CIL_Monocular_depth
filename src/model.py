import torch.nn as nn
from src.pipeline.encoders import build_encoder
from src.pipeline.heads import build_head


class DepthModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._encoder_frozen = True
        self.encoder = build_encoder({'name': 'dinov2', 'freeze': True})
        embed_dim, grid_size = self.encoder.embed_dim, self.encoder.grid_size
        self.head = build_head(cfg.get('head', {}), embed_dim, grid_size)

    def train(self, mode=True):
        super().train(mode)
        if mode and self._encoder_frozen and self.encoder is not None:
            self.encoder.eval()
        return self

    def forward(self, x):
        return self.head(self.encoder(x))
