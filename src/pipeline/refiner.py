import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentGuidedRefiner(nn.Module):
    """SAM-guided segment-mean depth refinement with confidence-gated blending."""

    def __init__(self, max_segs=256):
        super().__init__()
        self.max_segs = max_segs
        self.blend = nn.Conv2d(1, 1, 1)
        nn.init.constant_(self.blend.weight, -1.0)
        nn.init.zeros_(self.blend.bias)

    def forward(self, depth, log_conf, sam_feats):
        B, _, H, W = depth.shape
        sam_f = sam_feats.float()

        l1 = F.interpolate(sam_f[:, 6:7], size=(H, W), mode='nearest').squeeze(1).round().long()
        l2 = F.interpolate(sam_f[:, 7:8], size=(H, W), mode='nearest').squeeze(1).round().long()
        w1 = F.interpolate(sam_f[:, :1],  size=(H, W), mode='bilinear', align_corners=True)
        w2 = F.interpolate(sam_f[:, 1:2], size=(H, W), mode='bilinear', align_corners=True)

        mean1 = self._pool(depth, w1, l1)
        mean2 = self._pool(depth, w2, l2)

        w1_n      = w1 / (w1 + w2 + 1e-6)
        consensus = w1_n * mean1 + (1.0 - w1_n) * mean2
        alpha     = torch.sigmoid(self.blend(log_conf))
        return (alpha * consensus + (1.0 - alpha) * depth).clamp(1e-3, 80.0)

    def _pool(self, depth, conf_weight, labels):
        B, _, H, W = depth.shape
        N = H * W

        labels_norm = torch.zeros(B, N, dtype=torch.long, device=depth.device)
        for b in range(B):
            _, inv = torch.unique(labels[b].reshape(N), return_inverse=True)
            labels_norm[b] = inv.clamp(0, self.max_segs - 1)

        depth_flat = depth.squeeze(1).reshape(B, N)
        conf_flat  = conf_weight.squeeze(1).reshape(B, N).clamp(min=1e-6)

        weighted_sum = depth_flat.new_zeros(B, self.max_segs)
        weight_sum   = depth_flat.new_zeros(B, self.max_segs)
        weighted_sum.scatter_add_(1, labels_norm, depth_flat * conf_flat)
        weight_sum.scatter_add_(1, labels_norm, conf_flat)

        return (weighted_sum / (weight_sum + 1e-6)).gather(1, labels_norm).reshape(B, 1, H, W)


def build_refiner(cfg):
    return SegmentGuidedRefiner(max_segs=cfg.get('refiner', {}).get('max_segs', 256))
