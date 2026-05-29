import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block as VitBlock
from torch.utils.checkpoint import checkpoint as grad_ckpt


def _sincos_1d(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (100.0 ** (omega / (embed_dim / 2.0)))
    out = torch.einsum('m,d->md', pos.double().reshape(-1), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def _uv_sincos_embed(width, height, embed_dim, dtype, device):
    ar = float(width) / float(height)
    diag = (ar ** 2 + 1.0) ** 0.5
    sx, sy = ar / diag, 1.0 / diag
    xs = torch.linspace(-sx*(width-1)/width,   sx*(width-1)/width,  width,  dtype=dtype, device=device)
    ys = torch.linspace(-sy*(height-1)/height, sy*(height-1)/height, height, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(xs, ys, indexing='xy')
    grid = torch.stack([uu, vv], dim=-1).reshape(-1, 2)
    ex = _sincos_1d(embed_dim // 2, grid[:, 0])
    ey = _sincos_1d(embed_dim // 2, grid[:, 1])
    return torch.cat([ex, ey], dim=-1).view(width, height, embed_dim)


class ResidualConvUnit(nn.Module):

    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return x + out


class DPTFusionBlock(nn.Module):

    def __init__(self, features, has_residual=True):
        super().__init__()
        self.has_residual = has_residual
        if has_residual:
            self.rcu1 = ResidualConvUnit(features)
        self.rcu2     = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, 1, bias=True)

    def forward(self, x, res=None, size=None):
        output = x
        if self.has_residual and res is not None:
            output = output + self.rcu1(res)
        output = self.rcu2(output)
        if size is not None:
            output = F.interpolate(output.contiguous(), size=size, mode='bilinear', align_corners=True)
        else:
            output = F.interpolate(output.contiguous(), scale_factor=2, mode='bilinear', align_corners=True)
        return self.out_conv(output)


class AggregatorDPTHead(nn.Module):
    """Transformer aggregator followed by a DPT multi-scale decoder."""

    def __init__(self, grid_size=40, agg_dim=1024, num_blocks=4,
                 num_fusion=None, num_heads=16, features=256, target_size=560,
                 gradient_checkpointing=False, use_conf=False):
        super().__init__()
        if num_fusion is None:
            num_fusion = num_blocks
        assert num_blocks % num_fusion == 0, 'num_blocks must be divisible by num_fusion'
        self.num_fusion  = num_fusion
        self.grid_size   = grid_size
        self.use_ckpt    = gradient_checkpointing
        self.target_size = target_size

        self.blocks = nn.ModuleList([
            VitBlock(dim=agg_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True,
                     init_values=0.01)
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(agg_dim)

        self.layer_rn = nn.ModuleList([
            nn.Conv2d(agg_dim, features, 3, padding=1, bias=False)
            for _ in range(num_fusion)
        ])

        # Reassemble ops anchor scale at grid_size (40×40):
        #   from_end = num_fusion - 1 - i  (0 = deepest/coarsest tap)
        #   from_end == 0, N > 1  →  Conv2d stride-2       (→ 20×20)
        #   from_end == 1, N > 1  →  Identity               (→ 40×40)
        #   from_end == 1, N == 1 →  Identity               (→ 40×40)
        #   from_end == 2         →  ConvTranspose2d ×2     (→ 80×80)
        #   from_end == 3         →  ConvTranspose2d ×4     (→ 160×160)
        reassemble_ops = []
        for i in range(num_fusion):
            from_end = num_fusion - 1 - i
            if from_end == 0 and num_fusion > 1:
                reassemble_ops.append(nn.Conv2d(features, features, 3, stride=2, padding=1))
            elif from_end <= 1:
                reassemble_ops.append(nn.Identity())
            else:
                stride = 2 ** (from_end - 1)
                reassemble_ops.append(nn.ConvTranspose2d(features, features, stride, stride=stride))
        self.reassemble = nn.ModuleList(reassemble_ops)

        self.fusion = nn.ModuleList(
            [DPTFusionBlock(features, has_residual=False)] +
            [DPTFusionBlock(features, has_residual=True) for _ in range(num_fusion - 1)]
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(features, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Softplus(),
        )

        self.use_conf = use_conf
        if use_conf:
            self.conf_conv = nn.Sequential(
                nn.Conv2d(features, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
            )

    def _apply_pos_embed(self, x, ratio=0.1):
        _, C, patch_h, patch_w = x.shape
        embed = _uv_sincos_embed(patch_w, patch_h, C, dtype=torch.float32, device=x.device)
        embed = embed.permute(2, 0, 1).unsqueeze(0) * ratio
        return x + embed.to(x.dtype)

    def forward(self, encoder_out):
        feat = encoder_out[0]   # (B, N, embed_dim)
        B    = feat.shape[0]

        x = feat

        tap_stride = len(self.blocks) // self.num_fusion
        tapped     = []

        for i, block in enumerate(self.blocks):
            x = grad_ckpt(block, x, use_reentrant=False) if self.use_ckpt else block(x)
            if (i + 1) % tap_stride == 0:
                tapped.append(x)

        rn = []
        for i, (feat_t, rn_layer, reasm) in enumerate(zip(tapped, self.layer_rn, self.reassemble)):
            t = self.norm(feat_t)
            t = t.permute(0, 2, 1).reshape(B, -1, self.grid_size, self.grid_size)
            t = self._apply_pos_embed(t)
            t = rn_layer(t)      # (B, features, 40, 40)
            t = reasm(t)
            rn.append(t)

        N = self.num_fusion
        out = self.fusion[0](rn[N-1], size=rn[N-2].shape[2:] if N > 1 else None)
        for k in range(1, N):
            size = rn[N-2-k].shape[2:] if k < N-1 else None
            out  = self.fusion[k](out, rn[N-1-k], size=size)

        out = F.interpolate(out.contiguous(), size=self.target_size, mode='bilinear', align_corners=True)
        depth = self.output_conv(out)   # (B, 1, 560, 560)
        if self.use_conf:
            log_conf = self.conf_conv(out)  # (B, 1, 560, 560)
            return torch.cat([depth, log_conf], dim=1)  # (B, 2, 560, 560)
        return depth


def build_head(cfg, embed_dim, grid_size, target_size=560):
    agg = cfg.get('aggregator', {})
    return AggregatorDPTHead(
        grid_size=grid_size,
        agg_dim=agg.get('agg_dim', embed_dim),
        num_blocks=agg.get('num_vit_blocks', 4),
        num_fusion=agg.get('num_dpt_blocks', None),
        num_heads=agg.get('num_heads', 16),
        features=cfg.get('decoder_features', 256),
        target_size=target_size,
        gradient_checkpointing=agg.get('gradient_checkpointing', False),
        use_conf=cfg.get('use_conf', False),
    )
