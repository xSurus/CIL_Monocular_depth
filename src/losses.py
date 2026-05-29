import torch


def si_log_loss(pred, target, lam=0.85, eps=1e-6, max_depth=float('inf')):
    """scale-invariant log loss, lam=0.85 avoids collapse to constant prediction"""
    mask = (target > eps) & (target < max_depth) & (pred > eps) & torch.isfinite(pred)
    d    = torch.log(pred[mask]) - torch.log(target[mask])
    if d.numel() == 0:
        return pred.new_zeros(1).squeeze()
    d = d.clamp(-100, 100)
    return d.pow(2).mean() - lam * d.mean().pow(2)


def conf_depth_loss(pred_depth, log_conf, target, alpha=0.5, eps=1e-6, max_depth=float('inf')):
    """confidence-weighted depth loss, alpha prevents conf collapsing to 0"""
    mask = (target > eps) & (target < max_depth) & (pred_depth > eps) & torch.isfinite(pred_depth)
    if mask.sum() == 0:
        return pred_depth.new_zeros(1).squeeze()

    conf = log_conf.exp().clamp(min=1e-8, max=1e6)
    d = (torch.log(pred_depth.clamp(min=eps)) - torch.log(target.clamp(min=eps))).clamp(-100, 100)

    depth_term = (conf * d.abs())[mask].mean()
    conf_reg   = -alpha * torch.log(conf[mask]).mean()

    return depth_term + conf_reg


def build_loss(cfg):
    loss_cfg = cfg.get('losses', {'silog': 1.0})
    conf_alpha = loss_cfg.get('conf_alpha', 0.5)
    max_depth = float(cfg['max_depth']) if cfg.get('max_depth') else float('inf')

    def loss_fn(pred, target):
        depth    = pred[:, :1]
        log_conf = pred[:, 1:] if pred.shape[1] > 1 else None

        total = pred.new_zeros(1).squeeze()
        if loss_cfg.get('silog', 0.0) > 0:
            lam = cfg.get('silog_lam', 0.85)
            total = total + loss_cfg['silog'] * si_log_loss(depth, target, lam=lam, max_depth=max_depth)
        if loss_cfg.get('conf', 0.0) > 0 and log_conf is not None:
            total = total + loss_cfg['conf'] * conf_depth_loss(
                depth, log_conf, target, alpha=conf_alpha, max_depth=max_depth)
        return total

    return loss_fn
