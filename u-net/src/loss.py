import torch

def silog_loss(pred, target, mask, lambda_=0.5, eps=1e-6):
    """
    Scale-Invariant Log RMSE (SILog)

    pred, target: [B,1,H,W]
    mask:         [B,1,H,W] (1 = valid, 0 = ignore)
    """
    # keep only valid pixels
    pred = pred[mask > 0]
    target = target[mask > 0]
    
    # avoid log(0)
    pred = torch.clamp(pred, min=eps)
    target = torch.clamp(target, min=eps)
    
    log_diff = torch.log(pred) - torch.log(target)
    
    mse = torch.mean(log_diff ** 2)
    mean = torch.mean(log_diff)
    
    loss = mse - lambda_ * (mean ** 2)
    return loss