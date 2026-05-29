import torch
from loss import silog_loss 

def run_epoch(loader, model, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss = 0.0
    scaler = torch.amp.GradScaler('cuda') if is_train else None
    
    for batch in loader:
        images = batch["image"].to(device)
        depths = batch["depth"].to(device)
        masks = batch["mask"].to(device)
        
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast('cuda'):
                preds = model(images)
                loss = silog_loss(preds, depths, masks)
            
            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)