import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A

# Internal imports
from dataset import ChangeDetectionDataset
from model import get_model

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=20.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.register_buffer('pos_weight_tensor', torch.tensor([pos_weight]))
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_tensor)
        
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs)
        smooth = 1.0
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1.0),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ], additional_targets={'image0': 'image'})

def train():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = ChangeDetectionDataset(config['data']['train_dir'], transform=get_train_transforms())
    val_dataset = ChangeDetectionDataset(config['data']['val_dir'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    
    model = get_model(config, in_channels=4).to(device)
    
    criterion = BCEDiceLoss(
        bce_weight=config['loss']['bce_weight'],
        dice_weight=config['loss']['dice_weight'],
        pos_weight=config['loss']['pos_weight']
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    # MIXED PRECISION SCALER (For speed and memory efficiency)
    scaler = torch.amp.GradScaler('cuda')
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=config['training']['learning_rate'] * 10, 
        epochs=config['training']['epochs'],
        steps_per_epoch=len(train_loader)
    )
    
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    epochs = config['training']['epochs']
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            inputs, targets = batch['input'].to(device), batch['target'].to(device)
            
            optimizer.zero_grad()
            
            # Use Automatic Mixed Precision
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
            
        train_loss = train_loss / len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch['input'].to(device), batch['target'].to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss = val_loss / len(val_dataset)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("Saved new best model.")
            
    print("ULTIMATE MAX-POWER TRAINING COMPLETE.")

if __name__ == "__main__":
    train()
