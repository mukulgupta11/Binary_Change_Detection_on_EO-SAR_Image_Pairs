import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ChangeDetectionDataset
from model import get_model
from tqdm import tqdm
import sys

# Optional: Add script paths since they were installed to user site-packages
sys.path.append(r'C:\Users\intel\AppData\Roaming\Python\Python312\Scripts')
sys.path.append(r'C:\Users\intel\AppData\Roaming\Python\Python312\site-packages')

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        # Use pos_weight to handle class imbalance
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        
        inputs = torch.sigmoid(inputs)
        smooth = 1.0
        
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

def train():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset and Dataloaders
    train_dataset = ChangeDetectionDataset(config['data']['train_dir'])
    val_dataset = ChangeDetectionDataset(config['data']['val_dir'])
    
    # Check input channels from first sample
    sample = train_dataset[0]
    in_channels = sample['input'].shape[0]
    print(f"Detected {in_channels} input channels.")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Model
    model = get_model(config, in_channels=in_channels).to(device)
    
    # Loss and Optimizer
    criterion = BCEDiceLoss(
        bce_weight=config['loss']['bce_weight'],
        dice_weight=config['loss']['dice_weight'],
        pos_weight=config['loss']['pos_weight']
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
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
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss = train_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in pbar_val:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                pbar_val.set_postfix({'val_loss': loss.item()})
                
        val_loss = val_loss / len(val_dataset)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("Saved new best model.")
            
    print("Training completed.")

if __name__ == "__main__":
    train()
