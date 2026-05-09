import os
import yaml
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from dataset import ChangeDetectionDataset
from model import get_model
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

def calculate_iou(preds, targets):
    intersection = np.logical_and(targets == 1, preds == 1).sum()
    union = np.logical_or(targets == 1, preds == 1).sum()
    if union == 0:
        return 0.0
    return intersection / union

def evaluate(config_path, data_dir, weights_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset and Dataloader
    dataset = ChangeDetectionDataset(data_dir)
    
    # Check input channels
    sample = dataset[0]
    in_channels = sample['input'].shape[0]
    
    loader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Model setup
    model = get_model(config, in_channels=in_channels).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print(f"Evaluating on {data_dir}...")
    with torch.no_grad():
        for batch in tqdm(loader):
            inputs = batch['input'].to(device)
            targets = batch['target'].cpu().numpy()
            
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            preds = (preds > 0.5).astype(np.uint8)
            
            all_preds.append(preds.flatten())
            all_targets.append(targets.flatten())
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Metrics
    iou = calculate_iou(all_preds, all_targets)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    print("\n--- Evaluation Results ---")
    print(f"IoU (Change class): {iou:.4f}")
    print(f"Precision (Change class): {precision:.4f}")
    print(f"Recall (Change class): {recall:.4f}")
    print(f"F1 Score (Change class): {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Change Detection Model")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test or val dataset directory')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    
    args = parser.parse_args()
    evaluate(args.config, args.data_dir, args.weights)
