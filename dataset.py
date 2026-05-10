import os
import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio
import albumentations as A

class ChangeDetectionDataset(Dataset):
    """Dataset class for loading EO and SAR image pairs with optional augmentation."""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.pre_event_dir = os.path.join(data_dir, 'pre-event')
        self.post_event_dir = os.path.join(data_dir, 'post-event')
        self.target_dir = os.path.join(data_dir, 'target')
        
        if not os.path.exists(self.pre_event_dir):
            raise FileNotFoundError(f"Could not find {self.pre_event_dir}")
        
        # Sort filenames to ensure consistency
        self.filenames = sorted([f for f in os.listdir(self.pre_event_dir) if f.endswith('.tif')])
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        
        # Load Pre-event image (EO)
        pre_path = os.path.join(self.pre_event_dir, filename)
        with rasterio.open(pre_path) as src:
            pre_img = src.read() # (C, H, W)
            pre_img = np.transpose(pre_img, (1, 2, 0)) # (H, W, C)
        
        # Load Post-event image (SAR)
        post_path = os.path.join(self.post_event_dir, filename)
        with rasterio.open(post_path) as src:
            post_img = src.read() # (C, H, W)
            post_img = np.transpose(post_img, (1, 2, 0)) # (H, W, C)
        
        # Load Target mask
        target_path = os.path.join(self.target_dir, filename)
        with rasterio.open(target_path) as src:
            target_img = src.read(1) # (H, W)
        
        # Remap labels: 0,1 (No-Change) -> 0; 2,3 (Change) -> 1
        target_remapped = np.zeros_like(target_img, dtype=np.float32)
        target_remapped[target_img >= 2] = 1.0

        # Apply Augmentations
        if self.transform:
            augmented = self.transform(
                image=pre_img, 
                image0=post_img, 
                mask=target_remapped
            )
            pre_img = augmented['image']
            post_img = augmented['image0']
            target_remapped = augmented['mask']

        # Back to (C, H, W) for PyTorch
        pre_img = np.transpose(pre_img, (2, 0, 1)).astype(np.float32)
        post_img = np.transpose(post_img, (2, 0, 1)).astype(np.float32)
        
        # Simple Min-Max Normalization
        if pre_img.max() > pre_img.min():
            pre_img = (pre_img - pre_img.min()) / (pre_img.max() - pre_img.min() + 1e-8)
        if post_img.max() > post_img.min():
            post_img = (post_img - post_img.min()) / (post_img.max() - post_img.min() + 1e-8)

        # Concatenate images to form (Channels_Total, H, W)
        input_tensor = torch.from_numpy(np.concatenate((pre_img, post_img), axis=0))
        target_tensor = torch.from_numpy(np.expand_dims(target_remapped, axis=0))
        
        return {
            'input': input_tensor, 
            'target': target_tensor, 
            'filename': filename
        }
