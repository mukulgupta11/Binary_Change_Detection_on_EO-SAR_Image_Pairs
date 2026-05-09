"""Dataset module for loading EO and SAR image pairs."""
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio

class ChangeDetectionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Directory with 'pre-event', 'post-event', and 'target' subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.pre_event_dir = os.path.join(data_dir, 'pre-event')
        self.post_event_dir = os.path.join(data_dir, 'post-event')
        self.target_dir = os.path.join(data_dir, 'target')
        
        # Verify directories exist
        if not os.path.exists(self.pre_event_dir):
            raise FileNotFoundError(f"Could not find {self.pre_event_dir}")
        self.filenames = [f for f in os.listdir(self.pre_event_dir) if f.endswith('.tif')]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        
        # Load Pre-event image (EO)
        pre_path = os.path.join(self.pre_event_dir, filename)
        with rasterio.open(pre_path) as src:
            pre_img = src.read() # Shape: (C, H, W)
        
        # Load Post-event image (SAR)
        post_path = os.path.join(self.post_event_dir, filename)
        with rasterio.open(post_path) as src:
            post_img = src.read() # Shape: (C, H, W)
        
        # Load Target mask
        target_path = os.path.join(self.target_dir, filename)
        with rasterio.open(target_path) as src:
            target_img = src.read(1) # Shape: (H, W)
        
        # Ensure images are floats for neural network
        pre_img = pre_img.astype(np.float32)
        post_img = post_img.astype(np.float32)
        
        # Normalize (Basic Min-Max normalization per image)
        # In a real scenario, global dataset statistics are preferred, but this is a solid baseline.
        if pre_img.max() > pre_img.min():
            pre_img = (pre_img - pre_img.min()) / (pre_img.max() - pre_img.min() + 1e-8)
        if post_img.max() > post_img.min():
            post_img = (post_img - post_img.min()) / (post_img.max() - post_img.min() + 1e-8)

        # Concatenate pre and post event images to form a single input tensor
        input_tensor = np.concatenate((pre_img, post_img), axis=0)
        
        # Remap labels:
        # Original: 0 (Background), 1 (Intact), 2 (Damaged), 3 (Destroyed)
        # Remapped: 0 (No-Change), 1 (Change)
        # Thus, 0 and 1 -> 0; 2 and 3 -> 1
        target_remapped = np.zeros_like(target_img, dtype=np.float32)
        target_remapped[target_img >= 2] = 1.0
        
        # Add channel dimension to target: (1, H, W)
        target_remapped = np.expand_dims(target_remapped, axis=0)

        input_tensor = torch.from_numpy(input_tensor)
        target_tensor = torch.from_numpy(target_remapped)
        
        sample = {'input': input_tensor, 'target': target_tensor, 'filename': filename}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
