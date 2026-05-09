"""Script to inspect satellite TIFF datasets for GalaxEye assignment."""

import os
import sys

# Ensure user site-packages are accessible
sys.path.append(r'C:\Users\intel\AppData\Roaming\Python\Python312\site-packages')

import numpy as np
import rasterio

def inspect_data(base_path):
    """Iterate through the directory and print shapes and statistics of images."""
    pre_event_dir = os.path.join(base_path, 'pre-event')
    post_event_dir = os.path.join(base_path, 'post-event')
    target_dir = os.path.join(base_path, 'target')

    files = os.listdir(pre_event_dir)[:2]

    for f in files:
        print(f"File: {f}")
        with rasterio.open(os.path.join(pre_event_dir, f)) as src:
            pre_img = src.read()
        with rasterio.open(os.path.join(post_event_dir, f)) as src:
            post_img = src.read()
        with rasterio.open(os.path.join(target_dir, f)) as src:
            target_img = src.read()

        print(f"Pre-event: shape={pre_img.shape}, min={pre_img.min()}, max={pre_img.max()}")
        print(f"Post-event: shape={post_img.shape}, min={post_img.min()}, max={post_img.max()}")
        unique_vals = np.unique(target_img)
        print(f"Target: shape={target_img.shape}, unique={unique_vals}")
        print("-" * 50)

if __name__ == "__main__":
    inspect_data(r'd:\Mukul_Gupta_Galaxeye\train\train')
