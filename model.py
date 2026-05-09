"""Model definition for the Binary Change Detection assignment."""

import torch.nn as nn
import segmentation_models_pytorch as smp

class ChangeDetectionModel(nn.Module):
    """Early Fusion U-Net architecture for EO-SAR change detection."""
    
    def __init__(self, architecture="unet", encoder_name="resnet34", 
                 encoder_weights="imagenet", in_channels=6, classes=1):
        """Initialise the segmentation model dynamically."""
        super(ChangeDetectionModel, self).__init__()

        if architecture.lower() == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
            )
        else:
            raise ValueError(f"Architecture {architecture} not supported.")

    def forward(self, x):
        """Forward pass. Assumes x is (B, C, H, W)."""
        return self.model(x)

def get_model(config, in_channels):
    """Instantiate the model from a configuration dictionary."""
    model_cfg = config.get('model', {})
    architecture = model_cfg.get('architecture', 'unet')
    encoder_name = model_cfg.get('encoder_name', 'resnet34')
    encoder_weights = model_cfg.get('encoder_weights', 'imagenet')
    classes = model_cfg.get('classes', 1)

    return ChangeDetectionModel(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes
    )
