# Binary Change Detection on EO-SAR Image Pairs (GalaxEye)

## Project Description
This repository contains an end-to-end deep learning pipeline for binary change detection on paired EO-SAR satellite imagery. Designed to run under extreme memory and time constraints, the pipeline utilizes an Early Fusion U-Net++ architecture with a ResNet-34 backbone, Automatic Mixed Precision (AMP), and a custom balanced BCE-Dice loss function to tackle severe class imbalance.

## Requirements
See `requirements.txt`.
- Python 3.10+
- PyTorch 2.0+
- segmentation-models-pytorch
- albumentations

## Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure
The dataloader expects the following structure:
```
/tmp/data/
├── train/
│   ├── time_1/
│   ├── time_2/
│   └── label/
├── val/
├── test/
```

## Training
Command to run training from scratch:
```bash
python train.py
```
*(Note: The script automatically loads hyperparameters from `config.yaml`)*

## Evaluation
Command to evaluate on test data by passing a path to the dataset and weights:
```bash
python eval.py --data_dir /tmp/data/test --weights ./weights/best_model.pth
```

## Model Weights
[GOOGLE DRIVE LINK TO best_model.pth](https://drive.google.com/file/d/12OOjo7q5kSAmKgzQPAeBIANh1FaIajzM/view?usp=sharing)

## Results (Test Split)
| Metric | Score |
|---|---|
| Optimal Threshold | 0.30 |
| Mean IoU | 0.0257 |
| F1 Score | 0.0501 |
| Precision | 0.0341 |
| Recall | 0.0945 |

## Citation / References
- Zhou, Z., et al. (2018). UNet++: A Nested U-Net Architecture for Medical Image Segmentation.
- Jadon, S. (2020). A survey of loss functions for semantic segmentation.
- segmentation_models.pytorch library (Yakubovskiy, 2019)
