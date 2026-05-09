# Binary Change Detection — EO-SAR Image Pairs

This repository contains the codebase for the GalaxEye Space Satellite AI Research Intern Assignment. 

## Project Title & Description
**Binary Change Detection using EO-SAR Image Fusion**
This project performs binary change detection on paired Electro-Optical (EO) and Synthetic Aperture Radar (SAR) imagery. Given pre-event and post-event image pairs, the model produces a binary pixel-level change mask where 1 indicates change and 0 indicates no-change. The model utilizes an Early Fusion U-Net architecture to capture both spatial context and multimodal features.

## Requirements
- Python 3.8+
- The required dependencies are listed in `requirements.txt`.
You can install them using:
```bash
pip install -r requirements.txt
```

## Environment Setup
Create and activate a virtual environment before installing the requirements.
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Activate virtual environment (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure
Place the dataset in the project root directory such that it matches the paths configured in `config.yaml`.
```text
project_root/
├── train/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
├── test/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
├── dataset.py
├── model.py
├── train.py
├── eval.py
└── config.yaml
```

## Training
To train the model from scratch, run:
```bash
python train.py
```
*Note: Make sure your `config.yaml` is correctly configured with the data paths and desired hyperparameters.*

## Evaluation
To evaluate the model on the test or validation data, run:
```bash
python eval.py --data_path ./test --weights ./weights/best_model.pth --config config.yaml
```

## Model Weights
The trained model weights can be downloaded here:
[Public Link to Checkpoint] (Please insert Google Drive or HuggingFace link here before submitting)

## Results
*To be filled after running eval script on val and test splits.*

| Metric | Validation Split | Test Split |
|--------|------------------|------------|
| IoU    | TBD              | TBD        |
| Precision | TBD           | TBD        |
| Recall | TBD              | TBD        |
| F1 Score | TBD            | TBD        |

## Citation / References
- `segmentation_models_pytorch`: https://github.com/qubvel/segmentation_models.pytorch
- PyTorch: https://pytorch.org/
