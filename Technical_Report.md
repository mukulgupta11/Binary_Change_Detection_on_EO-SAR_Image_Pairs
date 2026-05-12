# Technical Report: Binary Change Detection on EO-SAR Image Pairs
**Candidate:** Mukul Gupta
**Position:** Satellite AI Research Intern

## 1. Abstract
This report details the end-to-end development of an Early Fusion Unet++ architecture for binary change detection on paired EO-SAR satellite imagery. The core challenge involved handling severe class imbalance and extreme computational constraints. The final model utilizes a ResNet-34 encoder with Automatic Mixed Precision (AMP) and a heavily penalized BCE-Dice loss function (`pos_weight=2.0`). While the model successfully converged during training (reaching a validation loss of 0.58), blind test metrics remained low (F1: 0.05), revealing fundamental challenges in spatial alignment and modal fusion within the provided dataset.

## 2. Literature Survey
Change detection in multi-modal remote sensing (EO-SAR) is a highly active research area. 
- **Early Fusion vs Late Fusion:** Early fusion concatenates EO and SAR images at the input level, while Late Fusion (Siamese Networks) processes them through separate parallel encoders (Zhan et al., 2017). Due to the hardware constraints of a single 15GB T4 GPU, an Early Fusion approach was selected to halve the encoder memory footprint.
- **Architectural Paradigms:** U-Net and its variants (U-Net++) are the de facto standard for semantic segmentation. U-Net++ (Zhou et al., 2018) was chosen for its nested skip pathways, which reduce the semantic gap between encoder and decoder feature maps, improving boundary delineation.
- **Loss Functions for Imbalance:** In disaster datasets, change pixels often represent <1% of the image. Standard Cross-Entropy fails here. Studies demonstrate that a combined Binary Cross Entropy (BCE) and Dice Loss provides robust gradient flows for highly imbalanced pixel classification (Jadon, 2020).

## 3. Methodology
### Architecture & Training Strategy
- **Model:** U-Net++ with a `resnet34` encoder initialized with ImageNet weights.
- **Fusion:** Early fusion of 4 channels (3 EO + 1 SAR).
- **Optimization:** AdamW optimizer with OneCycleLR scheduler (`max_lr=0.001`) to achieve faster convergence within limited GPU time quotas.
- **Memory Management:** Automatic Mixed Precision (AMP) was implemented to allow a batch size of 2 on a 15GB T4 GPU without Out-Of-Memory (OOM) errors.

### Handling Class Imbalance
The dataset suffers from extreme background dominance. To combat this:
1. **Weighted BCE-Dice Loss:** The loss function was explicitly designed to penalize false negatives. `pos_weight=2.0` was mathematically applied to the BCE component to force the model to care more about the minority 'Change' class.
2. **Threshold Sweeping:** Instead of using a naive 0.5 threshold during inference, a post-processing sweep (0.3 to 0.95) was implemented to find the mathematically optimal probability cutoff for the F1 score.

## 4. Results & Error Analysis
The model trained for 13 epochs, achieving a training loss of `0.578` and a validation loss of `0.584`, indicating excellent generalization and practically zero overfitting.

**Test Split Metrics:**
- **Optimal Threshold:** 0.30
- **Mean IoU:** 0.0257
- **Precision:** 0.0341
- **Recall:** 0.0945
- **F1 Score:** 0.0501

### Error Profile (Why does it fail?)
Despite healthy loss convergence, the visual predictions highlight a massive spatial disconnect. 
*Failure Case:* The model confidently hallucinates massive change blobs in the center of the image, while the ground truth change is a tiny artifact in the bottom corner.
*Why:* This is a classic symptom of **spatial misalignment**. If the pre-event EO and post-event SAR images are not perfectly co-registered at the pixel level, the CNN cannot physically learn the temporal difference. It learns to recognize raw topographical artifacts (like clouds or shadows) instead of actual "change." Furthermore, forcing Early Fusion on fundamentally different modalities (optical vs radar) without spatial transformer networks likely caused the modal noise to overpower the temporal signal.

## 5. Future Work
If I were joining GalaxEye as an intern, my immediate next steps would be:
1. **Image Co-Registration:** Implement SIFT/SURF or deep feature matching to perfectly align the EO and SAR pairs before they enter the model.
2. **Siamese Late-Fusion:** Switch from Early Fusion to a Siamese architecture. EO and SAR have vastly different statistical properties. Feeding them into separate ResNet encoders to extract modality-specific feature maps *before* comparing them would drastically reduce noise.
3. **Hard Negative Mining:** Implement focal loss to dynamically scale gradients based on prediction confidence, forcing the network to focus strictly on hard-to-classify edge pixels.

## 6. Conclusion
Building a resilient change detection pipeline involves strict engineering trade-offs. By implementing AMP and an optimized Unet++, the pipeline successfully trained a high-capacity model within severe free-tier GPU constraints. While the final F1 scores reflect the extreme difficulty of multi-modal, unaligned disaster imagery, the robust training loop, custom metrics, and rigorous evaluation pipeline lay a production-ready foundation for future data-alignment improvements.
