# Dog Breed Classification

A comprehensive PyTorch-based deep learning pipeline for fine-grained visual categorization, classifying 70 dog breeds with advanced computer vision techniques.

## Project Overview

This project implements an end-to-end machine learning system for dog breed classification using multiple deep learning approaches. The system incorporates:

- Transfer learning with three different CNN architectures
- Model ensemble with weighted voting mechanisms
- Test-time augmentation for robust inference
- Comprehensive data augmentation and preprocessing
- Detailed performance analysis and visualization

The implemented system achieves 95.29% validation accuracy using ensemble techniques with test-time augmentation, significantly outperforming individual models.

## Technical Specifications

- **Framework**: PyTorch
- **Hardware Used**: NVIDIA GeForce RTX 4070 Laptop GPU
- **Key Models**: ResNet50, EfficientNet B0, DenseNet121
- **Final Accuracy**: 95.29% (with ensemble and TTA)
- **Base Model Accuracy**: 93.14% (ResNet50)

## Installation and Requirements

```bash
git clone https://github.com/uehlingeric/dog-breed-classification.git
cd dog-breed-classification

pip install -r requirements.txt
```

### Dependencies

```
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
seaborn==0.13.2
tqdm==4.66.4
torch==2.6.0+cu118
torchvision==0.21.0
Pillow==10.3.0
scikit-learn==1.4.2
```

## Dataset Information

The dataset consists of:
- 7,946 training images
- 700 validation images
- 200 test images
- 70 different dog breed classes

**Class Distribution:**
- Most common class: Shih-Tzu (198 samples)
- Least common class: American Hairless (65 samples)
- Imbalance ratio (max/min): 3.05
- Average samples per class: 113.51

**Image Properties:**
- Standard image size: 224×224 pixels
- Average file size: 23.60 KB

## Pipeline Overview

The pipeline implements a complete deep learning workflow:

### 1. Data Preparation and Analysis
- Custom data loading with statistical analysis of class distribution
- Calculation of class imbalance metrics (imbalance ratio: 3.05)
- Image property analysis for dimensions, file sizes, and integrity
- File path validation with automated missing file detection

### 2. Data Processing and Engineering
- Implementation of PyTorch `Dataset` and `DataLoader` classes with custom transformations
- Data augmentation pipeline with RandomResizedCrop, RandomRotation, and ColorJitter
- Normalization using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Efficient batch processing with pin_memory and multiprocessing

### 3. Model Architecture and Transfer Learning
- Implementation of multiple CNN architectures (ResNet50, EfficientNet B0, DenseNet121)
- Transfer learning with pretrained weights and selective layer freezing
- Custom classifier heads optimized for the 70 dog breed classes
- Parameter efficiency analysis (90.43 MB model size, 23.6M parameters)

### 4. Training Pipeline and Optimization
- Learning rate finder with visualization for optimal convergence
- Custom training loop with early stopping and plateau detection
- Comprehensive checkpointing system with training history tracking
- Dynamic error analysis by class with confusion matrix generation

### 5. Advanced Techniques
- Weighted ensemble implementation with model-specific class weights
- Test-time augmentation system with multiple transformation types
- Class activation mapping for model interpretability
- Performance profiling (217.53 FPS inference speed, 4.60ms per image)

## Model Performance

### Base ResNet50 Model
- Validation accuracy: 93.14%
- Training time: ~8.5 minutes (15 epochs with early stopping)
- Inference speed: 217.53 frames per second (4.60ms per image)
- Model size: 90.43 MB

### Ensemble Model with TTA
- Validation accuracy: 95.29%
- Significant improvements for challenging breeds:
  - Scotch Terrier: +40% accuracy
  - Shih-Tzu: +20% accuracy
  - American Hairless: +20% accuracy

## Error Analysis

The model struggles most with:
1. Visually similar breeds (e.g., Boston Terrier vs. Bulldog)
2. Breeds with high variation in appearance
3. Breeds with limited training samples

## Technical Implementation Details

### Transfer Learning
- Freezing convolutional base layers to retain learned feature extraction capabilities
- Replacing classification layers with custom heads tailored to the 70 dog breeds
- Selectively unfreezing later layers during training to fine-tune high-level feature representations
- Using different model architectures (ResNet50, EfficientNet B0, DenseNet121) to capture diverse feature patterns

### Ensemble Method
- Combines predictions from multiple complementary architectures (ResNet, EfficientNet, DenseNet)
- Assigns architecture-specific weights based on validation performance on per-class basis
- Implements a weighted voting system that accounts for model confidence
- Balances strengths of different architectures (ResNet's depth, EfficientNet's efficiency, DenseNet's feature reuse)

### Test-Time Augmentation (TTA)
- Generating 5 variations of each test image (original plus 4 augmented versions)
- Creating versions with horizontal flips, slight rotations, and brightness adjustments
- Averaging predictions across all variations to reduce sensitivity to image orientation
- Demonstrating dramatic improvement for difficult classes (40% gain for Scotch Terrier)

### Data Augmentation
- RandomResizedCrop: Handles size variations while maintaining content integrity
- RandomHorizontalFlip: Improves generalization for orientation-invariant features
- RandomRotation: Builds resilience to different angles and positioning
- ColorJitter: Creates robustness to variations in lighting and color balance

### Training Optimization
- Learning rate finder implementation for optimal convergence parameters
- ReduceLROnPlateau scheduling to dynamically adjust learning rates
- Early stopping to prevent overfitting while maximizing performance
- Gradient accumulation for stable updates with challenging classes
- Strategic weight freezing/unfreezing during different training phases

### Error Analysis and Visualization
- Per-class confusion matrix to identify systematic misclassifications
- Class activation mapping (CAM) to visualize regions of interest in classification
- Fine-grained error rate analysis by breed characteristics
- Targeted performance improvements for most-confused breed pairs

## Technologies and Implementation Details

- **PyTorch**: Implementation of neural networks, custom datasets, optimizers, and loss functions
- **Computer Vision Techniques**: Implementation of transfer learning from pretrained networks, with selective layer freezing and fine-tuning
- **Model Architecture**: Integration of multiple CNN architectures (ResNet50, EfficientNet B0, DenseNet121) with custom classifier heads
- **Data Pipeline**: Custom augmentation strategies, normalization, and batch processing
- **Training Optimization**: Learning rate scheduling, early stopping, and gradient optimization techniques
- **Model Ensembling**: Weighted averaging of multiple model outputs with class-specific confidence weighting
- **Evaluation Methods**: Confusion matrix analysis, precision-recall metrics, and error pattern identification
- **Visualization**: Activation mapping and performance metric plotting

## License
This project is licensed under the MIT License - see the LICENSE file for details.