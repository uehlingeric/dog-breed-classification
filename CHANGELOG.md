# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-11

### Added
- Transfer learning pipeline with ResNet50, EfficientNet B0, and DenseNet121 architectures
- Model ensemble with weighted voting for improved accuracy (95.29% vs 93.14% baseline)
- Test-time augmentation system for robust inference
- Comprehensive data preprocessing with augmentation (RandomResizedCrop, RandomRotation, ColorJitter)
- Learning rate finder and early stopping mechanisms
- Class activation mapping for model interpretability
- Confusion matrix and per-class error analysis

### Details
- Achieves 95.29% validation accuracy with ensemble and TTA across 70 dog breeds
- 7,946 training images; baseline ResNet50 reaches 93.14% accuracy
- Inference speed: 217.53 FPS (4.60 ms per image)
- Handles class imbalance (ratio 3.05) with weighted loss and stratified splitting
