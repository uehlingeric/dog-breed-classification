# Dog Breed Classification Dataset

## Acquisition

The training dataset is proprietary and large (446 MB). To reproduce the project:

1. Source the dataset from the original distribution (if you have access).
2. Populate the following structure:

```
data/
├── raw/                    # Original raw data (not committed)
└── train_val.csv           # Train/validation split (not committed)
```

Then update `dogs_data/` directory structure with:

```
dogs_data/
├── train/                  # 7,946 training images, 70 breed subdirectories
├── valid/                  # 700 validation images
├── test/                   # 200 test images
├── train_val.csv           # Dataset split metadata
└── test.csv                # Test set metadata
```

## Data Properties

- **Training samples**: 7,946 images
- **Validation samples**: 700 images
- **Test samples**: 200 images
- **Classes**: 70 dog breeds
- **Image size**: 224×224 pixels (normalized)
- **Class imbalance ratio**: 3.05 (max/min)
- **Average samples per class**: 113.51

## Preprocessing

Images are normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) and augmented during training with RandomResizedCrop, RandomRotation, and ColorJitter transformations.
