import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import time
import copy
import pickle
from sklearn.metrics import confusion_matrix, classification_report


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    return device


def load_data_info(
    base_dir="dogs_data", train_val_csv="train_val.csv", test_csv="test.csv"
):
    train_val_path = os.path.join(base_dir, train_val_csv)
    train_val_df = pd.read_csv(train_val_path)

    train_df = train_val_df[train_val_df["data set"] == "train"]
    valid_df = train_val_df[train_val_df["data set"] == "valid"]

    test_path = os.path.join(base_dir, test_csv)
    test_df = pd.read_csv(test_path)

    classes = sorted(train_df["labels"].unique())

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Number of classes: {len(classes)}")

    return train_df, valid_df, test_df, classes


def get_class_distribution(df, label_col="labels"):
    return Counter(df[label_col])


def explore_image_properties(
    df, base_dir="dogs_data", filepath_col="filepaths", sample_size=100
):
    sample_df = df.sample(min(sample_size, len(df)))

    dimensions = []
    file_sizes = []

    for idx, row in tqdm(
        sample_df.iterrows(), total=len(sample_df), desc="Exploring images"
    ):
        img_path = os.path.join(base_dir, row[filepath_col])
        try:
            img = Image.open(img_path)
            dimensions.append(img.size)

            file_sizes.append(os.path.getsize(img_path))
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    properties = {
        "dimensions": dimensions,
        "file_sizes": file_sizes,
        "common_dims": Counter(dimensions).most_common(5),
        "avg_file_size": np.mean(file_sizes) / 1024,
        "min_file_size": min(file_sizes) / 1024,
        "max_file_size": max(file_sizes) / 1024,
    }

    return properties


def visualize_sample_images(
    df,
    base_dir="dogs_data",
    filepath_col="filepaths",
    label_col="labels",
    num_classes=5,
    samples_per_class=2,
):
    classes = df[label_col].unique()

    if num_classes < len(classes):
        selected_classes = np.random.choice(classes, num_classes, replace=False)
    else:
        selected_classes = classes

    fig, axes = plt.subplots(
        num_classes, samples_per_class, figsize=(samples_per_class * 3, num_classes * 3)
    )

    for i, cls in enumerate(selected_classes):
        cls_samples = df[df[label_col] == cls].sample(
            min(samples_per_class, len(df[df[label_col] == cls]))
        )

        for j, (_, row) in enumerate(cls_samples.iterrows()):
            img_path = os.path.join(base_dir, row[filepath_col])
            img = Image.open(img_path)

            if num_classes == 1:
                if samples_per_class == 1:
                    ax = axes
                else:
                    ax = axes[j]
            else:
                if samples_per_class == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]

            ax.imshow(img)
            ax.set_title(f"{cls}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def check_file_paths(
    df, base_dir="dogs_data", filepath_col="filepaths", sample_size=None
):
    if sample_size is not None:
        check_df = df.sample(min(sample_size, len(df)))
    else:
        check_df = df

    missing_files = []

    for idx, row in tqdm(
        check_df.iterrows(), total=len(check_df), desc="Checking files"
    ):
        file_path = os.path.join(base_dir, row[filepath_col])
        if not os.path.isfile(file_path):
            missing_files.append(file_path)

    results = {
        "total_checked": len(check_df),
        "missing_files": missing_files,
        "missing_count": len(missing_files),
        "missing_percent": (
            len(missing_files) / len(check_df) * 100 if len(check_df) > 0 else 0
        ),
    }

    print(f"Checked {results['total_checked']} files")
    print(
        f"Missing files: {results['missing_count']} ({results['missing_percent']:.2f}%)"
    )

    return results


def find_corrupted_images(
    df, base_dir="dogs_data", filepath_col="filepaths", sample_size=None
):
    if sample_size is not None:
        check_df = df.sample(min(sample_size, len(df)))
    else:
        check_df = df

    corrupted_files = []

    for idx, row in tqdm(
        check_df.iterrows(), total=len(check_df), desc="Checking images"
    ):
        file_path = os.path.join(base_dir, row[filepath_col])
        try:
            img = Image.open(file_path)
            img.verify()
        except Exception as e:
            corrupted_files.append((file_path, str(e)))

    print(f"Found {len(corrupted_files)} corrupted images out of {len(check_df)}")
    return corrupted_files


class DogBreedDataset(Dataset):
    def __init__(
        self,
        df,
        base_dir="dogs_data",
        filepath_col="filepaths",
        label_col="labels",
        classes=None,
        transform=None,
        is_test=False,
    ):
        self.df = df
        self.base_dir = base_dir
        self.filepath_col = filepath_col
        self.label_col = label_col
        self.classes = classes
        self.transform = transform
        self.is_test = is_test

        if classes is not None:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, self.df.iloc[idx][self.filepath_col])

        try:
            img = Image.open(img_path).convert("RGB")

            if self.transform:
                img = self.transform(img)

            if self.is_test:
                img_id = self.df.iloc[idx]["Id"] if "Id" in self.df.columns else idx
                return img, img_id

            label = self.df.iloc[idx][self.label_col]
            label_idx = self.class_to_idx[label]

            return img, label_idx

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = torch.zeros((3, 224, 224))
            return (img, -1) if not self.is_test else (img, -1)


def get_data_transforms(img_size=224, augment=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    val_transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize]
    )

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = val_transform

    return train_transform, val_transform


def create_dataloaders(
    train_df,
    valid_df,
    test_df=None,
    base_dir="dogs_data",
    batch_size=32,
    img_size=224,
    num_workers=4,
    filepath_col="filepaths",
    label_col="labels",
    augment=True,
):
    classes = sorted(train_df[label_col].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    train_transform, val_transform = get_data_transforms(img_size, augment)

    train_dataset = DogBreedDataset(
        train_df, base_dir, filepath_col, label_col, classes, train_transform
    )

    valid_dataset = DogBreedDataset(
        valid_df, base_dir, filepath_col, label_col, classes, val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = None
    if test_df is not None:
        test_dataset = DogBreedDataset(
            test_df,
            base_dir,
            filepath_col=filepath_col,
            label_col=None,
            classes=classes,
            transform=val_transform,
            is_test=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "idx_to_class": {idx: cls for cls, idx in class_to_idx.items()},
    }


def create_model(
    model_name="resnet50", num_classes=None, pretrained=True, freeze_base=True
):
    model_functions = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "vgg16": models.vgg16,
        "vgg19": models.vgg19,
        "densenet121": models.densenet121,
        "mobilenet_v2": models.mobilenet_v2,
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b1": models.efficientnet_b1,
    }

    if model_name not in model_functions:
        raise ValueError(
            f"Model {model_name} not supported. Choose from: {list(model_functions.keys())}"
        )

    weights = "DEFAULT" if pretrained else None
    model = model_functions[model_name](weights=weights)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    if "resnet" in model_name:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif "vgg" in model_name:
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif "densenet" in model_name:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif "mobilenet" in model_name:
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif "efficientnet" in model_name:
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def create_custom_cnn(num_classes, input_channels=3, dropout_rate=0.5):
    class CustomCNN(nn.Module):
        def __init__(self, num_classes, input_channels, dropout_rate):
            super(CustomCNN, self).__init__()

            self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(256)

            self.pool = nn.MaxPool2d(2, 2)

            self.dropout = nn.Dropout(dropout_rate)

            self.fc1 = nn.Linear(256 * 14 * 14, 512)
            self.fc_bn1 = nn.BatchNorm1d(512)
            self.fc2 = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = self.pool(F.relu(self.bn4(self.conv4(x))))

            x = x.view(-1, 256 * 14 * 14)

            x = self.dropout(F.relu(self.fc_bn1(self.fc1(x))))
            x = self.fc2(x)

            return x

    return CustomCNN(num_classes, input_channels, dropout_rate)


def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(
        f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)"
    )
    print(f"Total parameters: {total_params:,}")

    return trainable_params


def check_if_model_trained(checkpoint_path):
    return os.path.isfile(checkpoint_path)


def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    num_epochs=25,
    scheduler=None,
    device=None,
    early_stopping=None,
    checkpoint_path="model_data/best_model.pth",
):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    history_path = os.path.join(
        os.path.dirname(checkpoint_path),
        f"{os.path.basename(checkpoint_path).split('.')[0]}_history.pkl",
    )

    if device is None:
        device = get_device()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    no_improve_epochs = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                loader = dataloaders["train_loader"]
            else:
                model.eval()
                loader = dataloaders["valid_loader"]

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(loader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                if hasattr(model, "fc"):
                    num_classes = model.fc.out_features
                elif hasattr(model, "classifier") and isinstance(
                    model.classifier, nn.Linear
                ):
                    num_classes = model.classifier.out_features
                else:
                    num_classes = len(labels.unique())

                if torch.any(labels < 0) or torch.any(labels >= num_classes):
                    invalid_indices = torch.logical_or(
                        labels < 0, labels >= num_classes
                    )
                    invalid_count = torch.sum(invalid_indices).item()
                    if invalid_count > 0:
                        print(
                            f"Warning: Found {invalid_count} invalid labels. Fixing..."
                        )
                        labels = torch.clamp(labels, 0, num_classes - 1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    try:
                        loss = criterion(outputs, labels)
                    except Exception as e:
                        print(f"Error computing loss: {e}")
                        print(
                            f"Labels shape: {labels.shape}, unique values: {torch.unique(labels).cpu().numpy()}"
                        )
                        print(f"Outputs shape: {outputs.shape}")
                        continue

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            dataset_size = len(loader.dataset)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if phase == "val":
                        scheduler.step(epoch_loss)
                else:
                    if phase == "train":
                        scheduler.step()

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                no_improve_epochs = 0

                torch.save(model.state_dict(), checkpoint_path)

                with open(history_path, "wb") as f:
                    pickle.dump(history, f)

                print(f"Saved best model with accuracy {epoch_acc:.4f}")
            elif phase == "val":
                no_improve_epochs += 1
                print(f"No improvement for {no_improve_epochs} epochs")

        if early_stopping is not None and no_improve_epochs >= early_stopping:
            print(
                f"Early stopping after {no_improve_epochs} epochs without improvement"
            )
            break

        print()

    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)

    return model, history


def load_model(model, checkpoint_path, device=None):
    if device is None:
        device = get_device()

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Model loaded from {checkpoint_path}")

    history_path = os.path.join(
        os.path.dirname(checkpoint_path),
        f"{os.path.basename(checkpoint_path).split('.')[0]}_history.pkl",
    )

    if os.path.isfile(history_path):
        with open(history_path, "rb") as f:
            history = pickle.load(f)
        print(f"Loaded training history with {len(history['train_loss'])} epochs")
    else:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        print("No training history found")

    return model, history


def train_or_load_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    num_epochs=25,
    scheduler=None,
    device=None,
    early_stopping=None,
    checkpoint_path="model_data/best_model.pth",
    force_train=False,
):
    if not force_train and check_if_model_trained(checkpoint_path):
        return load_model(model, checkpoint_path, device)
    else:
        return train_model(
            model,
            dataloaders,
            criterion,
            optimizer,
            num_epochs,
            scheduler,
            device,
            early_stopping,
            checkpoint_path,
        )


def evaluate_model(model, dataloader, criterion, device=None, classes=None):
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    if hasattr(model, "fc"):
        num_classes = model.fc.out_features
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        num_classes = model.classifier.out_features
    else:
        num_classes = len(classes) if classes is not None else 1000

    print(f"Model output classes: {num_classes}")

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            try:
                inputs = inputs.to(device)
                labels = labels.to(device)

                if torch.any(labels < 0) or torch.any(labels >= num_classes):
                    invalid_indices = torch.logical_or(
                        labels < 0, labels >= num_classes
                    )
                    invalid_count = torch.sum(invalid_indices).item()
                    print(f"Warning: Found {invalid_count} invalid labels. Fixing...")

                    labels = torch.clamp(labels, 0, num_classes - 1)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                try:
                    loss = criterion(outputs, labels)
                    batch_loss = loss.item() * inputs.size(0)
                except Exception as e:
                    print(f"Error computing loss: {e}")
                    print(
                        f"Labels shape: {labels.shape}, unique values: {torch.unique(labels).cpu().numpy()}"
                    )
                    print(f"Outputs shape: {outputs.shape}")
                    batch_loss = 0.0

                running_loss += batch_loss
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

    dataset_size = len(all_labels)
    if dataset_size == 0:
        print("Warning: No valid samples were processed!")
        return {
            "loss": float("nan"),
            "accuracy": 0.0,
            "predictions": [],
            "true_labels": [],
        }

    loss = running_loss / dataset_size
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_preds)

    report = None
    if classes is not None:
        try:
            report = classification_report(
                all_labels, all_preds, target_names=classes, output_dict=True
            )
            print(classification_report(all_labels, all_preds, target_names=classes))
        except Exception as e:
            print(f"Error generating classification report: {e}")

    return {
        "loss": loss,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": all_preds,
        "true_labels": all_labels,
    }


def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Training Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_class_distribution(class_counts, figsize=(12, 8), rot=90):
    plt.figure(figsize=figsize)

    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_counts)

    sns.barplot(x=list(classes), y=list(counts))
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=rot)
    plt.tight_layout()
    plt.show()


def visualize_augmentations(image_path, transform, num_augmentations=5):
    plt.figure(figsize=(15, 3))

    image = Image.open(image_path).convert("RGB")

    plt.subplot(1, num_augmentations + 1, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    for i in range(num_augmentations):
        augmented = transform(image)
        if isinstance(augmented, torch.Tensor):
            augmented = augmented.permute(1, 2, 0).cpu().numpy()
            augmented = np.clip(
                augmented * np.array([0.229, 0.224, 0.225])
                + np.array([0.485, 0.456, 0.406]),
                0,
                1,
            )

        plt.subplot(1, num_augmentations + 1, i + 2)
        plt.imshow(augmented)
        plt.title(f"Aug {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def predict_on_test_data(model, test_loader, device=None):
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for inputs, ids in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for id_val, pred in zip(ids, preds.cpu().numpy()):
                predictions.append((id_val, pred))

    return predictions


def create_submission_file(predictions, idx_to_class, output_path="submission.csv"):
    submission_data = []
    for id_val, pred in predictions:
        if hasattr(id_val, "item"):
            id_val = id_val.item()

        breed = f"'{idx_to_class[pred]}'"

        submission_data.append((id_val, breed))

    submission_df = pd.DataFrame(submission_data, columns=["Id", "labels"])

    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")

    return submission_df


def create_ensemble_model(models, device=None):
    if device is None:
        device = get_device()

    for model in models:
        model.to(device)
        model.eval()

    def ensemble_predict(inputs):
        all_preds = []

        with torch.no_grad():
            for model in models:
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                all_preds.append(probs)

            avg_preds = torch.mean(torch.stack(all_preds), dim=0)

            return avg_preds

    return ensemble_predict


def predict_with_ensemble(ensemble_predict, dataloader, device=None):
    if device is None:
        device = get_device()

    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Predicting with ensemble"):
        inputs = inputs.to(device)

        probs = ensemble_predict(inputs)
        _, preds = torch.max(probs, 1)

        all_preds.extend(preds.cpu().numpy())

        if not isinstance(labels, torch.Tensor):
            continue

        all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels) if all_labels else None


def visualize_activation_maps(model, img_tensor, layer_name, device=None):
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(device)

    activations = {}

    def hook_fn(module, input, output):
        activations[layer_name] = output.detach()

    for name, module in model.named_modules():
        if name == layer_name:
            handle = module.register_forward_hook(hook_fn)
            break
    else:
        print(f"Layer {layer_name} not found in model")
        return

    with torch.no_grad():
        output = model(img_tensor)

    handle.remove()

    activations = activations[layer_name]

    img = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = np.clip(
        img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1
    )

    fig, axes = plt.subplots(1, min(5, activations.size(1) + 1), figsize=(15, 4))

    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis("off")

    for i in range(min(4, activations.size(1))):
        axes[i + 1].imshow(activations[0, i].cpu().numpy(), cmap="viridis")
        axes[i + 1].set_title(f"Filter {i+1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


def visualize_class_activation_maps(
    model, img_tensor, class_idx, target_layer_name="layer4", device=None
):
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()
    img_tensor = (
        img_tensor.unsqueeze(0).to(device)
        if img_tensor.dim() == 3
        else img_tensor.to(device)
    )

    print("Model structure:")
    for name, module in model.named_modules():
        if name and (target_layer_name in name or not target_layer_name):
            print(f"Found layer: {name} (type: {type(module).__name__})")

    features = None

    try:
        with torch.no_grad():  # Add this to prevent gradient tracking
            x = model.conv1(img_tensor)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            features = model.layer4(x)

            x = model.avgpool(features)
            x = torch.flatten(x, 1)
            output = model.fc(x)

            weights = model.fc.weight[class_idx].view(-1, 1, 1)

            cam = torch.sum(weights * features, dim=1)

            cam = F.relu(cam)

            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

            cam = F.interpolate(
                cam.unsqueeze(0),
                size=(img_tensor.size(2), img_tensor.size(3)),
                mode="bilinear",
                align_corners=False,
            )
            cam = cam.squeeze().cpu().detach().numpy()  # Use detach() before numpy()

            img = (
                img_tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
            )  # Use detach() here too
            img = np.clip(
                img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]),
                0,
                1,
            )

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title("Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(cam, cmap="jet")
            plt.title("Class Activation Map")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(img)
            plt.imshow(cam, cmap="jet", alpha=0.5)
            plt.title("Overlay")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error generating CAM: {e}")
        print("This function is designed specifically for ResNet architecture.")
        print("For other architectures, you'll need to adapt the code.")

        if features is None:
            print(
                "Failed to extract features. Model architecture may not match expectations."
            )


def analyze_class_imbalance(df, label_col="labels"):
    class_counts = df[label_col].value_counts()

    total_samples = len(df)
    num_classes = len(class_counts)
    min_count = class_counts.min()
    max_count = class_counts.max()
    imbalance_ratio = max_count / min_count
    mean_count = class_counts.mean()
    std_count = class_counts.std()

    metrics = {
        "total_samples": total_samples,
        "num_classes": num_classes,
        "min_count": min_count,
        "max_count": max_count,
        "imbalance_ratio": imbalance_ratio,
        "mean_count": mean_count,
        "std_count": std_count,
        "class_counts": class_counts,
    }

    print(f"Total samples: {total_samples}")
    print(f"Number of classes: {num_classes}")
    print(f"Minimum samples per class: {min_count}")
    print(f"Maximum samples per class: {max_count}")
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    print(f"Mean samples per class: {mean_count:.2f}")
    print(f"Standard deviation: {std_count:.2f}")

    return metrics


def calculate_class_weights(df, label_col="labels", method="inverse"):
    class_counts = df[label_col].value_counts().sort_index()
    classes = class_counts.index.tolist()
    counts = class_counts.values

    if method == "inverse":
        weights = [1.0 / count for count in counts]
    elif method == "sqrt_inverse":
        weights = [1.0 / np.sqrt(count) for count in counts]
    elif method == "balanced":
        n_samples = len(df)
        n_classes = len(class_counts)
        weights = [n_samples / (n_classes * count) for count in counts]
    else:
        raise ValueError(
            f"Method {method} not supported. Choose from 'inverse', 'sqrt_inverse', or 'balanced'"
        )

    total_weight = sum(weights)
    weights = [weight / total_weight * len(weights) for weight in weights]

    class_weights = {cls: weight for cls, weight in zip(classes, weights)}

    tensor_weights = torch.FloatTensor(weights)

    return {"class_weights": class_weights, "tensor_weights": tensor_weights}


def create_balanced_sampler(dataset, labels):
    class_counts = Counter(labels)

    weights = [1.0 / class_counts[label] for label in labels]

    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=weights, num_samples=len(dataset), replacement=True
    )

    return sampler


def save_checkpoint(
    model, optimizer, scheduler, epoch, best_acc, history, filename="checkpoint.pth"
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_acc": best_acc,
        "history": history,
    }

    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(
    model, optimizer, scheduler, filename="checkpoint.pth", device=None
):
    if device is None:
        device = get_device()

    if not os.path.isfile(filename):
        print(f"Checkpoint {filename} not found")
        return None

    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint["epoch"],
        "best_acc": checkpoint["best_acc"],
        "history": checkpoint["history"],
    }


def create_learning_rate_finder(model, train_loader, criterion, optimizer, device=None):
    if device is None:
        device = get_device()

    model = model.to(device)
    model.train()

    def find_learning_rate(
        start_lr=1e-7, end_lr=10, num_iterations=100, step_mode="exp"
    ):
        if step_mode == "exp":
            lr_schedule = np.geomspace(start_lr, end_lr, num_iterations)
        else:
            lr_schedule = np.linspace(start_lr, end_lr, num_iterations)

        learning_rates = []
        losses = []

        original_params = copy.deepcopy(model.state_dict())

        for param_group in optimizer.param_groups:
            param_group["lr"] = start_lr

        iteration = 0
        for inputs, labels in tqdm(train_loader, desc="Finding learning rate"):
            if iteration >= num_iterations:
                break

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            for param_group in optimizer.param_groups:
                current_lr = param_group["lr"]

            learning_rates.append(current_lr)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

            if iteration < num_iterations - 1:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_schedule[iteration + 1]

            iteration += 1

        model.load_state_dict(original_params)

        return {"learning_rates": learning_rates, "losses": losses}

    def plot_lr_finder(lr_finder_results, skip_start=10, skip_end=5):
        learning_rates = lr_finder_results["learning_rates"]
        losses = lr_finder_results["losses"]

        if skip_start > 0:
            learning_rates = learning_rates[skip_start:]
            losses = losses[skip_start:]

        if skip_end > 0:
            learning_rates = learning_rates[:-skip_end]
            losses = losses[:-skip_end]

        plt.figure(figsize=(10, 6))
        plt.plot(learning_rates, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.grid(True)
        plt.show()

    return find_learning_rate, plot_lr_finder


def analyze_errors(model, dataloader, classes, device=None, num_samples=10):
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    errors = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Analyzing errors"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            error_mask = preds != labels
            error_indices = torch.nonzero(error_mask).squeeze()

            if error_indices.ndim == 0 and error_indices.nelement() > 0:
                error_indices = [error_indices.item()]
            elif error_indices.ndim >= 1:
                error_indices = error_indices.cpu().numpy()
            else:
                error_indices = []

            for idx in error_indices:
                errors.append(
                    {
                        "image": inputs[idx].cpu(),
                        "true_label": labels[idx].item(),
                        "pred_label": preds[idx].item(),
                        "prob": F.softmax(outputs[idx], dim=0).cpu().numpy(),
                    }
                )

    print(f"Found {len(errors)} errors in {len(dataloader.dataset)} samples")
    print(f"Error rate: {len(errors) / len(dataloader.dataset):.4f}")

    error_counts = {}
    confusion_pairs = Counter()

    for error in errors:
        true_cls = classes[error["true_label"]]
        pred_cls = classes[error["pred_label"]]

        if true_cls not in error_counts:
            error_counts[true_cls] = {"total": 0, "errors": 0}

        error_counts[true_cls]["errors"] += 1
        confusion_pairs[(true_cls, pred_cls)] += 1

    for batch in dataloader:
        labels = batch[1]
        for label in labels:
            cls = classes[label.item()]
            if cls not in error_counts:
                error_counts[cls] = {"total": 0, "errors": 0}
            error_counts[cls]["total"] += 1

    for cls in error_counts:
        if error_counts[cls]["total"] > 0:
            error_counts[cls]["error_rate"] = (
                error_counts[cls]["errors"] / error_counts[cls]["total"]
            )
        else:
            error_counts[cls]["error_rate"] = 0.0

    sorted_error_rates = sorted(
        [(cls, info["error_rate"]) for cls, info in error_counts.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    most_confused = confusion_pairs.most_common(10)

    if len(errors) > 0 and num_samples > 0:
        num_samples = min(num_samples, len(errors))
        plt.figure(figsize=(15, 3 * (num_samples // 3 + 1)))

        for i in range(num_samples):
            plt.subplot(num_samples // 3 + 1, 3, i + 1)

            img = errors[i]["image"].permute(1, 2, 0).numpy()
            img = np.clip(
                img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]),
                0,
                1,
            )

            plt.imshow(img)

            true_label = classes[errors[i]["true_label"]]
            pred_label = classes[errors[i]["pred_label"]]

            plt.title(f"True: {true_label}\nPred: {pred_label}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    return {
        "num_errors": len(errors),
        "error_rate": len(errors)
        / sum(info["total"] for info in error_counts.values()),
        "error_counts": error_counts,
        "sorted_error_rates": sorted_error_rates,
        "most_confused": most_confused,
        "errors": errors,
    }


def plot_error_distribution(error_analysis):
    classes = []
    error_rates = []

    for cls, rate in error_analysis["sorted_error_rates"]:
        classes.append(cls)
        error_rates.append(rate)

    sorted_indices = np.argsort(error_rates)[::-1]
    classes = [classes[i] for i in sorted_indices]
    error_rates = [error_rates[i] for i in sorted_indices]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(classes)), error_rates)

    for i, bar in enumerate(bars):
        if error_rates[i] > 0.5:
            bar.set_color("red")
        elif error_rates[i] > 0.25:
            bar.set_color("orange")
        else:
            bar.set_color("green")

    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Error Rate")
    plt.title("Error Rate by Class")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def plot_confusion_pairs(error_analysis):
    true_classes = []
    pred_classes = []
    counts = []

    for (true_cls, pred_cls), count in error_analysis["most_confused"]:
        true_classes.append(true_cls)
        pred_classes.append(pred_cls)
        counts.append(count)

    plt.figure(figsize=(12, 8))
    y_pos = range(len(true_classes))

    plt.barh(y_pos, counts)
    plt.yticks(
        y_pos,
        [
            f"{true_cls} → {pred_cls}"
            for true_cls, pred_cls in zip(true_classes, pred_classes)
        ],
    )
    plt.xlabel("Count")
    plt.title("Most Common Confusion Pairs")
    plt.tight_layout()
    plt.show()


def estimate_training_time(
    model, dataloader, device=None, num_epochs=1, num_batches=10
):
    if device is None:
        device = get_device()

    model = model.to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    batch_times = []

    iterator = iter(dataloader)
    for _ in tqdm(
        range(min(num_batches, len(dataloader))), desc="Estimating training time"
    ):
        try:
            inputs, labels = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            inputs, labels = next(iterator)

        inputs = inputs.to(device)
        labels = labels.to(device)

        start_time = time.time()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_time = time.time() - start_time
        batch_times.append(batch_time)

    avg_batch_time = np.mean(batch_times)
    batches_per_epoch = len(dataloader)
    estimated_epoch_time = avg_batch_time * batches_per_epoch
    estimated_total_time = estimated_epoch_time * num_epochs

    print(f"Average batch time: {avg_batch_time:.4f} seconds")
    print(
        f"Estimated time per epoch: {estimated_epoch_time:.2f} seconds ({estimated_epoch_time / 60:.2f} minutes)"
    )
    print(
        f"Estimated total training time for {num_epochs} epochs: {estimated_total_time:.2f} seconds ({estimated_total_time / 60:.2f} minutes)"
    )

    return {
        "avg_batch_time": avg_batch_time,
        "batches_per_epoch": batches_per_epoch,
        "estimated_epoch_time": estimated_epoch_time,
        "estimated_total_time": estimated_total_time,
    }


def measure_inference_time(model, input_size, device=None, num_runs=100):
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(1, *input_size).to(device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    total_time = time.time() - start_time

    avg_time = total_time / num_runs
    fps = 1.0 / avg_time

    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Frames per second: {fps:.2f}")

    return {
        "avg_time": avg_time,
        "fps": fps,
        "total_time": total_time,
        "num_runs": num_runs,
    }


def summary(model, input_size):
    print("Model Structure:")
    print(model)
    print("\n")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")


def calculate_model_size(model):
    state_dict = model.state_dict()

    model_size_bytes = sum(
        param.nelement() * param.element_size() for param in state_dict.values()
    )
    model_size_mb = model_size_bytes / (1024 * 1024)

    print(f"Model size: {model_size_mb:.2f} MB")

    return model_size_mb


def freeze_layers(model, layers_to_freeze):
    for name, param in model.named_parameters():
        for layer in layers_to_freeze:
            if layer in name:
                param.requires_grad = False
                break

    return model


def unfreeze_layers(model, layers_to_unfreeze):
    for name, param in model.named_parameters():
        for layer in layers_to_unfreeze:
            if layer in name:
                param.requires_grad = True
                break

    return model


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def visualize_model_predictions(model, dataloader, classes, device, num_images=9):
    model.to(device)
    model.eval()

    # Get a batch of images
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]

    # Make predictions
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

    # Plot images with predictions
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i, (img, label, pred) in enumerate(zip(images, labels, preds)):
        # Convert image for display
        img = img.cpu().numpy().transpose((1, 2, 0))
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        # Get class names
        true_class = classes[label.item()]
        pred_class = classes[pred.item()]

        # Display image and predictions
        axes[i].imshow(img)
        title_color = "green" if label.item() == pred.item() else "red"
        axes[i].set_title(f"True: {true_class}\nPred: {pred_class}", color=title_color)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def perform_test_time_augmentation(model, img_tensor, num_augs=5, device=None):
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    # Make prediction on original image
    original_img = img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor
    original_img = original_img.to(device)

    with torch.no_grad():
        base_output = model(original_img)
        base_probs = F.softmax(base_output, dim=1)

    # Store all predictions, starting with the original
    all_probs = [base_probs]

    # Get mean and std for normalization/denormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    # First denormalize the image
    img_denorm = img_tensor * std + mean

    # Define simpler TTA transformations
    transforms_list = [
        transforms.functional.hflip,  # Horizontal flip
        lambda x: transforms.functional.rotate(x, 10),  # Slight right rotation
        lambda x: transforms.functional.rotate(x, -10),  # Slight left rotation
        lambda x: transforms.functional.adjust_brightness(x, 1.1),  # Slightly brighter
        lambda x: transforms.functional.adjust_brightness(x, 0.9),  # Slightly darker
    ]

    # Use only as many transforms as requested
    transforms_to_use = transforms_list[: min(num_augs, len(transforms_list))]

    # Apply each transform
    for transform_func in transforms_to_use:
        # Apply transform to denormalized image
        aug_img = transform_func(img_denorm)

        # Renormalize
        aug_img_norm = (aug_img - mean) / std

        # Get prediction
        with torch.no_grad():
            aug_output = model(aug_img_norm)
            aug_probs = F.softmax(aug_output, dim=1)
            all_probs.append(aug_probs)

    # Average all predictions
    avg_probs = torch.mean(torch.cat(all_probs, dim=0), dim=0, keepdim=True)

    return avg_probs


def calculate_model_class_weights(models, val_loader, classes, device=None):
    if device is None:
        device = get_device()

    model_class_weights = {}

    for i, model in enumerate(models):
        model.to(device)
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(
                val_loader, desc=f"Evaluating model {i+1}/{len(models)}"
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        class_accuracies = []
        for class_idx in range(len(classes)):
            class_mask = np.array(all_labels) == class_idx
            if np.sum(class_mask) > 0:
                class_acc = np.mean(np.array(all_preds)[class_mask] == class_idx)
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)

        class_weights = torch.tensor(class_accuracies, dtype=torch.float32).to(device)

        if class_weights.sum() > 0:
            class_weights = class_weights / class_weights.sum() * len(classes)

        model_class_weights[i] = class_weights

    return model_class_weights


def create_weighted_ensemble_model(models, class_weights=None, device=None):
    if device is None:
        device = get_device()

    for model in models:
        model.to(device)
        model.eval()

    def weighted_ensemble_predict(inputs, use_tta=False, num_tta=5):
        all_preds = []

        with torch.no_grad():
            for i, model in enumerate(models):
                if use_tta:
                    batch_probs = []
                    for img in inputs:
                        img_probs = perform_test_time_augmentation(
                            model, img, num_augs=num_tta, device=device
                        )
                        batch_probs.append(img_probs)
                    probs = torch.cat(batch_probs, dim=0)
                else:
                    outputs = model(inputs)
                    probs = F.softmax(outputs, dim=1)

                if class_weights is not None and i in class_weights:
                    weighted_probs = probs * class_weights[i].unsqueeze(0)
                    weighted_probs = weighted_probs / weighted_probs.sum(
                        dim=1, keepdim=True
                    )
                    all_preds.append(weighted_probs)
                else:
                    all_preds.append(probs)

            avg_preds = torch.mean(torch.stack(all_preds), dim=0)

            return avg_preds

    return weighted_ensemble_predict


def predict_with_tta_ensemble(
    ensemble_predict, dataloader, device=None, use_tta=True, num_tta=5
):
    if device is None:
        device = get_device()

    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Predicting with TTA ensemble"):
        inputs = inputs.to(device)

        probs = ensemble_predict(inputs, use_tta=use_tta, num_tta=num_tta)
        _, preds = torch.max(probs, 1)

        all_preds.extend(preds.cpu().numpy())

        if not isinstance(labels, torch.Tensor):
            continue

        all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels) if all_labels else None


def apply_tta_to_batch(model, inputs, num_augs=5, device=None):
    if device is None:
        device = get_device()

    batch_probs = []

    for img in inputs:
        img_probs = perform_test_time_augmentation(
            model, img, num_augs=num_augs, device=device
        )
        batch_probs.append(img_probs)

    return torch.cat(batch_probs, dim=0)


def run_complete_pipeline(
    base_dir="dogs_data",
    model_name="resnet50",
    batch_size=32,
    img_size=224,
    num_epochs=25,
    lr=0.001,
    early_stopping=5,
    checkpoint_dir="model_data",
    model_filename=None,
    force_train=False,
):

    device = get_device()
    set_seed()

    train_df, valid_df, test_df, classes = load_data_info(base_dir)

    dataloaders = create_dataloaders(
        train_df,
        valid_df,
        test_df,
        base_dir=base_dir,
        batch_size=batch_size,
        img_size=img_size,
    )

    if model_filename is None:
        model_filename = f"{model_name}_model.pth"

    checkpoint_path = os.path.join(checkpoint_dir, model_filename)

    model = create_model(
        model_name=model_name, num_classes=len(classes), pretrained=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )

    model, history = train_or_load_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        scheduler=scheduler,
        device=device,
        early_stopping=early_stopping,
        checkpoint_path=checkpoint_path,
        force_train=force_train,
    )

    eval_results = evaluate_model(
        model, dataloaders["valid_loader"], criterion, device=device, classes=classes
    )

    plot_training_history(history)

    test_predictions = None
    if dataloaders["test_loader"] is not None:
        test_predictions = predict_on_test_data(
            model, dataloaders["test_loader"], device=device
        )

        submission_df = create_submission_file(
            test_predictions,
            dataloaders["idx_to_class"],
            output_path=os.path.join(checkpoint_dir, "submission.csv"),
        )

    return {
        "model": model,
        "history": history,
        "eval_results": eval_results,
        "test_predictions": test_predictions,
        "dataloaders": dataloaders,
    }
