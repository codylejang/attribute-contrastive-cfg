"""
Attribute Accuracy Evaluation

Trains a lightweight ResNet-18 classifier on CelebA for specific binary
attributes, then evaluates what fraction of generated samples exhibit
those attributes.

Usage:
    # Train classifier and evaluate all sample directories
    python evaluate_attributes.py \
        --data_root data/celeba-subset \
        --output_json outputs/hw4_results/attribute_accuracy.json

    # Evaluate only (skip training if classifier already exists)
    python evaluate_attributes.py \
        --data_root data/celeba-subset \
        --classifier_path outputs/hw4_results/eyeglasses_classifier.pt \
        --eval_only \
        --output_json outputs/hw4_results/attribute_accuracy.json
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image

from src.data import CelebADataset, CELEBA_ATTRIBUTES


# Eyeglasses is the focal attribute for our contrastive CFG experiments
EYEGLASSES_IDX = CELEBA_ATTRIBUTES.index("Eyeglasses")
MALE_IDX = CELEBA_ATTRIBUTES.index("Male")
SMILING_IDX = CELEBA_ATTRIBUTES.index("Smiling")
BLOND_HAIR_IDX = CELEBA_ATTRIBUTES.index("Blond_Hair")

TARGET_ATTRS = {
    "Eyeglasses": EYEGLASSES_IDX,
    "Male": MALE_IDX,
    "Smiling": SMILING_IDX,
    "Blond_Hair": BLOND_HAIR_IDX,
}


class ImageFolderDataset(Dataset):
    """Load PNG images from a directory for classification."""

    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def create_classifier(num_attrs: int = 4) -> nn.Module:
    """Create a multi-attribute ResNet-18 classifier."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_attrs)
    return model


def train_classifier(data_root: str, save_path: str, device: torch.device,
                     num_epochs: int = 5, batch_size: int = 128, lr: float = 1e-3):
    """Train multi-attribute classifier on CelebA."""
    print("=" * 60)
    print("Training attribute classifier on CelebA")
    print("=" * 60)

    # ImageNet normalization for pretrained ResNet
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load CelebA with labels
    dataset = CelebADataset(
        root=data_root, split="train", image_size=64,
        augment=False, return_labels=True, from_hub=True,
    )

    # Wrap to apply ImageNet normalization instead of [-1,1]
    class RenormalizedDataset(Dataset):
        def __init__(self, base_dataset, transform):
            self.base = base_dataset
            self.transform = transform
            # Override the base dataset's transform to only convert to tensor
            self.base.transform = transforms.Compose([transforms.ToTensor()])

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, labels = self.base[idx]
            # img is already a tensor in [0,1] from ToTensor()
            # Apply ImageNet normalization
            img = transforms.functional.normalize(
                img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            # Extract target attributes
            target = torch.tensor([labels[TARGET_ATTRS[a]].item() for a in TARGET_ATTRS])
            return img, target

    train_dataset = RenormalizedDataset(dataset, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)

    model = create_classifier(num_attrs=len(TARGET_ATTRS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (logits > 0).float()
            correct += (preds == labels).float().sum().item()
            total += labels.numel()

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        avg_loss = total_loss / len(train_dataset)
        avg_acc = correct / total
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved classifier to {save_path}")

    return model


def evaluate_on_celeba(model: nn.Module, data_root: str, device: torch.device,
                       batch_size: int = 128) -> dict:
    """Evaluate classifier accuracy on CelebA validation split."""
    print("\nEvaluating on CelebA validation split...")

    dataset = CelebADataset(
        root=data_root, split="train", image_size=64,
        augment=False, return_labels=True, from_hub=True,
    )

    class RenormalizedDataset(Dataset):
        def __init__(self, base_dataset):
            self.base = base_dataset
            self.base.transform = transforms.Compose([transforms.ToTensor()])

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            img, labels = self.base[idx]
            img = transforms.functional.normalize(
                img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            target = torch.tensor([labels[TARGET_ATTRS[a]].item() for a in TARGET_ATTRS])
            return img, target

    val_dataset = RenormalizedDataset(dataset)
    # Use last 5000 samples as a held-out eval set
    indices = list(range(len(val_dataset) - 5000, len(val_dataset)))
    val_subset = torch.utils.data.Subset(val_dataset, indices)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    model.eval()
    per_attr_correct = {a: 0 for a in TARGET_ATTRS}
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = (logits > 0).float()

            for i, attr in enumerate(TARGET_ATTRS):
                per_attr_correct[attr] += (preds[:, i] == labels[:, i]).sum().item()
            total += images.size(0)

    per_attr_acc = {a: per_attr_correct[a] / total for a in TARGET_ATTRS}
    print("  Per-attribute validation accuracy:")
    for attr, acc in per_attr_acc.items():
        print(f"    {attr}: {acc:.4f}")

    return per_attr_acc


def classify_generated_samples(model: nn.Module, sample_dir: str,
                               device: torch.device, batch_size: int = 64) -> dict:
    """Classify generated samples and return per-attribute positive rates."""
    eval_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolderDataset(sample_dir, transform=eval_transform)
    if len(dataset) == 0:
        return {}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    model.eval()
    per_attr_positive = {a: 0 for a in TARGET_ATTRS}
    total = 0

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            logits = model(images)
            preds = (logits > 0).float()

            for i, attr in enumerate(TARGET_ATTRS):
                per_attr_positive[attr] += preds[:, i].sum().item()
            total += images.size(0)

    per_attr_pct = {a: per_attr_positive[a] / total for a in TARGET_ATTRS}
    return {"per_attr_pct": per_attr_pct, "n_samples": total}


def main():
    parser = argparse.ArgumentParser(description="Attribute accuracy evaluation")
    parser.add_argument("--data_root", type=str, default="data/celeba-subset")
    parser.add_argument("--classifier_path", type=str,
                        default="outputs/hw4_results/eyeglasses_classifier.pt")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, load existing classifier")
    parser.add_argument("--output_json", type=str,
                        default="outputs/hw4_results/attribute_accuracy.json")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train or load classifier
    if args.eval_only and os.path.exists(args.classifier_path):
        print(f"Loading classifier from {args.classifier_path}")
        model = create_classifier(num_attrs=len(TARGET_ATTRS)).to(device)
        model.load_state_dict(torch.load(args.classifier_path, map_location=device))
    else:
        model = train_classifier(args.data_root, args.classifier_path, device,
                                 num_epochs=args.num_epochs)

    # Evaluate on CelebA validation
    val_acc = evaluate_on_celeba(model, args.data_root, device)

    # Classify generated samples
    base = "outputs/hw4_results/kid_samples"
    results = {"classifier_val_accuracy": val_acc}

    sample_dirs = [
        ("unconditional", "Unconditional CFG"),
        ("standard_eyeglasses", "Standard CFG (Eyeglasses,Male,Young)"),
        ("contrastive_eyeglasses", "Contrastive CFG (focal=Eyeglasses)"),
    ]

    # Check for guidance sweep directories (both eyeglasses and smiling)
    for w in ["1.0", "2.0", "3.0", "5.0", "7.0"]:
        for prefix in ["standard", "contrastive"]:
            for focal in ["eyeglasses", "smiling"]:
                dirname = f"{prefix}_{focal}_w{w}"
                if os.path.isdir(os.path.join(base, dirname)):
                    label = f"{'Standard' if prefix == 'standard' else 'Contrastive'} CFG ({focal}) w={w}"
                    sample_dirs.append((dirname, label))

    for dirname, label in sample_dirs:
        d = os.path.join(base, dirname)
        if os.path.isdir(d) and len(os.listdir(d)) > 0:
            print(f"\nClassifying: {label} ({d})")
            r = classify_generated_samples(model, d, device)
            results[dirname] = {"name": label, **r}
            for attr, pct in r["per_attr_pct"].items():
                print(f"  {attr}: {pct:.4f} ({pct*100:.1f}%)")

    # Save results
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
