"""
PyTorch Dataset and Deep Neural Network architecture (`ArtistStyleNet`) for
predicting the artist from a painting and extracting implicit style embeddings.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

from src.style_features import compute_gram_matrix_descriptor

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, eval_transform).
    Training uses random crops across different scales of the painting so the network
    learns local brushstroke and texture signatures rather than memorizing composition.
    """
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.55, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.08, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((int(image_size * 1.14), int(image_size * 1.14))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


class PaintingDataset(Dataset):
    """PyTorch Dataset backed by `data/metadata.csv`."""

    def __init__(self, df: pd.DataFrame, root_dir: str | Path = ".", transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row["filepath"]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)
        label = int(row["label"])
        return tensor, label, idx


class ArtistStyleNet(nn.Module):
    """
    Hybrid Deep Neural Network combining:
    1. Multi-scale Gatys Gram-matrix texture descriptors from intermediate CNN blocks
       (`layer1`, `layer2`, `layer3`) that capture spatial-invariant brushstroke & color co-occurrence.
    2. High-level semantic & compositional features from `layer4` (global average + max pooling).
    3. A 256-D latent Style Bottleneck (`style_embedding`) trained to discriminate artists.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        gram_pool_channels: int = 24,
        embedding_dim: int = 256,
        dropout: float = 0.35,
    ):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # (B, 256, 56, 56) -> fine edges, canvas grain, brushstrokes
        self.layer2 = backbone.layer2  # (B, 512, 28, 28) -> local texture & color harmony
        self.layer3 = backbone.layer3  # (B, 1024, 14, 14) -> structural motifs & shading
        self.layer4 = backbone.layer4  # (B, 2048, 7, 7) -> high-level subject & composition

        self.gram_pool_channels = gram_pool_channels
        gram_single_dim = (gram_pool_channels * (gram_pool_channels + 1)) // 2  # 300 for C=24
        self.gram_total_dim = gram_single_dim * 3  # 900 dims across layer1, layer2, layer3

        # Project multi-scale Gram texture descriptors into a 128-D texture stream
        self.gram_proj = nn.Sequential(
            nn.Linear(self.gram_total_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        # Project global layer4 features (2048 avg + 2048 max = 4096) into a 256-D semantic stream
        self.semantic_proj = nn.Sequential(
            nn.Linear(2048 * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )

        # Fuse Gram texture stream (128-D) + deep semantic stream (256-D) -> 256-D Implicit Style Embedding
        self.style_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 + 256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def freeze_early_layers(self, freeze_through_layer2: bool = True):
        """Freezes early backbone stages to prevent overfitting and speed up training."""
        for module in [self.stem, self.layer1]:
            for p in module.parameters():
                p.requires_grad = False
        if freeze_through_layer2:
            for p in self.layer2.parameters():
                p.requires_grad = False

    def unfreeze_upper_backbone(self):
        """Unfreezes layer2, layer3, and layer4 for fine-tuning."""
        for module in [self.layer2, self.layer3, self.layer4]:
            for p in module.parameters():
                p.requires_grad = True

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        # Multi-scale Gatys Gram-matrix texture vectors
        g1 = compute_gram_matrix_descriptor(f1, self.gram_pool_channels)
        g2 = compute_gram_matrix_descriptor(f2, self.gram_pool_channels)
        g3 = compute_gram_matrix_descriptor(f3, self.gram_pool_channels)
        gram_concat = torch.cat([g1, g2, g3], dim=1)
        gram_emb = self.gram_proj(gram_concat)

        # Global average + max pooling on layer4
        avg_pool = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        max_pool = F.adaptive_max_pool2d(f4, 1).flatten(1)
        sem_emb = self.semantic_proj(torch.cat([avg_pool, max_pool], dim=1))

        # Unified 256-D implicit style embedding
        style_emb = self.style_head(torch.cat([gram_emb, sem_emb], dim=1))
        logits = self.classifier(style_emb)

        if return_features:
            return logits, {
                "style_embedding": F.normalize(style_emb, p=2, dim=1),
                "gram_raw": gram_concat,
                "gram_embedding": F.normalize(gram_emb, p=2, dim=1),
                "semantic_embedding": F.normalize(sem_emb, p=2, dim=1),
            }
        return logits

