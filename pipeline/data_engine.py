"""
================================================================================
BioDINO-Edge: Data Loading, Splitting & Augmentation Engine (data_engine.py)
================================================================================
Standardized data pipeline:
  1. PairDataset: High-efficiency dual-transform Dataset (50% disk I/O savings).
  2. Dataset Resolvers: PlantVillage (color) and PlantDoc directory hierarchies.
  3. Episodic Loaders: Deterministic few-shot support/query partitioning.
  4. Augmentation: Dynamic normalization extraction and multi-view expansions.
================================================================================
"""

import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from PIL import Image

try:
    from pipeline.taxonomic_prompts import PLANTVILLAGE_TAXONOMIC_PROMPTS, PLANTDOC_TAXONOMIC_PROMPTS
except (ModuleNotFoundError, ImportError):
    from taxonomic_prompts import PLANTVILLAGE_TAXONOMIC_PROMPTS, PLANTDOC_TAXONOMIC_PROMPTS  # type: ignore[import-not-found]


class PairDataset(Dataset):
    """
    Unified dual-transformation Dataset.
    Opens each raw leaf image exactly once from disk, producing both DINOv2 and
    BioCLIP tensors simultaneously. Cuts disk I/O bottlenecks by ~50%.
    
    Args:
        samples: List of (image_path, target_class_label) tuples.
        bioclip_transform: Transform pipeline for BioCLIP (OpenCLIP normalization).
        dino_transform: Transform pipeline for DINOv2 (ImageNet normalization).
    """
    def __init__(self, samples: List[Tuple[str, int]], bioclip_transform: Any, dino_transform: Any):
        self.samples = samples
        self.bioclip_transform = bioclip_transform
        self.dino_transform = dino_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.bioclip_transform(image), self.dino_transform(image), label


def resolve_plantvillage_path(root: Path) -> Optional[Path]:
    """
    Validates and resolves PlantVillage (PV) color directory by checking
    the presence of all three target botanical genera: Apple, Grape, Potato.
    """
    root = Path(root)
    candidates = [root / "plantvillage dataset" / "color", root / "color", root]
    for candidate in candidates:
        if candidate.is_dir():
            sub_names = [d.name for d in candidate.iterdir() if d.is_dir()]
            has_apple = any(n.startswith("Apple") for n in sub_names)
            has_grape = any(n.startswith("Grape") for n in sub_names)
            has_potato = any(n.startswith("Potato") for n in sub_names)
            if has_apple and has_grape and has_potato:
                return candidate

    for candidate in root.rglob("color"):
        if candidate.is_dir():
            sub_names = [d.name for d in candidate.iterdir() if d.is_dir()]
            if any(n.startswith("Apple") for n in sub_names) and any(n.startswith("Potato") for n in sub_names):
                return candidate
    return None


resolve_pv_path = resolve_plantvillage_path


def resolve_plantdoc_paths(root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Resolves PlantDoc in-the-wild benchmark train and test directories.
    Handles nested unzipped directory variations transparently.
    """
    root = Path(root)
    if (root / "train").is_dir() and (root / "test").is_dir():
        return root / "train", root / "test"
    for train_dir in root.rglob("train"):
        if train_dir.is_dir() and (train_dir.parent / "test").is_dir():
            return train_dir, train_dir.parent / "test"
    return None, None


resolve_pd_paths = resolve_plantdoc_paths


def find_healthy_indices(class_names: List[str]) -> List[int]:
    """
    Dynamically identifies class indices representing healthy foliar status.
    Used for computing the Clinical False Negative Rate (FNR).
    """
    return [i for i, name in enumerate(class_names) if "healthy" in name.lower()]


def load_plantvillage(root: Path, k_shot: int = 5, seed: int = 42):
    """
    Loads PlantVillage novel subset (Apple, Grape, Potato; 11 classes).
    Uses a single shared RNG across all classes for unbiased episodic shuffling.
    
    Returns:
        support_samples: List of (path, local_label) tuples for support set.
        query_samples: List of (path, local_label) tuples for query evaluation.
        novel_classes: List of names for the 11 target classes.
        prompts: Pre-authored botanical taxonomic prompts for BioCLIP.
    """
    plantvillage_dir = resolve_plantvillage_path(root)
    if not plantvillage_dir:
        raise FileNotFoundError(f"Valid PlantVillage color directory (Apple/Grape/Potato) not found under {root}")

    raw_dataset = ImageFolder(root=str(plantvillage_dir))
    novel_indices = [i for i, c in enumerate(raw_dataset.classes) if c.startswith(("Apple", "Grape", "Potato"))]
    novel_classes = [raw_dataset.classes[i] for i in sorted(novel_indices)]
    
    # Global-to-Local class index mapping dictionary
    global_to_local_map = {global_idx: local_idx for local_idx, global_idx in enumerate(sorted(novel_indices))}

    samples_by_class = defaultdict(list)
    for path, label in raw_dataset.samples:
        if label in global_to_local_map:
            samples_by_class[label].append((path, global_to_local_map[label]))

    rng = random.Random(seed)
    support_samples, query_samples = [], []
    for global_label in sorted(samples_by_class.keys()):
        image_list = sorted(samples_by_class[global_label])
        shuffled_images = image_list.copy()
        rng.shuffle(shuffled_images)
        support_samples.extend(shuffled_images[:k_shot])
        query_samples.extend(shuffled_images[k_shot:])

    return support_samples, query_samples, novel_classes, PLANTVILLAGE_TAXONOMIC_PROMPTS


def load_plantdoc(root: Path, k_shot: int = 5, seed: int = 42):
    """
    Loads PlantDoc in-the-wild cross-domain benchmark dataset.
    
    Returns:
        support_samples: List of (path, local_label) tuples for support set.
        query_samples: List of (path, local_label) tuples for query evaluation.
        common_classes: List of shared class names across train and test splits.
        prompts: Pre-authored botanical taxonomic prompts for BioCLIP.
    """
    train_dir, test_dir = resolve_plantdoc_paths(root)
    if not train_dir or not test_dir:
        raise FileNotFoundError(f"PlantDoc train/test directories not found under {root}")

    classes_train = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    classes_test = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    common_classes = sorted(list(set(classes_train).intersection(set(classes_test))))

    rng = random.Random(seed)
    support_samples, query_samples = [], []
    for class_idx, class_name in enumerate(common_classes):
        train_images = sorted(list((train_dir / class_name).glob("*.*")))
        shuffled_train = train_images.copy()
        rng.shuffle(shuffled_train)
        support_samples.extend([(str(p), class_idx) for p in shuffled_train[:k_shot]])
        query_samples.extend([(str(p), class_idx) for p in sorted(list((test_dir / class_name).glob("*.*")))])

    return support_samples, query_samples, common_classes, PLANTDOC_TAXONOMIC_PROMPTS


def extract_norm_stats(
    transform: Any,
    default_mean: Optional[List[float]] = None,
    default_std: Optional[List[float]] = None,
) -> Tuple[List[float], List[float]]:
    """Dynamically reads mean and std normalization constants from a transform pipeline."""
    for t in getattr(transform, "transforms", []):
        if isinstance(t, T.Normalize):
            return list(t.mean), list(t.std)
        if hasattr(t, "mean") and hasattr(t, "std"):
            return list(t.mean), list(t.std)
    if isinstance(transform, T.Normalize):
        return list(transform.mean), list(transform.std)
    if hasattr(transform, "mean") and hasattr(transform, "std"):
        return list(transform.mean), list(transform.std)
    if default_mean is not None and default_std is not None:
        return default_mean, default_std
    raise ValueError(f"Could not dynamically extract Normalize constants from {transform}")


def build_augmentation_pipeline(mean: List[float], std: List[float]) -> T.Compose:
    """Builds multi-view augmentation pipeline for few-shot support expansion."""
    return T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


build_aug = build_augmentation_pipeline

