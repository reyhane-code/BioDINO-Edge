"""
================================================================================
BioDINO-Edge: Biologically-Grounded Multi-Scale Vision Foundation Models
with Regularized Fisher-Discriminant Alignment for Trustworthy Edge Plant Pathology
================================================================================
Modular Architecture:
  - Core Math: pipeline.core_math (Fisher LDA, Shannon Gating, Transductive Refinement)
  - Data Engine: pipeline.data_engine (PairDataset, PlantVillage, PlantDoc, Transforms)
  - Benchmark Engine: pipeline.benchmark_engine (Monte Carlo 30-Ep, 5-Way Episodic, INT8 Latency)
Zero-Degradation, Conceptual & Clean Naming, Full Scientific Reproducibility.
================================================================================
"""

import os
import sys
import json
import random
import argparse
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

# Ensure project root and pipeline dir are on sys.path
_CURR = Path(__file__).resolve().parent if "__file__" in locals() else Path(".").resolve()
_PROJECT_ROOT = _CURR if (_CURR / "plantvillage").exists() else _CURR.parent
for _p in [str(_PROJECT_ROOT), str(_CURR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from PIL import Image
import timm
try:
    from timm.data import create_transform, resolve_model_data_config  # type: ignore
except ImportError:
    from timm.data.transforms_factory import create_transform  # type: ignore
    from timm.data.config import resolve_model_data_config  # type: ignore
import open_clip

# Module imports with dual import paths (direct script execution or package import)
try:
    from pipeline.core_math import (
        calculate_shannon_confidence,
        calc_confidence,
        fit_fisher_lda,
        compute_class_prototypes,
        compute_fisher_ensemble_probs,
        refine_transductive,
    )
    from pipeline.data_engine import (
        PairDataset,
        resolve_plantvillage_path,
        resolve_pv_path,
        resolve_plantdoc_paths,
        resolve_pd_paths,
        find_healthy_indices,
        load_plantvillage,
        load_plantdoc,
        extract_norm_stats,
        build_augmentation_pipeline,
        build_aug,
    )
    from pipeline.benchmark_engine import (
        run_plantvillage_montecarlo,
        run_plantdoc_episodic,
        benchmark_latency,
    )
except (ModuleNotFoundError, ImportError):
    from core_math import (  # type: ignore[no-redef]
        calculate_shannon_confidence,
        calc_confidence,
        fit_fisher_lda,
        compute_class_prototypes,
        compute_fisher_ensemble_probs,
        refine_transductive,
    )
    from data_engine import (  # type: ignore[no-redef]
        PairDataset,
        resolve_plantvillage_path,
        resolve_pv_path,
        resolve_plantdoc_paths,
        resolve_pd_paths,
        find_healthy_indices,
        load_plantvillage,
        load_plantdoc,
        extract_norm_stats,
        build_augmentation_pipeline,
        build_aug,
    )
    from benchmark_engine import (  # type: ignore[no-redef]
        run_plantvillage_montecarlo,
        run_plantdoc_episodic,
        benchmark_latency,
    )

# =============================================================================
# 0. Configuration & Global State
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = _PROJECT_ROOT
PLANTVILLAGE_ROOT = PROJECT_ROOT / "plantvillage"
PLANTDOC_ROOT = PROJECT_ROOT / "plantdoc"
CACHE_DIR = PROJECT_ROOT / "cache_artifacts"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_FILE = PROJECT_ROOT / "biodino_hierarchical_results.json"


@dataclass
class EdgeConfig:
    """
    Concise hyperparameter configuration for BioDINO-Edge.
    
    Attributes:
        seed: Master random seed for deterministic scientific reproducibility.
        k_shot: Number of support leaf samples per novel class.
        batch_size: Query inference batch size for streaming.
        scale: Inverse temperature logit scaling factor (gamma = 20.0).
        w_text: Weight for text prototype fusion (0.20 text + 0.80 visual).
        w_lda: Weight for Linear Discriminant Analysis in ensemble (0.75 LDA + 0.25 text).
        w_base: Weight for base visual ensemble (0.65 BioCLIP + 0.35 DINOv2).
        tau: Confidence gating threshold (tau_conf) for Shannon margin-entropy routing.
        n_views: Multi-view support augmentation count (1 canonical + 7 perturbed views).
    """
    seed: int = 42
    k_shot: int = 5
    batch_size: int = 128
    scale: float = 20.0
    w_text: float = 0.20   # Prototype fusion: 0.80 visual + 0.20 text
    w_lda: float = 0.75    # Ensemble: 0.75 LDA + 0.25 text
    w_base: float = 0.65   # Base visual ensemble: 0.65 BioCLIP + 0.35 DINOv2
    tau: float = 0.35      # Shannon margin-entropy routing threshold
    n_views: int = 8       # 1 original + 7 augmentations for support set


@dataclass
class ModelBackbones:
    """Encapsulates foundation models and pre-configured transformation pipelines."""
    dino_model: Any
    bioclip_model: Any
    tokenizer: Any
    dino_transforms: Any
    bioclip_transforms: Any
    dino_config: dict


def get_autocast_ctx(device: torch.device):
    """
    Provides a safe Automatic Mixed Precision (AMP) autocast context manager.
    Active on CUDA (using float16); falls back to a nullcontext on CPU.
    """
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# =============================================================================
# 1. Model Loading & Feature Extraction Engine
# =============================================================================
def load_backbones(device: torch.device = DEVICE) -> ModelBackbones:
    """Loads frozen DINOv2-Small and BioCLIP foundation models with dynamic transforms."""
    print("[*] Loading Foundation Backbones:")
    # Fast path: DINOv2 Small (ViT-S/14)
    dino_model = timm.create_model("vit_small_patch14_dinov2", pretrained=True, dynamic_img_size=True).to(device).eval()
    dino_config = resolve_model_data_config(dino_model)  # type: ignore
    dino_config["input_size"] = (3, 224, 224)
    dino_transforms = create_transform(**dino_config, is_training=False)  # type: ignore

    # Deep path: BioCLIP ViT-B/16
    bioclip_model, _, bioclip_transforms = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    bioclip_model = bioclip_model.to(device).eval()

    return ModelBackbones(
        dino_model=dino_model,
        bioclip_model=bioclip_model,
        tokenizer=tokenizer,
        dino_transforms=dino_transforms,
        bioclip_transforms=bioclip_transforms,
        dino_config=dino_config,
    )


def extract_support_features(
    support_samples: List[Tuple[str, int]],
    models: ModelBackbones,
    device: torch.device = DEVICE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extracts 8-view augmented support embeddings for BioCLIP and DINOv2."""
    bioclip_mean, bioclip_std = extract_norm_stats(models.bioclip_transforms)
    aug_bioclip = build_augmentation_pipeline(bioclip_mean, bioclip_std)
    aug_dino = build_augmentation_pipeline(models.dino_config["mean"], models.dino_config["std"])

    print(f"[*] Extracting multi-view support vectors ({len(support_samples) * 8} views)...")
    support_bioclip_list, support_dino_list, support_labels_list = [], [], []

    with torch.no_grad():
        for image_path, class_label in support_samples:
            image = Image.open(image_path).convert("RGB")
            bioclip_views = [models.bioclip_transforms(image)] + [aug_bioclip(image) for _ in range(7)]
            dino_views = [models.dino_transforms(image)] + [aug_dino(image) for _ in range(7)]

            bioclip_features = F.normalize(models.bioclip_model.encode_image(torch.stack(bioclip_views).to(device)), dim=-1)
            dino_features = F.normalize(models.dino_model(torch.stack(dino_views).to(device)), dim=-1)

            support_bioclip_list.append(bioclip_features)
            support_dino_list.append(dino_features)
            support_labels_list.append(torch.full((8,), class_label, dtype=torch.long, device=device))

    return (
        torch.cat(support_bioclip_list, dim=0),
        torch.cat(support_dino_list, dim=0),
        torch.cat(support_labels_list, dim=0),
    )


def build_text_prototypes(
    class_names: List[str],
    prompts_map: Dict[str, List[str]],
    bioclip_model: Any,
    tokenizer: Any,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """Synthesizes normalized taxonomic text prototypes using BioCLIP."""
    print("[*] Synthesizing taxonomic domain prompts via BioCLIP text encoder...")
    prototypes = []
    with torch.no_grad():
        for class_name in class_names:
            tokens = tokenizer(prompts_map[class_name]).to(device)
            text_features = F.normalize(bioclip_model.encode_text(tokens), dim=-1)
            prototypes.append(F.normalize(text_features.mean(dim=0), dim=-1))
    return torch.stack(prototypes)


# =============================================================================
# 2. Cascaded Inference & Evaluation Engine
# =============================================================================
class CascadedEvaluator:
    """
    True Adaptive Cascaded Edge Predictor & Evaluator.
    
    Executes DINOv2 fast-path first (~21 ms on CPU / <2 ms on GPU); only uncertain
    queries whose Shannon confidence falls below threshold tau trigger the
    compute-intensive BioCLIP (ViT-B) + Fisher-LDA deep path on edge.
    Bypasses deep computation for >70% of confident samples, drastically reducing latency.
    """
    def __init__(
        self,
        dino_model: Any,
        bioclip_model: Any,
        dino_prototypes: torch.Tensor,
        bioclip_prototypes: torch.Tensor,
        text_prototypes: torch.Tensor,
        fisher_weights: torch.Tensor,
        fisher_biases: torch.Tensor,
        cfg: EdgeConfig,
        device: torch.device = DEVICE,
    ):
        self.dino_model = dino_model
        self.bioclip_model = bioclip_model
        self.device = device
        self.autocast_ctx = get_autocast_ctx(device)
        self.dino_prototypes = dino_prototypes.to(device)          # (n_classes, 384)
        self.bioclip_prototypes = bioclip_prototypes.to(device)    # (n_classes, 512)
        self.text_prototypes = text_prototypes.to(device)          # (n_classes, 512)
        self.fisher_weights = fisher_weights.to(device)            # (n_classes, 896)
        self.fisher_biases = fisher_biases.to(device)              # (n_classes,)
        self.cfg = cfg

    def _infer_fast_path(
        self,
        dino_features: torch.Tensor,
        n_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Internal reusable helper: Computes DINOv2 fast-path logits, probabilities,
        confidence gating, and initial predictions.
        """
        prototypes = self.dino_prototypes.to(device=dino_features.device, dtype=dino_features.dtype)
        dino_logits = self.cfg.scale * (dino_features @ prototypes.t())
        dino_prob = F.softmax(dino_logits, dim=-1)
        dino_conf = calculate_shannon_confidence(dino_prob, n_classes)
        dino_pred = dino_logits.argmax(dim=-1)
        return dino_logits, dino_prob, dino_conf, dino_pred

    def predict_cascaded(
        self,
        dino_images: torch.Tensor,
        bioclip_images: torch.Tensor,
        n_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        True lazy edge inference: Only uncertain queries (conf < tau) trigger BioCLIP.
        Confident samples bypass BioCLIP completely.
        """
        with torch.no_grad(), self.autocast_ctx:
            # 1. Fast Path
            dino_features = F.normalize(self.dino_model(dino_images), dim=-1)
            _, _, dino_conf, dino_pred = self._infer_fast_path(dino_features, n_classes)
            cascaded_pred = dino_pred.clone()

            # 2. Confidence Gating
            deep_mask = (dino_conf < self.cfg.tau)
            if deep_mask.any():
                uncertain_bioclip_images = bioclip_images[deep_mask]
                uncertain_bioclip_features = F.normalize(self.bioclip_model.encode_image(uncertain_bioclip_images), dim=-1)
                uncertain_dino_features = dino_features[deep_mask]

                ensemble_prob, _, _ = compute_fisher_ensemble_probs(
                    fisher_weights=self.fisher_weights,
                    fisher_biases=self.fisher_biases,
                    bioclip_features=uncertain_bioclip_features,
                    dino_features=uncertain_dino_features,
                    text_prototypes=self.text_prototypes,
                    scale=self.cfg.scale,
                    w_lda=self.cfg.w_lda,
                    device=self.device,
                )
                cascaded_pred[deep_mask] = ensemble_prob.argmax(dim=-1)

        return cascaded_pred, deep_mask

    def evaluate_tensors(
        self,
        query_dino_features: torch.Tensor,
        query_bioclip_features: torch.Tensor,
        targets: List[int],
        n_classes: int,
    ) -> Dict:
        """
        Evaluates query tensors across all baseline model tiers using pre-extracted
        cached feature embeddings (Fast-Cache Mode).
        """
        with torch.no_grad(), self.autocast_ctx:
            dino_features = query_dino_features.to(self.device)
            bioclip_features = query_bioclip_features.to(self.device)

            # Fast Path: DINOv2 alone
            dino_logits, _, dino_conf, dino_pred = self._infer_fast_path(dino_features, n_classes)
            deep_mask = (dino_conf < self.cfg.tau)
            deep_trigger_count = deep_mask.sum().item()

            # Deep Path: BioCLIP alone
            bio_proto = self.bioclip_prototypes.to(device=bioclip_features.device, dtype=bioclip_features.dtype)
            bioclip_logits = self.cfg.scale * (bioclip_features @ bio_proto.t())
            bioclip_pred = bioclip_logits.argmax(dim=-1)

            # Consensus baseline (Base visual fusion: 65% BioCLIP + 35% DINOv2, no LDA)
            consensus_pred = (self.cfg.w_base * bioclip_logits + (1.0 - self.cfg.w_base) * dino_logits).argmax(dim=-1)

            # Joint 896-dim Fisher LDA + Text ensemble
            joint_bio_dino_features = torch.cat([bioclip_features.float(), dino_features.float()], dim=-1)
            ensemble_prob, _, text_prob = compute_fisher_ensemble_probs(
                fisher_weights=self.fisher_weights,
                fisher_biases=self.fisher_biases,
                joint_features=joint_bio_dino_features,
                bioclip_features=bioclip_features,
                text_prototypes=self.text_prototypes,
                scale=self.cfg.scale,
                w_lda=self.cfg.w_lda,
                device=self.device,
            )
            fisher_pred = ensemble_prob.argmax(dim=-1)

            # Adaptive Cascaded Routing prediction
            cascaded_pred = dino_pred.clone()
            if deep_mask.any():
                cascaded_pred[deep_mask] = fisher_pred[deep_mask]

        return {
            "targets": targets,
            "preds_dino": dino_pred.cpu().tolist(),
            "preds_bio": bioclip_pred.cpu().tolist(),
            "preds_bf": consensus_pred.cpu().tolist(),
            "preds_fe": fisher_pred.cpu().tolist(),
            "preds_casc": cascaded_pred.cpu().tolist(),
            "joint_bio_dino_features": joint_bio_dino_features.cpu(),
            "probs_ensemble": ensemble_prob.cpu(),
            "probs_text": text_prob.cpu(),
            "deep_triggers": deep_trigger_count,
            "total_queries": len(targets),
        }

    def evaluate_queries(
        self,
        query_loader: DataLoader,
        n_classes: int,
    ) -> Dict:
        """Runs true cascaded inference over DataLoader queries, gathering all baseline comparisons."""
        targets = []
        preds_dino, preds_bio, preds_bf, preds_fe, preds_casc = [], [], [], [], []
        joint_features_list, ensemble_probs_list, text_probs_list = [], [], []
        deep_trigger_count, total_queries = 0, 0

        with torch.no_grad():
            for bioclip_images, dino_images, labels in query_loader:
                bioclip_images = bioclip_images.to(self.device)
                dino_images = dino_images.to(self.device)
                batch_size = len(labels)
                total_queries += batch_size

                with self.autocast_ctx:
                    # 1. Fast Path
                    dino_features = F.normalize(self.dino_model(dino_images), dim=-1)
                    dino_logits, _, dino_conf, dino_pred = self._infer_fast_path(dino_features, n_classes)
                    deep_mask = (dino_conf < self.cfg.tau)
                    deep_trigger_count += deep_mask.sum().item()

                    # 2. Deep Path & Baselines
                    bioclip_features = F.normalize(self.bioclip_model.encode_image(bioclip_images), dim=-1)
                    bio_proto = self.bioclip_prototypes.to(device=bioclip_features.device, dtype=bioclip_features.dtype)
                    bioclip_logits = self.cfg.scale * (bioclip_features @ bio_proto.t())
                    bioclip_pred = bioclip_logits.argmax(dim=-1)

                    # Consensus baseline
                    consensus_pred = (self.cfg.w_base * bioclip_logits + (1.0 - self.cfg.w_base) * dino_logits).argmax(dim=-1)

                    # Joint 896-dim Fisher-LDA ensemble
                    joint_bio_dino_features = torch.cat([bioclip_features.float(), dino_features.float()], dim=-1)
                    ensemble_prob, _, text_prob = compute_fisher_ensemble_probs(
                        fisher_weights=self.fisher_weights,
                        fisher_biases=self.fisher_biases,
                        joint_features=joint_bio_dino_features,
                        bioclip_features=bioclip_features,
                        text_prototypes=self.text_prototypes,
                        scale=self.cfg.scale,
                        w_lda=self.cfg.w_lda,
                        device=self.device,
                    )
                    fisher_pred = ensemble_prob.argmax(dim=-1)

                    # Cascaded prediction
                    cascaded_pred = dino_pred.clone()
                    if deep_mask.any():
                        cascaded_pred[deep_mask] = fisher_pred[deep_mask]

                targets.extend(labels.tolist())
                preds_dino.extend(dino_pred.cpu().tolist())
                preds_bio.extend(bioclip_pred.cpu().tolist())
                preds_bf.extend(consensus_pred.cpu().tolist())
                preds_fe.extend(fisher_pred.cpu().tolist())
                preds_casc.extend(cascaded_pred.cpu().tolist())

                joint_features_list.append(joint_bio_dino_features.cpu())
                ensemble_probs_list.append(ensemble_prob.cpu())
                text_probs_list.append(text_prob.cpu())

        return {
            "targets": targets,
            "preds_dino": preds_dino,
            "preds_bio": preds_bio,
            "preds_bf": preds_bf,
            "preds_fe": preds_fe,
            "preds_casc": preds_casc,
            "joint_bio_dino_features": torch.cat(joint_features_list, dim=0),
            "probs_ensemble": torch.cat(ensemble_probs_list, dim=0),
            "probs_text": torch.cat(text_probs_list, dim=0),
            "deep_triggers": deep_trigger_count,
            "total_queries": total_queries,
        }


# =============================================================================
# 3. Hierarchical Mixture-of-Experts (MoE) & Species Routing Engine
# =============================================================================
def extract_botanical_species(class_name: str) -> str:
    """Extracts botanical host plant genus/species from standard dataset folder name."""
    c = class_name.lower().replace("-", "_")
    if "___" in c:
        return c.split("___")[0]
    for s in [
        "apple", "bell_pepper", "blueberry", "cherry", "corn", "grape",
        "peach", "potato", "raspberry", "soyabean", "soybean", "squash",
        "strawberry", "tomato"
    ]:
        if s in c:
            return "soybean" if "soya" in s else s
    return "other"


def is_healthy_leaf(class_name: str) -> bool:
    """Detects healthy foliar status for computing clinical diagnostic safety rates."""
    c = class_name.lower()
    if "healthy" in c:
        return True
    diseases = ["scab", "rust", "spot", "blight", "mildew", "virus", "mold", "rot"]
    if any(d in c for d in diseases):
        return False
    return c.endswith("_leaf") or c.endswith("leaf")


def evaluate_hierarchical_moe(
    support_joint_features: np.ndarray,
    support_labels: np.ndarray,
    query_joint_features: np.ndarray,
    targets: List[int],
    class_names: List[str],
    cfg: EdgeConfig,
    device: torch.device = DEVICE,
) -> Dict[str, Any]:
    """
    Evaluates Hierarchical Mixture-of-Experts (MoE) Architecture across:
      1. Crop-Guided Mode (Farmer Selection / Interactive Mobile Oracle)
      2. Hard AI Router (Top-1 Cascaded Single-Crop)
      3. Soft AI Router (Top-2 Balanced MoE)
      4. Soft AI Router (Top-3 Balanced MoE)
    """
    species_names = [extract_botanical_species(c) for c in class_names]
    unique_species = sorted(list(set(species_names)))
    species_to_id = {s: i for i, s in enumerate(unique_species)}
    class_to_species = np.array([species_to_id[s] for s in species_names])
    species_to_classes = {s_idx: np.where(class_to_species == s_idx)[0] for s_idx in range(len(unique_species))}

    sup_species = class_to_species[support_labels]
    qry_species_true = class_to_species[np.array(targets)]

    # 1. Species Prototypes
    species_prototypes = np.stack([
        support_joint_features[sup_species == s].mean(axis=0) for s in range(len(unique_species))
    ])
    species_prototypes = species_prototypes / (np.linalg.norm(species_prototypes, axis=-1, keepdims=True) + 1e-12)

    # 2. Fit intra-species Fisher LDA classifiers
    species_classifiers = {}
    for s_idx in range(len(unique_species)):
        s_classes = species_to_classes[s_idx]
        s_sup_mask = (sup_species == s_idx)
        s_sup_x = support_joint_features[s_sup_mask]
        s_sup_y = support_labels[s_sup_mask]
        if len(s_classes) > 1:
            local_map = {c: i for i, c in enumerate(s_classes)}
            s_sup_y_local = np.array([local_map[y] for y in s_sup_y])
            _, _, clf_exp = fit_fisher_lda(s_sup_x, s_sup_y_local, device="cpu")
            species_classifiers[s_idx] = (clf_exp, s_classes)
        else:
            species_classifiers[s_idx] = (None, s_classes)

    # 3. Router Inference
    qry_norm = query_joint_features / (np.linalg.norm(query_joint_features, axis=-1, keepdims=True) + 1e-12)
    species_sim = qry_norm @ species_prototypes.T
    species_probs = F.softmax(torch.tensor(species_sim) * 12.0, dim=-1).numpy()

    acc_r_top1 = float(np.mean(species_sim.argmax(axis=-1) == qry_species_true) * 100.0)
    top2_idx = np.argsort(species_sim, axis=-1)[:, -2:]
    acc_r_top2 = float(np.mean([qry_species_true[i] in top2_idx[i] for i in range(len(targets))]) * 100.0)
    top3_idx = np.argsort(species_sim, axis=-1)[:, -3:]
    acc_r_top3 = float(np.mean([qry_species_true[i] in top3_idx[i] for i in range(len(targets))]) * 100.0)

    # 4. Mode A: Crop-Guided Mode (Farmer Selection / Interactive Oracle)
    preds_farmer = []
    per_species_acc = {}
    for s_idx, s_name in enumerate(unique_species):
        s_mask = (qry_species_true == s_idx)
        if not np.any(s_mask):
            continue
        clf_exp, s_classes = species_classifiers[s_idx]
        if clf_exp is None or len(s_classes) == 1:
            s_preds = [s_classes[0]] * int(np.sum(s_mask))
        else:
            s_qry = query_joint_features[s_mask]
            p_loc = clf_exp.predict(s_qry)
            s_preds = [s_classes[p] for p in p_loc]
        s_targets = np.array(targets)[s_mask]
        s_acc = float(accuracy_score(s_targets, s_preds) * 100.0)
        per_species_acc[s_name] = {
            "accuracy": s_acc,
            "n_images": int(len(s_targets)),
            "n_classes": int(len(s_classes)),
        }

    for i in range(len(targets)):
        s_true = qry_species_true[i]
        clf_exp, s_classes = species_classifiers[s_true]
        if clf_exp is None or len(s_classes) == 1:
            preds_farmer.append(s_classes[0])
        else:
            p_loc = clf_exp.predict(query_joint_features[i:i+1])[0]
            preds_farmer.append(s_classes[p_loc])

    # 5. Mode B: Hard AI Router (Top-1 Cascaded)
    preds_hard = []
    for i in range(len(targets)):
        s_pred = species_sim[i].argmax()
        clf_exp, s_classes = species_classifiers[s_pred]
        if clf_exp is None or len(s_classes) == 1:
            preds_hard.append(s_classes[0])
        else:
            p_loc = clf_exp.predict(query_joint_features[i:i+1])[0]
            preds_hard.append(s_classes[p_loc])

    # 6. Mode C: Soft AI Router (Top-2 Balanced MoE)
    preds_soft2 = []
    for i in range(len(targets)):
        top2_s = np.argsort(species_probs[i])[-2:]
        cand_scores = {}
        for s in top2_s:
            clf_exp, s_classes = species_classifiers[s]
            p_s = species_probs[i, s]
            k_s = len(s_classes)
            if clf_exp is None or k_s == 1:
                cand_scores[s_classes[0]] = p_s * 1.0
            else:
                loc_probs = clf_exp.predict_proba(query_joint_features[i:i+1])[0]
                for idx_c, c in enumerate(s_classes):
                    cand_scores[c] = p_s * (k_s * loc_probs[idx_c])
        preds_soft2.append(max(cand_scores, key=cand_scores.get))

    # 7. Mode D: Soft AI Router (Top-3 Balanced MoE)
    preds_soft3 = []
    for i in range(len(targets)):
        top3_s = np.argsort(species_probs[i])[-3:]
        cand_scores = {}
        for s in top3_s:
            clf_exp, s_classes = species_classifiers[s]
            p_s = species_probs[i, s]
            k_s = len(s_classes)
            if clf_exp is None or k_s == 1:
                cand_scores[s_classes[0]] = p_s * 1.0
            else:
                loc_probs = clf_exp.predict_proba(query_joint_features[i:i+1])[0]
                for idx_c, c in enumerate(s_classes):
                    cand_scores[c] = p_s * (k_s * loc_probs[idx_c])
        preds_soft3.append(max(cand_scores, key=cand_scores.get))

    # Helper for Diagnostic Safety Rates (FNR, FPR)
    healthy_classes = set([i for i, c in enumerate(class_names) if is_healthy_leaf(c)])
    def compute_rates(preds):
        acc = float(accuracy_score(targets, preds) * 100.0)
        f1 = float(f1_score(targets, preds, average="macro", zero_division=0) * 100.0)
        diseased_mask = np.array([t not in healthy_classes for t in targets])
        healthy_mask = np.array([t in healthy_classes for t in targets])
        pred_healthy = np.array([p in healthy_classes for p in preds])
        pred_diseased = np.array([p not in healthy_classes for p in preds])
        fn = int(np.sum(diseased_mask & pred_healthy))
        fnr = float((fn / np.sum(diseased_mask)) * 100.0) if np.sum(diseased_mask) > 0 else 0.0
        fp = int(np.sum(healthy_mask & pred_diseased))
        fpr = float((fp / np.sum(healthy_mask)) * 100.0) if np.sum(healthy_mask) > 0 else 0.0
        return {"accuracy": acc, "f1": f1, "fnr": fnr, "fpr": fpr, "fn_count": fn, "fp_count": fp}

    return {
        "router_accuracy": {"top1": acc_r_top1, "top2": acc_r_top2, "top3": acc_r_top3},
        "crop_guided": compute_rates(preds_farmer),
        "hard_router_top1": compute_rates(preds_hard),
        "soft_router_top2": compute_rates(preds_soft2),
        "soft_router_top3": compute_rates(preds_soft3),
        "per_species_breakdown": per_species_acc,
    }


# =============================================================================
# 4. Benchmark Orchestration & CLI
# =============================================================================
def run_benchmark(
    dataset_name: str,
    support_samples: List[Tuple[str, int]],
    query_samples: List[Tuple[str, int]],
    class_names: List[str],
    taxonomic_prompts: Dict[str, List[str]],
    models: ModelBackbones,
    cfg: EdgeConfig = EdgeConfig(),
    run_latency_bench: bool = False,
    use_cache: bool = False,
) -> Dict:
    """
    Executes the end-to-end Few-Shot Benchmark Pipeline for a specified dataset.
    Coordinates feature extraction, prototype synthesis, cascaded routing evaluation,
    closed-form transductive recalibration, and statistical reporting.
    """
    n_classes = len(class_names)
    print("=" * 80 + f"\n  BENCHMARK: {dataset_name.upper()} ({n_classes} Classes | {cfg.k_shot}-Shot Setup)\n" +
          f"  Support Images: {len(support_samples)} | Query Images: {len(query_samples)}\n" + "=" * 80)

    # 1. Multi-view support features & taxonomic prototypes
    support_bioclip, support_dino, support_labels = extract_support_features(support_samples, models, device=DEVICE)
    text_prototypes = build_text_prototypes(class_names, taxonomic_prompts, models.bioclip_model, models.tokenizer, device=DEVICE)

    # Prototype construction
    dino_prototypes = compute_class_prototypes(support_dino, support_labels, n_classes)
    bioclip_visual_prototypes = compute_class_prototypes(support_bioclip, support_labels, n_classes)
    bioclip_prototypes = F.normalize(
        (1.0 - cfg.w_text) * bioclip_visual_prototypes + cfg.w_text * text_prototypes, dim=-1
    )

    # Joint 896-dimensional manifold representation
    support_joint_features = torch.cat([support_bioclip, support_dino], dim=-1).cpu().numpy()
    support_labels_np = support_labels.cpu().numpy()
    fisher_weights, fisher_biases, _ = fit_fisher_lda(support_joint_features, support_labels_np, device=DEVICE)

    # 2. Query Inference via CascadedEvaluator
    evaluator = CascadedEvaluator(
        dino_model=models.dino_model,
        bioclip_model=models.bioclip_model,
        dino_prototypes=dino_prototypes,
        bioclip_prototypes=bioclip_prototypes,
        text_prototypes=text_prototypes,
        fisher_weights=fisher_weights,
        fisher_biases=fisher_biases,
        cfg=cfg,
        device=DEVICE,
    )

    cache_file = CACHE_DIR / "pv_novel_embeddings.pt"
    if not cache_file.exists():
        cache_file = PROJECT_ROOT / "pv_novel_embeddings.pt"

    if use_cache and dataset_name == "plantvillage" and cache_file.exists():
        print(f"[*] Fast-Cache Mode: Loading pre-extracted embeddings from {cache_file.name}...")
        cache_data = torch.load(cache_file, map_location="cpu")
        path_to_idx = {Path(p).resolve(): i for i, p in enumerate(cache_data["paths"])}
        query_indices = [path_to_idx[Path(p).resolve()] for p, _ in query_samples]
        query_dino_features = cache_data["dino_feats"][query_indices]
        query_bioclip_features = cache_data["bio_feats"][query_indices]
        query_labels = [y for _, y in query_samples]
        inf_res = evaluator.evaluate_tensors(query_dino_features, query_bioclip_features, query_labels, n_classes)
    else:
        print(f"[*] Evaluating on {len(query_samples)} Test Queries via DataLoader...")
        query_dataset = PairDataset(query_samples, models.bioclip_transforms, models.dino_transforms)
        query_loader = DataLoader(query_dataset, batch_size=cfg.batch_size, shuffle=False)
        inf_res = evaluator.evaluate_queries(query_loader, n_classes)

    targets = inf_res["targets"]
    dino_preds = inf_res["preds_dino"]
    bioclip_preds = inf_res["preds_bio"]
    consensus_preds = inf_res["preds_bf"]
    fisher_preds = inf_res["preds_fe"]
    cascaded_preds = inf_res["preds_casc"]

    # 3. Transductive Refinement
    query_joint_features = inf_res["joint_bio_dino_features"]
    ensemble_probs = inf_res["probs_ensemble"]
    text_probs = inf_res["probs_text"]
    top_confs, _ = ensemble_probs.max(dim=-1)

    refined_preds, _, _ = refine_transductive(
        support_features=support_joint_features,
        support_labels=support_labels_np,
        query_features=query_joint_features,
        initial_preds=np.array(fisher_preds),
        confidences=top_confs,
        text_prob=text_probs,
        n_classes=n_classes,
        support_weight=3 if dataset_name == "plantvillage" else 1,
        max_anchors=20 if dataset_name == "plantvillage" else 2,
        conf_threshold=0.70 if dataset_name == "plantvillage" else None,
        device=DEVICE,
        w_lda=cfg.w_lda,
    )

    # 4. Hierarchical MoE & Species Routing Evaluation
    moe_res = evaluate_hierarchical_moe(
        support_joint_features=support_joint_features,
        support_labels=support_labels_np,
        query_joint_features=query_joint_features.numpy(),
        targets=targets,
        class_names=class_names,
        cfg=cfg,
        device=DEVICE,
    )

    # 5. Metrics Reporting
    calc_acc = lambda p: float(accuracy_score(targets, p) * 100.0)
    calc_f1 = lambda p: float(f1_score(targets, p, average="macro", zero_division=0) * 100.0)
    fast_ratio = (1.0 - (inf_res["deep_triggers"] / max(inf_res["total_queries"], 1))) * 100.0

    print("-" * 80 + f"\n  {dataset_name.upper()} RESULTS SUMMARY (Seed {cfg.seed})\n" + "-" * 80)
    print("  [Standard Flat Baselines]")
    print(f"  1. Fast-Path (DINOv2 Small Alone)            : Acc = {calc_acc(dino_preds):.2f}% | F1 = {calc_f1(dino_preds):.2f}%")
    print(f"  2. Deep-Path (BioCLIP ViT-B Alone)           : Acc = {calc_acc(bioclip_preds):.2f}% | F1 = {calc_f1(bioclip_preds):.2f}%")
    print(f"  3. Static Prototype Consensus (Base BioDINO) : Acc = {calc_acc(consensus_preds):.2f}% | F1 = {calc_f1(consensus_preds):.2f}%")
    print(f"  4. BioDINO + Fisher-LDA (Inductive SOTA)     : Acc = {calc_acc(fisher_preds):.2f}% | F1 = {calc_f1(fisher_preds):.2f}%")
    print(f"  5. BioDINO-Edge (Adaptive Cascaded Routing)  : Acc = {calc_acc(cascaded_preds):.2f}% | F1 = {calc_f1(cascaded_preds):.2f}%")
    print(f"  6. BioDINO Transductive Refinement (SOTA)    : Acc = {calc_acc(refined_preds):.2f}% | F1 = {calc_f1(refined_preds):.2f}%")
    print("-" * 80)
    print("  [Hierarchical MoE & Crop-Routing Tiers]")
    print(f"  Router Species Top-1 / Top-2 / Top-3 Acc     : {moe_res['router_accuracy']['top1']:.2f}% / {moe_res['router_accuracy']['top2']:.2f}% / {moe_res['router_accuracy']['top3']:.2f}%")
    print(f"  7. Hard AI Router (Top-1 Single-Crop)        : Acc = {moe_res['hard_router_top1']['accuracy']:.2f}% | F1 = {moe_res['hard_router_top1']['f1']:.2f}% | FNR = {moe_res['hard_router_top1']['fnr']:.2f}% | FPR = {moe_res['hard_router_top1']['fpr']:.2f}%")
    print(f"  8. Soft AI Router (Top-2 Balanced MoE)       : Acc = {moe_res['soft_router_top2']['accuracy']:.2f}% | F1 = {moe_res['soft_router_top2']['f1']:.2f}% | FNR = {moe_res['soft_router_top2']['fnr']:.2f}% | FPR = {moe_res['soft_router_top2']['fpr']:.2f}%")
    print(f"  9. Soft AI Router (Top-3 Balanced MoE)       : Acc = {moe_res['soft_router_top3']['accuracy']:.2f}% | F1 = {moe_res['soft_router_top3']['f1']:.2f}% | FNR = {moe_res['soft_router_top3']['fnr']:.2f}% | FPR = {moe_res['soft_router_top3']['fpr']:.2f}%")
    print(f" 10. Crop-Guided Mode (Farmer Selection / MoE) : Acc = {moe_res['crop_guided']['accuracy']:.2f}% | F1 = {moe_res['crop_guided']['f1']:.2f}% | FNR = {moe_res['crop_guided']['fnr']:.2f}% | FPR = {moe_res['crop_guided']['fpr']:.2f}%")
    print("=" * 80)
    print(f"\n  [Per-Species Intra-Crop Breakdown (Crop-Guided Mode)]")
    for s_name, s_info in sorted(moe_res["per_species_breakdown"].items()):
        print(f"    - {s_name:<16}: Acc = {s_info['accuracy']:6.2f}% ({s_info['n_images']:2d} images, {s_info['n_classes']:2d} classes)")
    print("=" * 80)

    # 6. Multi-Episode Benchmarks & Latency
    multi_episode_res = {}
    if dataset_name == "plantvillage":
        if cache_file.exists():
            multi_episode_res = run_plantvillage_montecarlo(
                cache_file, n_classes, class_names, k_shot=cfg.k_shot, n_episodes=30, seed=cfg.seed, scale=cfg.scale, w_lda=cfg.w_lda
            )
    elif dataset_name == "plantdoc":
        multi_episode_res = run_plantdoc_episodic(
            support_joint_features=support_joint_features,
            support_labels=support_labels_np,
            query_joint_features=query_joint_features.numpy(),
            query_labels=np.array(targets),
            query_bioclip_features=query_joint_features[:, :512],
            text_prototypes=text_prototypes.cpu(),
            n_classes=n_classes,
            n_episodes=50,
            seed=cfg.seed,
            scale=cfg.scale,
            w_lda=cfg.w_lda,
        )

    latency_res = benchmark_latency(models.dino_model, models.bioclip_model, fast_ratio, 100.0 - fast_ratio) if run_latency_bench else {}
    print(f"\n[*] Detailed Classification Report ({dataset_name} Transductive):")
    print(classification_report(targets, refined_preds, target_names=[c.replace("___", " ").replace("_", " ") for c in class_names], digits=3, zero_division=0))

    return {
        "dataset": dataset_name,
        "n_classes": n_classes,
        "k_shot": cfg.k_shot,
        "metrics": {
            "accuracy_fast_dino": calc_acc(dino_preds),
            "f1_fast_dino": calc_f1(dino_preds),
            "accuracy_deep_bio": calc_acc(bioclip_preds),
            "f1_deep_bio": calc_f1(bioclip_preds),
            "accuracy_base_fused": calc_acc(consensus_preds),
            "f1_base_fused": calc_f1(consensus_preds),
            "accuracy_peak_fisher": calc_acc(fisher_preds),
            "f1_peak_fisher": calc_f1(fisher_preds),
            "accuracy_cascaded_edge": calc_acc(cascaded_preds),
            "f1_cascaded_edge": calc_f1(cascaded_preds),
            "accuracy_refined_transductive": calc_acc(refined_preds),
            "f1_refined_transductive": calc_f1(refined_preds),
            "hierarchical_moe": moe_res,
            "multi_episode_30": multi_episode_res,
        },
        "edge_profiling": latency_res,
        "confusion_matrix": confusion_matrix(targets, refined_preds).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="BioDINO-Edge Dynamic Multi-Dataset Benchmark Pipeline")
    parser.add_argument("--dataset", choices=["plantvillage", "plantdoc", "both"], default="plantvillage")
    parser.add_argument("--plantvillage_dir", type=str, default=str(PLANTVILLAGE_ROOT))
    parser.add_argument("--plantdoc_dir", type=str, default=str(PLANTDOC_ROOT))
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile_latency", action="store_true")
    parser.add_argument("--use_cache", action="store_true", help="Use cached feature embeddings for fast evaluation")
    args = parser.parse_args()

    print("=" * 80 + f"\n  BioDINO-Edge Pipeline | Device: {DEVICE} | Seed: {args.seed} | Mode: {args.dataset}\n" + "=" * 80)
    models = load_backbones(DEVICE)
    cfg = EdgeConfig(seed=args.seed, k_shot=args.k_shot)
    results = {}

    runs = []
    if args.dataset in ["plantvillage", "both"]:
        support_samples, query_samples, class_names, prompts = load_plantvillage(Path(args.plantvillage_dir), args.k_shot, args.seed)
        runs.append(("plantvillage", support_samples, query_samples, class_names, prompts, args.profile_latency or (args.dataset == "plantvillage")))
    if args.dataset in ["plantdoc", "both"]:
        support_samples, query_samples, class_names, prompts = load_plantdoc(Path(args.plantdoc_dir), args.k_shot, args.seed)
        runs.append(("plantdoc", support_samples, query_samples, class_names, prompts, False))

    for dataset_name, support_samples, query_samples, class_names, prompts, profile_lat in runs:
        results[dataset_name] = run_benchmark(
            dataset_name=dataset_name,
            support_samples=support_samples,
            query_samples=query_samples,
            class_names=class_names,
            taxonomic_prompts=prompts,
            models=models,
            cfg=cfg,
            run_latency_bench=profile_lat,
            use_cache=args.use_cache,
        )

    if args.dataset == "both":
        print("\n" + "=" * 80 + "\n  DUAL-BENCHMARK CROSS-DATASET EMPIRICAL COMPARISON\n" + "=" * 80)
        pv_m = results["plantvillage"]["metrics"]
        pd_m = results["plantdoc"]["metrics"]
        for label, k in [
            ("DINOv2 Small", "fast_dino"),
            ("BioCLIP ViT-B", "deep_bio"),
            ("Base BioDINO", "base_fused"),
            ("BioDINO + Fisher-LDA", "peak_fisher"),
            ("BioDINO Transductive (SOTA)", "refined_transductive"),
        ]:
            print(f"{label:<32} | PV: {pv_m['accuracy_' + k]:>6.2f}% ({pv_m['f1_' + k]:>5.2f}% F1) | PD: {pd_m['accuracy_' + k]:>6.2f}% ({pd_m['f1_' + k]:>5.2f}% F1)")
        print("-" * 80)
        print("  [Hierarchical MoE & Crop-Guided Modes]")
        pv_moe = pv_m.get("hierarchical_moe", {})
        pd_moe = pd_m.get("hierarchical_moe", {})
        if pv_moe and pd_moe:
            for label, k in [
                ("Species Router Top-1", lambda m: m["router_accuracy"]["top1"]),
                ("Species Router Top-2", lambda m: m["router_accuracy"]["top2"]),
                ("Hard AI Router (Top-1)", lambda m: m["hard_router_top1"]["accuracy"]),
                ("Soft AI Router (Top-2)", lambda m: m["soft_router_top2"]["accuracy"]),
                ("Crop-Guided (Farmer Selects)", lambda m: m["crop_guided"]["accuracy"]),
            ]:
                print(f"{label:<32} | PV: {k(pv_moe):>6.2f}% | PD: {k(pd_moe):>6.2f}%")
        print("=" * 80)

    out_json = RESULTS_FILE
    existing = json.load(open(out_json, "r", encoding="utf-8")) if out_json.exists() else {}
    existing.update(results)
    json.dump(existing, open(out_json, "w", encoding="utf-8"), indent=2)
    print(f"\n[OK] Results exported to {out_json}!")


if __name__ == "__main__":
    main()
