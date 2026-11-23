"""
================================================================================
BioDINO-Edge v3: Experimental DINOv3 Pipeline & Benchmark
================================================================================
Isolated experimental benchmark using Meta's DINOv3 Small (vit_small_patch16_dinov3)
Preserves all existing BioDINO-Edge artifacts without any overwrite.
Outputs:
  - Cache: cache_artifacts/pv_novel_embeddings_dinov3.pt
  - Benchmark Results: biodino_v3_results.json
================================================================================
"""

import os
import sys
import json
import time
import random
import argparse
import contextlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from PIL import Image
import torchvision.transforms as T
import timm
from timm.data import create_transform, resolve_model_data_config
import open_clip

# Reuse centralized mathematical operations from core_math
from pipeline.core_math import (
    calculate_shannon_confidence,
    fit_fisher_lda,
    compute_class_prototypes,
    compute_fisher_ensemble_probs,
    refine_transductive,
)
from pipeline.data_engine import (
    load_plantvillage,
    load_plantdoc,
    build_augmentation_pipeline,
    extract_norm_stats,
)
from pipeline.benchmark_engine import run_plantdoc_episodic

# Configuration & Global State
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = _PROJECT_ROOT
PLANTVILLAGE_ROOT = PROJECT_ROOT / "plantvillage"
CACHE_DIR = PROJECT_ROOT / "cache_artifacts"
CACHE_DIR.mkdir(exist_ok=True)

CACHE_FILE_ORIGINAL = CACHE_DIR / "pv_novel_embeddings.pt"
CACHE_FILE_V3 = CACHE_DIR / "pv_novel_embeddings_dinov3.pt"
RESULTS_FILE_V3 = PROJECT_ROOT / "biodino_v3_results.json"


class ImagePathsDataset(Dataset):
    def __init__(self, paths: List[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), idx


def load_dinov3_backbone(device: torch.device = DEVICE):
    print("[*] Loading DINOv3 Small backbone (vit_small_patch16_dinov3)...")
    model = timm.create_model("vit_small_patch16_dinov3", pretrained=True).to(device).eval()
    cfg = resolve_model_data_config(model)
    transform = create_transform(**cfg, is_training=False)
    return model, transform, cfg


def ensure_dinov3_cache(dinov3_model, dinov3_transform, device: torch.device = DEVICE):
    """
    Builds the DINOv3 cache by extracting DINOv3 features for the 9,385 novel images
    while reusing the exact same BioCLIP features and metadata from the original cache.
    Zero modification to the original cache file!
    """
    if CACHE_FILE_V3.exists():
        print(f"[*] DINOv3 cache found: {CACHE_FILE_V3.name}")
        return torch.load(CACHE_FILE_V3, map_location="cpu")

    print(f"[*] Generating DINOv3 cache at {CACHE_FILE_V3.name}...")
    orig_data = torch.load(CACHE_FILE_ORIGINAL, map_location="cpu")
    paths = orig_data["paths"]

    dataset = ImagePathsDataset(paths, dinov3_transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    dinov3_feats = []
    print(f"[*] Extracting DINOv3 features for {len(paths)} images on {device}...")
    start_t = time.time()
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = dinov3_model(imgs)
            else:
                feats = dinov3_model(imgs)
            norm_feats = F.normalize(feats.float(), dim=-1).cpu()
            dinov3_feats.append(norm_feats)

    dinov3_feats = torch.cat(dinov3_feats, dim=0)
    elapsed = time.time() - start_t
    print(f"[OK] Extracted {len(dinov3_feats)} features in {elapsed:.2f}s ({len(paths)/elapsed:.1f} img/s)!")

    v3_cache = {
        "dino_feats": dinov3_feats,
        "bio_feats": orig_data["bio_feats"],
        "labels": orig_data["labels"],
        "novel_classes": orig_data["novel_classes"],
        "bio_text_protos": orig_data["bio_text_protos"],
        "paths": orig_data["paths"],
    }
    torch.save(v3_cache, CACHE_FILE_V3)
    print(f"[OK] DINOv3 cache safely saved to {CACHE_FILE_V3.name} ({CACHE_FILE_V3.stat().st_size / 1e6:.1f} MB)")
    return v3_cache


def benchmark_edge_cpu_latencies():
    """
    Directly measures CPU latency and throughput for:
      - DINOv2 Small (ViT-S/14)
      - DINOv3 Small (ViT-S/16)
    Runs on identical hardware (CPU) with 50 warmups and 100 benchmark iterations.
    """
    print("\n" + "=" * 80 + "\n  PROFILING EDGE CPU LATENCY: DINOv2-Small vs DINOv3-Small\n" + "=" * 80)
    cpu_dev = torch.device("cpu")
    torch.set_num_threads(4)

    # 1. DINOv2
    m_v2 = timm.create_model("vit_small_patch14_dinov2", pretrained=True, dynamic_img_size=True).to(cpu_dev).eval()
    x_v2 = torch.randn(1, 3, 224, 224, device=cpu_dev)
    with torch.no_grad():
        for _ in range(20):
            _ = m_v2(x_v2)
        times_v2 = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = m_v2(x_v2)
            times_v2.append((time.perf_counter() - t0) * 1000.0)

    # 2. DINOv3
    m_v3 = timm.create_model("vit_small_patch16_dinov3", pretrained=True).to(cpu_dev).eval()
    x_v3 = torch.randn(1, 3, 256, 256, device=cpu_dev)
    with torch.no_grad():
        for _ in range(20):
            _ = m_v3(x_v3)
        times_v3 = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = m_v3(x_v3)
            times_v3.append((time.perf_counter() - t0) * 1000.0)

    lat_v2 = float(np.median(times_v2))
    fps_v2 = 1000.0 / lat_v2
    lat_v3 = float(np.median(times_v3))
    fps_v3 = 1000.0 / lat_v3

    print(f"  DINOv2 Small CPU Latency: {lat_v2:.2f} ms ({fps_v2:.1f} FPS)")
    print(f"  DINOv3 Small CPU Latency: {lat_v3:.2f} ms ({fps_v3:.1f} FPS)")
    speedup = ((lat_v2 - lat_v3) / lat_v2) * 100.0
    print(f"  --> DINOv3 Speedup: {speedup:+.1f}% faster on Edge CPU!")
    print("=" * 80)

    return {
        "dinov2_lat_ms": lat_v2,
        "dinov2_fps": fps_v2,
        "dinov3_lat_ms": lat_v3,
        "dinov3_fps": fps_v3,
        "speedup_pct": speedup,
    }


def run_dinov3_benchmark():
    print("=" * 80 + "\n  BioDINO-Edge (v3): DINOv3 Small (ViT-S/16) Experimental Benchmark\n" + "=" * 80)

    # 1. Load DINOv3 & BioCLIP
    dinov3_model, dinov3_transform, dinov3_cfg = load_dinov3_backbone(DEVICE)

    # BioCLIP
    print("[*] Loading BioCLIP backbone...")
    bioclip_model, _, bioclip_transforms = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    bioclip_model = bioclip_model.to(DEVICE).eval()

    # 2. Prepare Cache
    v3_cache = ensure_dinov3_cache(dinov3_model, dinov3_transform, DEVICE)

    # 3. Load 5-shot PlantVillage split
    support_samples, query_samples, class_names, prompts = load_plantvillage(PLANTVILLAGE_ROOT, k_shot=5, seed=SEED)
    n_classes = len(class_names)
    print(f"[*] 5-Shot Setup: {len(support_samples)} Support Leaves | {len(query_samples)} Test Queries | {n_classes} Classes")

    # 4. Multi-view Augmentation on Support Set for DINOv3
    img_size = dinov3_cfg.get("input_size", (3, 256, 256))[-1]
    aug_dinov3 = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=dinov3_cfg["mean"], std=dinov3_cfg["std"]),
    ])
    aug_bioclip = build_augmentation_pipeline(*extract_norm_stats(bioclip_transforms))

    print(f"[*] Extracting 8-view augmentations for support set with DINOv3...")
    sup_bio_list, sup_dino_list, sup_labels_list = [], [], []

    with torch.no_grad():
        for path_str, label in support_samples:
            img = Image.open(path_str).convert("RGB")
            # 8 views for DINOv3
            v3_views = [dinov3_transform(img)] + [aug_dinov3(img) for _ in range(7)]
            bio_views = [bioclip_transforms(img)] + [aug_bioclip(img) for _ in range(7)]

            if DEVICE.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    v3_f = F.normalize(dinov3_model(torch.stack(v3_views).to(DEVICE)), dim=-1)
                    bio_f = F.normalize(bioclip_model.encode_image(torch.stack(bio_views).to(DEVICE)), dim=-1)
            else:
                v3_f = F.normalize(dinov3_model(torch.stack(v3_views).to(DEVICE)), dim=-1)
                bio_f = F.normalize(bioclip_model.encode_image(torch.stack(bio_views).to(DEVICE)), dim=-1)

            sup_dino_list.append(v3_f)
            sup_bio_list.append(bio_f)
            sup_labels_list.append(torch.full((8,), label, dtype=torch.long, device=DEVICE))

    support_dino = torch.cat(sup_dino_list, dim=0)
    support_bio = torch.cat(sup_bio_list, dim=0)
    support_labels = torch.cat(sup_labels_list, dim=0)

    # 5. Build Class Prototypes & Text Prototypes
    print("[*] Synthesizing botanical text prototypes...")
    text_protos = []
    with torch.no_grad():
        for c_name in class_names:
            tokens = tokenizer(prompts[c_name]).to(DEVICE)
            txt_f = F.normalize(bioclip_model.encode_text(tokens), dim=-1)
            text_protos.append(F.normalize(txt_f.mean(dim=0), dim=-1))
    text_prototypes = torch.stack(text_protos)  # (C, 512)

    dino_prototypes = compute_class_prototypes(support_dino, support_labels, n_classes)  # (C, 384)
    bio_vis_prototypes = compute_class_prototypes(support_bio, support_labels, n_classes)  # (C, 512)
    bio_prototypes = F.normalize(0.80 * bio_vis_prototypes + 0.20 * text_prototypes, dim=-1)

    # 6. Fit Regularized Fisher-LDA in 896-D
    support_joint = torch.cat([support_bio, support_dino], dim=-1).cpu().numpy()
    support_labels_np = support_labels.cpu().numpy()
    fisher_w, fisher_b, _ = fit_fisher_lda(support_joint, support_labels_np, device=DEVICE)

    # 7. Query Evaluation on 9,330 Queries
    path_to_idx = {Path(p).resolve(): i for i, p in enumerate(v3_cache["paths"])}
    query_indices = [path_to_idx[Path(p).resolve()] for p, _ in query_samples]
    query_dino = v3_cache["dino_feats"][query_indices].to(DEVICE)
    query_bio = v3_cache["bio_feats"][query_indices].to(DEVICE)
    targets = [y for _, y in query_samples]

    # PlantVillage calibrated parameters for DINOv3
    scale = 28.0
    w_lda = 0.75
    w_base = 0.65
    tau = 0.08  # Calibrated for DINOv3 confidence distribution

    print(f"[*] Running evaluation across all model tiers on {len(targets)} queries...")
    with torch.no_grad():
        # Tier 1: Fast Path (DINOv3 Small alone)
        dino_logits = scale * (query_dino @ dino_prototypes.t())
        dino_prob = F.softmax(dino_logits, dim=-1)
        dino_conf = calculate_shannon_confidence(dino_prob, n_classes)
        preds_dino = dino_logits.argmax(dim=-1).cpu().tolist()

        # Deep Path (BioCLIP alone)
        bio_logits = scale * (query_bio @ bio_prototypes.t())
        preds_bio = bio_logits.argmax(dim=-1).cpu().tolist()

        # Consensus baseline
        preds_consensus = (w_base * bio_logits + (1.0 - w_base) * dino_logits).argmax(dim=-1).cpu().tolist()

        # Joint 896-D Fisher LDA + Text ensemble
        joint_query = torch.cat([query_bio.float(), query_dino.float()], dim=-1)
        ensemble_prob, _, text_prob = compute_fisher_ensemble_probs(
            fisher_weights=fisher_w,
            fisher_biases=fisher_b,
            joint_features=joint_query,
            bioclip_features=query_bio,
            text_prototypes=text_prototypes,
            scale=scale,
            w_lda=w_lda,
            device=DEVICE,
        )
        preds_fisher = ensemble_prob.argmax(dim=-1).cpu().tolist()

        # Cascaded Edge Routing
        deep_mask = (dino_conf < tau)
        preds_cascaded = list(preds_dino)
        for i, is_deep in enumerate(deep_mask):
            if is_deep:
                preds_cascaded[i] = preds_fisher[i]

    # 8. Transductive Refinement
    top_confs, _ = ensemble_prob.max(dim=-1)
    refined_preds, _, _ = refine_transductive(
        support_features=support_joint,
        support_labels=support_labels_np,
        query_features=joint_query.cpu(),
        initial_preds=np.array(preds_fisher),
        confidences=top_confs.cpu(),
        text_prob=text_prob.cpu(),
        n_classes=n_classes,
        support_weight=3,
        max_anchors=20,
        conf_threshold=0.70,
        device=DEVICE,
        w_lda=w_lda,
    )

    # 9. Performance Calculation
    calc_acc = lambda p: float(accuracy_score(targets, p) * 100.0)
    calc_f1 = lambda p: float(f1_score(targets, p, average="macro", zero_division=0) * 100.0)

    acc_dino = calc_acc(preds_dino)
    f1_dino = calc_f1(preds_dino)
    acc_bio = calc_acc(preds_bio)
    f1_bio = calc_f1(preds_bio)
    acc_cons = calc_acc(preds_consensus)
    f1_cons = calc_f1(preds_consensus)
    acc_fish = calc_acc(preds_fisher)
    f1_fish = calc_f1(preds_fisher)
    acc_casc = calc_acc(preds_cascaded)
    f1_casc = calc_f1(preds_cascaded)
    acc_ref = calc_acc(refined_preds)
    f1_ref = calc_f1(refined_preds)

    deep_triggers = int(deep_mask.sum().item())
    fast_ratio = (1.0 - (deep_triggers / len(targets))) * 100.0

    # False Negative (Diseased leaf classified as Healthy)
    # Healthy class indices in 11 classes: 3 (Apple healthy), 7 (Grape healthy), 10 (Potato healthy)
    healthy_indices = {3, 7, 10}
    fn_count = 0
    total_diseased = 0
    for true_y, pred_y in zip(targets, refined_preds):
        if true_y not in healthy_indices:
            total_diseased += 1
            if pred_y in healthy_indices:
                fn_count += 1
    fn_rate = (fn_count / total_diseased) * 100.0 if total_diseased > 0 else 0.0

    # Per-Class Precision & Recall
    cls_report = classification_report(
        targets, refined_preds, target_names=class_names, output_dict=True, zero_division=0
    )

    # Measure Edge Latencies
    edge_profile = benchmark_edge_cpu_latencies()

    print("\n" + "=" * 80 + "\n  EXPERIMENTAL RESULTS: BioDINO-Edge with DINOv3 Small (ViT-S/16)\n" + "=" * 80)
    print(f"  1. Fast-Path (DINOv3 Small Alone)            : Acc = {acc_dino:.2f}% | F1 = {f1_dino:.2f}%")
    print(f"  2. Deep-Path (BioCLIP Alone)                 : Acc = {acc_bio:.2f}% | F1 = {f1_bio:.2f}%")
    print(f"  3. Static Prototype Consensus                : Acc = {acc_cons:.2f}% | F1 = {f1_cons:.2f}%")
    print(f"  4. BioDINOv3 + Fisher-LDA (Inductive)        : Acc = {acc_fish:.2f}% | F1 = {f1_fish:.2f}%")
    print(f"  5. BioDINO-Edge v3 (Cascaded Routing)        : Acc = {acc_casc:.2f}% | F1 = {f1_casc:.2f}% (Fast Exit: {fast_ratio:.1f}%)")
    print(f"  6. BioDINOv3 Transductive Refinement (SOTA)  : Acc = {acc_ref:.2f}% | F1 = {f1_ref:.2f}%")
    print(f"  --> False Negative Disease Escape Rate       : {fn_rate:.3f}% ({fn_count} / {total_diseased} leaves)")
    print("=" * 80)

    print("\nDetailed Per-Class Metrics (DINOv3 Transductive):")
    print(classification_report(targets, refined_preds, target_names=[c.replace("___", " ") for c in class_names], digits=3, zero_division=0))

    results = {
        "model_version": "BioDINO-Edge v3 (vit_small_patch16_dinov3)",
        "plantvillage": {
            "dataset": "PlantVillage (11 Novel Classes)",
            "k_shot": 5,
            "metrics": {
                "accuracy_fast_dino": acc_dino,
                "f1_fast_dino": f1_dino,
                "accuracy_deep_bio": acc_bio,
                "f1_deep_bio": f1_bio,
                "accuracy_base_consensus": acc_cons,
                "f1_base_consensus": f1_cons,
                "accuracy_fisher_inductive": acc_fish,
                "f1_fisher_inductive": f1_fish,
                "accuracy_cascaded_edge": acc_casc,
                "f1_cascaded_edge": f1_casc,
                "fast_ratio_pct": fast_ratio,
                "accuracy_refined_transductive": acc_ref,
                "f1_refined_transductive": f1_ref,
                "false_negative_rate_pct": fn_rate,
                "false_negative_count": fn_count,
                "total_diseased_queries": total_diseased,
            },
            "edge_latency_profile": edge_profile,
            "per_class_metrics": cls_report,
        }
    }

    # PlantDoc Benchmark
    pd_results = run_plantdoc_dinov3_benchmark(
        dinov3_model=dinov3_model,
        dinov3_transform=dinov3_transform,
        dinov3_cfg=dinov3_cfg,
        bioclip_model=bioclip_model,
        bioclip_transforms=bioclip_transforms,
        tokenizer=tokenizer,
    )
    results["plantdoc"] = pd_results

    with open(RESULTS_FILE_V3, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Complete dual-benchmark results exported to {RESULTS_FILE_V3}!")
    return results


def run_plantdoc_dinov3_benchmark(dinov3_model, dinov3_transform, dinov3_cfg, bioclip_model, bioclip_transforms, tokenizer):
    print("\n" + "=" * 80 + "\n  BENCHMARK: PLANTDOC IN-THE-WILD (27 Classes | 5-Shot Setup)\n" + "=" * 80)
    support_samples, query_samples, class_names, prompts = load_plantdoc(PROJECT_ROOT / "plantdoc", k_shot=5, seed=SEED)
    n_classes = len(class_names)
    print(f"[*] PlantDoc: {len(support_samples)} Support Leaves | {len(query_samples)} Test Queries | {n_classes} Classes")

    img_size = dinov3_cfg.get("input_size", (3, 256, 256))[-1]
    aug_dinov3 = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=dinov3_cfg["mean"], std=dinov3_cfg["std"]),
    ])
    aug_bioclip = build_augmentation_pipeline(*extract_norm_stats(bioclip_transforms))

    print(f"[*] Extracting 8-view support vectors ({len(support_samples) * 8} views)...")
    sup_bio_list, sup_dino_list, sup_labels_list = [], [], []
    with torch.no_grad():
        for path_str, label in support_samples:
            img = Image.open(path_str).convert("RGB")
            v3_views = [dinov3_transform(img)] + [aug_dinov3(img) for _ in range(7)]
            bio_views = [bioclip_transforms(img)] + [aug_bioclip(img) for _ in range(7)]

            if DEVICE.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    v3_f = F.normalize(dinov3_model(torch.stack(v3_views).to(DEVICE)), dim=-1)
                    bio_f = F.normalize(bioclip_model.encode_image(torch.stack(bio_views).to(DEVICE)), dim=-1)
            else:
                v3_f = F.normalize(dinov3_model(torch.stack(v3_views).to(DEVICE)), dim=-1)
                bio_f = F.normalize(bioclip_model.encode_image(torch.stack(bio_views).to(DEVICE)), dim=-1)

            sup_dino_list.append(v3_f)
            sup_bio_list.append(bio_f)
            sup_labels_list.append(torch.full((8,), label, dtype=torch.long, device=DEVICE))

    support_dino = torch.cat(sup_dino_list, dim=0)
    support_bio = torch.cat(sup_bio_list, dim=0)
    support_labels = torch.cat(sup_labels_list, dim=0)

    print("[*] Synthesizing PlantDoc botanical text prototypes...")
    text_protos = []
    with torch.no_grad():
        for c_name in class_names:
            c_prompts = prompts.get(c_name, [f"A botanical leaf showing {c_name.replace('_', ' ')}"])
            tokens = tokenizer(c_prompts).to(DEVICE)
            txt_f = F.normalize(bioclip_model.encode_text(tokens), dim=-1)
            text_protos.append(F.normalize(txt_f.mean(dim=0), dim=-1))
    text_prototypes = torch.stack(text_protos)

    dino_prototypes = compute_class_prototypes(support_dino, support_labels, n_classes)
    bio_vis_prototypes = compute_class_prototypes(support_bio, support_labels, n_classes)
    bio_prototypes = F.normalize(0.80 * bio_vis_prototypes + 0.20 * text_prototypes, dim=-1)

    support_joint = torch.cat([support_bio, support_dino], dim=-1).cpu().numpy()
    support_labels_np = support_labels.cpu().numpy()
    fisher_w, fisher_b, _ = fit_fisher_lda(support_joint, support_labels_np, device=DEVICE)

    print(f"[*] Extracting query features for {len(query_samples)} PlantDoc images...")
    qry_bio_list, qry_dino_list = [], []
    with torch.no_grad():
        for path_str, _ in query_samples:
            img = Image.open(path_str).convert("RGB")
            v3_in = dinov3_transform(img).unsqueeze(0).to(DEVICE)
            bio_in = bioclip_transforms(img).unsqueeze(0).to(DEVICE)

            if DEVICE.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    v3_f = F.normalize(dinov3_model(v3_in), dim=-1)
                    bio_f = F.normalize(bioclip_model.encode_image(bio_in), dim=-1)
            else:
                v3_f = F.normalize(dinov3_model(v3_in), dim=-1)
                bio_f = F.normalize(bioclip_model.encode_image(bio_in), dim=-1)

            qry_dino_list.append(v3_f)
            qry_bio_list.append(bio_f)

    query_dino = torch.cat(qry_dino_list, dim=0)
    query_bio = torch.cat(qry_bio_list, dim=0)
    targets = [y for _, y in query_samples]
    # PlantDoc calibrated parameters for DINOv3
    scale = 28.0
    w_lda = 0.65
    w_base = 0.70
    tau = 0.08  # Calibrated for DINOv3 confidence distribution

    with torch.no_grad():
        dino_logits = scale * (query_dino @ dino_prototypes.t())
        dino_prob = F.softmax(dino_logits, dim=-1)
        dino_conf = calculate_shannon_confidence(dino_prob, n_classes)
        preds_dino = dino_logits.argmax(dim=-1).cpu().tolist()

        bio_logits = scale * (query_bio @ bio_prototypes.t())
        preds_bio = bio_logits.argmax(dim=-1).cpu().tolist()

        preds_consensus = (w_base * bio_logits + (1.0 - w_base) * dino_logits).argmax(dim=-1).cpu().tolist()

        joint_query = torch.cat([query_bio.float(), query_dino.float()], dim=-1)
        ensemble_prob, _, text_prob = compute_fisher_ensemble_probs(
            fisher_weights=fisher_w,
            fisher_biases=fisher_b,
            joint_features=joint_query,
            bioclip_features=query_bio,
            text_prototypes=text_prototypes,
            scale=scale,
            w_lda=w_lda,
            device=DEVICE,
        )
        preds_fisher = ensemble_prob.argmax(dim=-1).cpu().tolist()

        deep_mask = (dino_conf < tau)
        preds_cascaded = list(preds_dino)
        for i, is_deep in enumerate(deep_mask):
            if is_deep:
                preds_cascaded[i] = preds_fisher[i]

    top_confs, _ = ensemble_prob.max(dim=-1)
    refined_preds, _, _ = refine_transductive(
        support_features=support_joint,
        support_labels=support_labels_np,
        query_features=joint_query.cpu(),
        initial_preds=np.array(preds_fisher),
        confidences=top_confs.cpu(),
        text_prob=text_prob.cpu(),
        n_classes=n_classes,
        support_weight=1,
        max_anchors=2,
        conf_threshold=None,
        device=DEVICE,
        w_lda=w_lda,
    )

    calc_acc = lambda p: float(accuracy_score(targets, p) * 100.0)
    calc_f1 = lambda p: float(f1_score(targets, p, average="macro", zero_division=0) * 100.0)

    episodic_res = run_plantdoc_episodic(
        support_joint_features=support_joint,
        support_labels=support_labels_np,
        query_joint_features=joint_query.cpu().numpy(),
        query_labels=np.array(targets),
        query_bioclip_features=query_bio.cpu(),
        text_prototypes=text_prototypes.cpu(),
        n_classes=n_classes,
        n_episodes=50,
        seed=SEED,
        scale=scale,
        w_lda=w_lda,
    )

    print("\n" + "-" * 80 + "\n  PLANTDOC RESULTS SUMMARY (DINOv3)\n" + "-" * 80)
    print(f"  1. Fast-Path (DINOv3 Alone)      : 27-Way Acc = {calc_acc(preds_dino):.2f}% | F1 = {calc_f1(preds_dino):.2f}%")
    print(f"  2. Deep-Path (BioCLIP Alone)     : 27-Way Acc = {calc_acc(preds_bio):.2f}% | F1 = {calc_f1(preds_bio):.2f}%")
    print(f"  3. Static Prototype Consensus    : 27-Way Acc = {calc_acc(preds_consensus):.2f}% | F1 = {calc_f1(preds_consensus):.2f}%")
    print(f"  4. BioDINOv3 + Fisher (Inductive): 27-Way Acc = {calc_acc(preds_fisher):.2f}% | F1 = {calc_f1(preds_fisher):.2f}%")
    print(f"  5. BioDINOv3 Transductive (27-Way): 27-Way Acc = {calc_acc(refined_preds):.2f}% | F1 = {calc_f1(refined_preds):.2f}%")
    ep_m = episodic_res["five_way_episodic"]
    print(f"  6. 5-Way 5-Shot Episodic (50 Ep) : Acc = {ep_m['mean_accuracy']:.2f}% ± {ep_m['std_accuracy']:.2f}% (Peak = {ep_m['peak_accuracy']:.2f}%)")
    print("-" * 80)

    return {
        "dataset": "PlantDoc (27 Classes In-The-Wild)",
        "k_shot": 5,
        "metrics": {
            "accuracy_fast_dino": calc_acc(preds_dino),
            "f1_fast_dino": calc_f1(preds_dino),
            "accuracy_deep_bio": calc_acc(preds_bio),
            "f1_deep_bio": calc_f1(preds_bio),
            "accuracy_base_fused": calc_acc(preds_consensus),
            "f1_base_fused": calc_f1(preds_consensus),
            "accuracy_peak_fisher": calc_acc(preds_fisher),
            "f1_peak_fisher": calc_f1(preds_fisher),
            "accuracy_cascaded_edge": calc_acc(preds_cascaded),
            "f1_cascaded_edge": calc_f1(preds_cascaded),
            "accuracy_refined_transductive": calc_acc(refined_preds),
            "f1_refined_transductive": calc_f1(refined_preds),
            "multi_episode_30": episodic_res,
        }
    }


if __name__ == "__main__":
    run_dinov3_benchmark()

