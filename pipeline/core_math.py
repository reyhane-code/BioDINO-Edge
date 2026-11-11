"""
================================================================================
BioDINO-Edge: Core Mathematical & Algorithmic Operations (core_math.py)
================================================================================
Centralized mathematical foundations:
  1. calculate_shannon_confidence: Shannon margin-entropy routing gate metric.
  2. fit_fisher_lda: Closed-form regularized Fisher-LDA with Ledoit-Wolf shrinkage.
  3. compute_class_prototypes: L2-normalized mean class prototype synthesis.
  4. compute_fisher_ensemble_probs: Visual LDA + BioCLIP text late ensemble fusion.
  5. refine_transductive: Transductive pseudo-label recalibration in closed form.
================================================================================
"""

from typing import Dict, List, Literal, Optional, Tuple, Union, overload
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


@overload
def calculate_shannon_confidence(
    prob: torch.Tensor,
    n_classes: int,
    return_details: Literal[False] = False,
) -> torch.Tensor:
    ...


@overload
def calculate_shannon_confidence(
    prob: torch.Tensor,
    n_classes: int,
    return_details: Literal[True],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ...


@overload
def calculate_shannon_confidence(
    prob: torch.Tensor,
    n_classes: int,
    return_details: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    ...


def calculate_shannon_confidence(
    prob: torch.Tensor,
    n_classes: int,
    return_details: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Shannon margin-entropy heuristic confidence gating metric.
    
    Computes:
        Confidence C(x) = M(x) * (1 - H_norm(x))
    where:
        M(x) is the margin between top-1 and top-2 class probabilities,
        H_norm(x) is the Shannon entropy normalized by ln(K) into [0, 1].
    
    Args:
        prob: Class probability tensor of shape (N, n_classes) or (n_classes,).
        n_classes: Total number of active novel classes.
        return_details: If True, returns (confidence, margin, normalized_entropy).
        
    Returns:
        torch.Tensor of shape (N,) with non-negative confidence scores in [0, 1],
        or tuple of (confidence, margin, normalized_entropy) if return_details is True.
    """
    top_p, _ = torch.topk(prob, k=min(2, n_classes), dim=-1)
    margin = top_p[..., 0] - top_p[..., 1] if n_classes > 1 else top_p[..., 0]
    norm_factor = max(float(np.log(n_classes)), 1e-6)
    entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=-1) / norm_factor
    confidence = (margin * (1.0 - entropy)).clamp(min=0.0)
    if return_details:
        return confidence, margin, entropy
    return confidence


# Backwards-compatibility alias
calc_confidence = calculate_shannon_confidence


def fit_fisher_lda(
    features: np.ndarray,
    labels: np.ndarray,
    device: Union[torch.device, str] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, LinearDiscriminantAnalysis]:
    
    dev = torch.device(device)
    classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(features, labels)
    fisher_weights = torch.tensor(classifier.coef_, dtype=torch.float32, device=dev)
    fisher_biases = torch.tensor(classifier.intercept_, dtype=torch.float32, device=dev)
    return fisher_weights, fisher_biases, classifier


def compute_class_prototypes(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    """Computes L2-normalized class prototypes by averaging support embeddings per class."""
    return torch.stack([
        F.normalize(features[labels == c].mean(dim=0), dim=-1)
        for c in range(n_classes)
    ])


    # Centralized Fisher-LDA + BioCLIP Taxonomic Text ensemble probability predictor.
    # Eliminates redundant matrix multiplication, softmax scaling, and convex combination.
    
    # Computes:
    #     1. joint_features = [bioclip_features, dino_features] (if not directly supplied)
    #     2. lda_prob = Softmax(joint_features @ fisher_weights.T + fisher_biases)
    #     3. text_prob = Softmax(scale * (bioclip_features @ text_prototypes.T))
    #     4. ensemble_prob = w_lda * lda_prob + (1.0 - w_lda) * text_prob
        
    # Returns:
    #     ensemble_prob: Ensemble class probability distribution tensor (N, n_classes).
    #     lda_prob: Pure visual Fisher-LDA class probability distribution tensor (N, n_classes).
    #     text_prob: Pure textual taxonomic class probability distribution tensor (N, n_classes).
def compute_fisher_ensemble_probs(
    fisher_weights: torch.Tensor,
    fisher_biases: torch.Tensor,
    joint_features: Optional[Union[torch.Tensor, np.ndarray]] = None,
    bioclip_features: Optional[torch.Tensor] = None,
    dino_features: Optional[torch.Tensor] = None,
    text_prob: Optional[torch.Tensor] = None,
    text_prototypes: Optional[torch.Tensor] = None,
    scale: float = 20.0,
    w_lda: float = 0.75,
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    dev = torch.device(device) if device is not None else fisher_weights.device

    # 1. Prepare joint representation on target device
    if joint_features is None:
        if bioclip_features is not None and dino_features is not None:
            joint_features_t = torch.cat([bioclip_features.to(dev).float(), dino_features.to(dev).float()], dim=-1)
        else:
            raise ValueError("Either joint_features or both (bioclip_features, dino_features) must be provided.")
    elif isinstance(joint_features, np.ndarray):
        joint_features_t = torch.tensor(joint_features, dtype=torch.float32, device=dev)
    else:
        joint_features_t = joint_features.to(dev).float()

    W_t = fisher_weights.to(dev)
    b_t = fisher_biases.to(dev)

    # 2. Visual Fisher-LDA projection
    lda_logits = joint_features_t @ W_t.t() + b_t
    lda_prob = F.softmax(lda_logits, dim=-1)

    # 3. Taxonomic BioCLIP text prototype projection
    if text_prob is not None:
        txt_prob = text_prob.to(dev)
    elif bioclip_features is not None and text_prototypes is not None:
        f_bio = bioclip_features.to(dev)
        p_txt = text_prototypes.to(dev)
        text_logits = scale * (f_bio @ p_txt.t())
        txt_prob = F.softmax(text_logits, dim=-1)
    else:
        # Fallback to pure visual Fisher LDA if no text modality is available
        return lda_prob, lda_prob, lda_prob

    # 4. Convex late ensemble fusion
    ensemble_prob = w_lda * lda_prob + (1.0 - w_lda) * txt_prob
    return ensemble_prob, lda_prob, txt_prob


def refine_transductive(
    support_features: np.ndarray,
    support_labels: np.ndarray,
    query_features: torch.Tensor,
    initial_preds: np.ndarray,
    confidences: torch.Tensor,
    text_prob: torch.Tensor,
    n_classes: int,
    support_weight: int = 3,
    max_anchors: int = 20,
    conf_threshold: Optional[float] = None,
    device: Union[torch.device, str] = "cpu",
    w_lda: float = 0.75,
) -> Tuple[np.ndarray, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Transductive closed-form pseudo-label recalibration with strict threshold gating.
    
    Selects high-confidence query predictions as pseudo-labeled anchors, repeats the
    ground-truth support set by `support_weight` to prevent manifold drift, and re-fits
    Fisher LDA in closed form (<25ms) without backpropagation or gradient descent.
    
    Args:
        support_features: Support set concatenated feature embeddings (N_support, 896).
        support_labels: Support set ground-truth integer class labels (N_support,).
        query_features: Query set concatenated feature embeddings (N_query, 896).
        initial_preds: Initial inductive class predictions for query samples (N_query,).
        confidences: Confidence values (top-1 probabilities) for query samples (N_query,).
        text_prob: Zero-shot textual prototype probabilities from BioCLIP (N_query, n_classes).
        n_classes: Total number of novel classes.
        support_weight: Re-weighting factor for canonical ground-truth support samples.
        max_anchors: Maximum number of high-confidence query anchors per class.
        conf_threshold: Minimum confidence required for a query sample to become an anchor.
        device: Target execution device.
        w_lda: Weight for Fisher-LDA in late ensemble.
        
    Returns:
        refined_preds: Updated class predictions array of shape (N_query,).
        refined_prob: Updated ensemble probabilities tensor of shape (N_query, n_classes).
        (refined_weights, refined_biases): Refined Fisher LDA weight matrix and bias vector.
    """
    dev = torch.device(device)
    refinement_features = [support_features] * support_weight
    refinement_labels = [support_labels] * support_weight

    for c in range(n_classes):
        class_mask = (initial_preds == c)
        if not np.any(class_mask):
            continue

        class_query_indices = np.where(class_mask)[0]
        class_confidences = confidences[class_query_indices]

        # Strict threshold check: discard candidates if none pass threshold
        if conf_threshold is not None:
            high_mask = class_confidences >= conf_threshold
            if high_mask.sum() == 0:
                continue
            high_mask_np = high_mask.cpu().numpy() if isinstance(high_mask, torch.Tensor) else high_mask
            class_query_indices = class_query_indices[high_mask_np]
            class_confidences = class_confidences[high_mask]

        # Select top-k anchors with highest confidence for class c
        k_anchors = min(max_anchors, len(class_query_indices))
        if k_anchors > 0:
            top_indices = torch.topk(class_confidences, k_anchors).indices
            best_indices = class_query_indices[top_indices.cpu().numpy()]
            selected_query_feats = query_features[best_indices]
            refinement_features.append(
                selected_query_feats.cpu().numpy()
                if isinstance(selected_query_feats, torch.Tensor)
                else selected_query_feats
            )
            refinement_labels.append(np.full(k_anchors, c))

    # Concatenate all support replicates and pseudo-labeled anchors
    concat_features = np.concatenate(refinement_features, axis=0)
    concat_labels = np.concatenate(refinement_labels, axis=0)

    # Re-fit regularized Fisher LDA in closed form (<25ms)
    refined_weights, refined_biases, _ = fit_fisher_lda(concat_features, concat_labels, device=dev)

    # Compute refined ensemble probabilities via centralized helper
    refined_prob, _, _ = compute_fisher_ensemble_probs(
        fisher_weights=refined_weights,
        fisher_biases=refined_biases,
        joint_features=query_features,
        text_prob=text_prob,
        w_lda=w_lda,
        device=dev,
    )
    refined_preds = refined_prob.argmax(dim=-1).cpu().numpy()
    return refined_preds, refined_prob, (refined_weights, refined_biases)
