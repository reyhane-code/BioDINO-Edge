# Response to Reviewer — Comment #8 (Fair and Reproducible Baseline Comparison)

**Manuscript:** BioDINO-Edge: Gradient-Free Cascades of Frozen Vision Foundation Models for On-Device Few-Shot Plant Disease Diagnosis
**Target journal:** Engineering Applications of Artificial Intelligence (Elsevier)
**Source of truth:** `experiments/baseline_comparison/controlled_baseline_results.json` (controlled re-runs); `experiments/transductive_validation/transductive_validation_results.json` (canonical reference run)
**Reproduction script:** `experiments/baseline_comparison/run_controlled_baselines.py`
**Revised manuscript location:** Table `tab:comparative` (controlled comparison), Table `tab:literature` (as-reported context), Appendix `app:repro` (reproducibility) and Table `tab:provenance`.

---

## Reviewer comment (restated)

> In Table I, BioDINO-Edge is compared with several general and specialized few-shot methods. However:
> **(a)** It is not clear whether all baselines used identical dataset splits, the same K-shot support count, similar preprocessing, and the same evaluation protocol.
> **(b)** It must be specified which results were obtained by re-running the methods and which were taken directly from published papers.
> **(c)** Comparing experiments with different class counts (e.g., 5-Way vs. 27-Way) without a controlled run under identical conditions cannot count as direct evidence of superiority.
> **(d)** For reproducibility, provide the exact splits, random seeds, model versions, hyperparameter settings, and the code version corresponding to the reported results.

---

## Summary of response

We thank the reviewer for this decisive comment. We agree that the original comparative table mixed author-supplied numbers with literature values under non-identical protocols, and that some cross-regime comparisons were not valid evidence of superiority. We have rebuilt the comparison from scratch:

1. **Controlled re-runs (answers a, c).** Every baseline is now re-run by us on the *identical* frozen BioCLIP+DINOv2 embeddings, support/query splits, seeds, K-shot counts, and evaluation protocol as BioDINO-Edge. No backbone is retrained, so the comparison isolates the adaptation algorithm, not the encoder. We add the standard **5-way K-shot ladder (K = 1, 5, 10, 15)** used by VSF and AgriFewNet so our results can be compared *in-regime*.
2. **Explicit provenance (answers b).** A new table lists specialized architectures *as reported* in their original publications, clearly separated from our controlled re-runs, with a per-value provenance table in the appendix.
3. **Withdrawn invalid claims (answers c).** We removed the cross-protocol superiority claims (6.90 / 13.00 / 29.37 percentage-point comparisons) and corrected misattributed literature numbers.
4. **Reproducibility appendix (answers d).** Exact splits, seeds, model versions, hyperparameters, pinned environment, and code paths are now specified.

We also corrected the literature values themselves: the original table's AgriFewNet (88.45 / 86.12) and the placement of VSF's 90.51 in the 11-class column did not match the published papers (Section 5).

---

## 1. Controlled comparison protocol — answers (a)

*Manuscript Table `tab:comparative`. Dataset: PlantVillage 11-class novel subset (Apple/Grape/Potato; 9,385 images), frozen embeddings cached once.*

All methods share, **within each episode**, the identical support indices, query indices, and frozen features:

- **Backbones (frozen, no fine-tuning):** DINOv2 ViT-S/14 (`timm vit_small_patch14_dinov2`, 384-D) + BioCLIP ViT-B/16 (`open_clip hf-hub:imageomics/bioclip`, 512-D); joint `z = [bio || dino] ∈ R^896`.
- **11-class regime:** 30 Monte-Carlo episodes; episode `e` uses `random.Random(42 + 17e)`; per class, shuffle indices, first 5 = support, all remaining = query (9,330 queries/episode; support ∩ query = ∅).
- **5-way K-shot regime:** 50 episodes per `K ∈ {1,5,10,15}`; a single `random.Random(42)` stream samples 5 of 11 classes (`rng.sample(range(11),5)`), shuffles each class, first `K` = support, rest = query.
- **Fixed hyperparameters:** logit scale γ = 20.0, Fisher blend w_lda = 0.75, gate τ = 0.35; transductive support weight 3×, max anchors 20/class, confidence threshold 0.70; Tip-Adapter α = 1.0, β = 5.5.

**Baselines (training-free, correctly referenced):** Zero-Shot CLIP/BioCLIP [Radford 2021; Stevens 2024], Tip-Adapter [Zhang 2022], Prototypical Networks [Snell 2017], SimpleShot [Wang 2019]. Prototypical Networks and SimpleShot coincide exactly because Euclidean and cosine rankings are identical on ℓ2-normalized features.

### Controlled results (mean ± std over episodes, %)

| Method | 11-class Acc | 11-class F1 | 5-way 1-shot | 5-way 5-shot | 5-way 10-shot | 5-way 15-shot |
|---|---|---|---|---|---|---|
| Zero-Shot CLIP/BioCLIP | 19.18 ± 0.02 | 18.49 | 38.89 ± 12.78 | 38.86 ± 12.79 | 38.83 ± 12.81 | 38.79 ± 12.83 |
| Tip-Adapter | 49.97 ± 1.62 | 46.54 | 45.15 ± 13.42 | 68.63 ± 12.80 | 86.01 ± 7.49 | 93.23 ± 4.00 |
| Prototypical Networks | 93.28 ± 1.24 | 92.47 | 88.44 ± 9.13 | 97.45 ± 2.03 | 98.35 ± 1.45 | 98.57 ± 1.27 |
| SimpleShot | 93.28 ± 1.24 | 92.47 | 88.44 ± 9.13 | 97.45 ± 2.03 | 98.35 ± 1.45 | 98.57 ± 1.27 |
| BioDINO-Edge (Inductive) | 95.08 ± 1.27 | 94.51 | 88.45 ± 9.12 | 97.96 ± 1.85 | 99.10 ± 0.87 | 99.41 ± 0.60 |
| **BioDINO-Edge (Transductive, Ours)** | **97.41 ± 0.71** | **97.09** | **93.42 ± 6.83** | **98.77 ± 1.68** | **99.41 ± 0.60** | **99.63 ± 0.39** |

**Honest reading.** In the easy 5-way regime the centroid baselines already reach 97.45% (5-shot) / 98.57% (15-shot), so the margin narrows; BioDINO-Edge leads at every K, and its transductive gain is largest in the hardest 1-shot setting (93.42% vs. 88.44%). The decisive gains are in the harder 11-class regime (97.41% vs. 93.28% for the strongest baseline). We no longer claim uniform dominance "across every regime."

---

## 2. Provenance: re-run vs. as-reported — answers (b)

*Manuscript Table `tab:literature` and Appendix Table `tab:provenance`.*

| Value | Provenance |
|---|---|
| Table `tab:comparative` baselines | **Re-run by us**, identical splits/seeds/features |
| Table `tab:comparative` BioDINO 11-class | Canonical reference run (reproduced ≤ 0.04 pp by the script) |
| Table `tab:comparative` BioDINO 5-way K-shot | **Controlled re-run** (this work) |
| Table `tab:literature` AgriFewNet / VSF / Zhao | **As reported** in original publications, *not* re-run |
| PlantDoc 83.61% / 56.35% | Our PlantDoc field-domain run (labeled target-domain support; protocol in appendix) |

### Literature values, as reported (not a controlled head-to-head)

| Method | Dataset | Regime | Reported Acc (%) | Source |
|---|---|---|---|---|
| AgriFewNet | PlantVillage / New PV | 1/5/10/15-shot | 87.3 / 94.8 / 97.1 / 98.2 | Babu et al., *Appl. Sci.* 15(23):12787, 2025 |
| VSF | PlantVillage | 5-way 1/5-shot | 79.01 / 90.51 | Zhou et al., *AgriEngineering* 8(10):407, 2026 |
| VSF | PlantDoc | 5-way 1/5-shot | 59.82 / 70.61 | Zhou et al., *AgriEngineering* 8(10):407, 2026 |
| Zhao et al. | PV → PlantDoc (cross-domain) | 1/5-shot | 42.69 ± 0.62 / 54.24 ± 0.72 | Zhao et al., *Sensors* 26(12):3632, 2026 |

These use different encoders, class subsets, splits, and preprocessing. They provide **in-regime context only**; we draw **no direct superiority claim** from them.

---

## 3. In-regime comparison and removed cross-protocol claims — answers (c)

The reviewer is correct that 5-Way vs. 27-Way comparisons under different conditions are not direct evidence of superiority. We restructured accordingly:

- The **controlled table** contains only regimes run under identical conditions (11-class and 5-way K-shot on the same PlantVillage subset). The 27-way all-class stress test is **not** placed head-to-head with 5-way literature values.
- The **5-way K-shot ladder (K = 1, 5, 10, 15)** matches the measurement protocols of VSF (5-way 1/5-shot) and AgriFewNet (1/5/10/15-shot), enabling *in-regime* comparison (same regime, different splits — explicitly caveated).
- **PlantDoc** results (83.61% 5-way 5-shot; 56.35% 27-way) are reported as our field-domain results (labeled target-domain support); literature PlantDoc values (VSF 70.61, Zhao 54.24) are listed separately as context, not as a controlled comparison.

**Withdrawn claims (removed from the manuscript):**

| Earlier claim | Why invalid | Disposition |
|---|---|---|
| "outperforms VSF by 6.90 pp (97.41 vs. 90.51)" | 97.41 is 11-class; 90.51 is VSF's **5-way 5-shot** — different regimes | **Removed** |
| "surpassing VSF by 13.00 pp" (PlantDoc) | Different splits/encoders; not controlled | **Removed** |
| "Zhao by 29.37 points" | Different protocol (cross-domain calibration) | **Removed** |
| "outperforms all baselines across every regime" | Margin narrows in easy 5-way regime | **Rescoped** to per-regime statements |

---

## 4. Reproducibility — answers (d)

*Manuscript Appendix `app:repro`.*

- **Splits & seeds:** PV 11-class = 30 episodes, seed `42 + 17e`, per-class shuffle, 5-shot support / rest query. PV 5-way = 50 episodes per K, seed 42, `rng.sample(range(11),5)`. PlantDoc 5-way = 50 episodes, seed 42, `rng.sample(range(27),5)`; 27-way = 135 support (8-view → 1,080) / 252 queries.
- **Model versions:** `timm vit_small_patch14_dinov2` (ViT-S/14, 384-D); `open_clip hf-hub:imageomics/bioclip` (ViT-B/16, 512-D); both frozen.
- **Hyperparameters:** γ = 20.0, w_lda = 0.75, τ = 0.35; transductive (3×, 20 anchors, 0.70); Tip-Adapter (α = 1.0, β = 5.5); Fisher-LDA = Ledoit–Wolf shrinkage (`sklearn LDA solver=lsqr, shrinkage=auto`).
- **Pinned environment:** Python 3.12.1, torch 2.6.0 (CUDA 12.4 build; all experiments executed on CPU), scikit-learn 1.9.1, numpy 2.5.4, scipy 1.18.1, timm 1.0.30, open_clip 3.3.0.
- **Code:** `run_controlled_baselines.py` → `controlled_baseline_results.json` (controlled comparison); `benchmark_engine.py` + `run_transductive_validation.py` (canonical reference). Repository: `github.com/reyhane-code/BioDINO-Edge-android` (release pinned).

**1-shot singularity handling.** At 5-way 1-shot the raw support has N = C = 5 and the within-class scatter vanishes, so the canonical sklearn LDA (which requires N > C) is undefined. We resolve this with a data-driven ridge floor (mean support feature variance) added to the Ledoit–Wolf covariance, reducing Fisher to regularized nearest-centroid — the correct degenerate limit. For N > C the canonical estimator is used unchanged. This is documented in the appendix.

**Numerical reproducibility (disclosed).** The controlled script reproduces the canonical **inductive** accuracy exactly (95.08%) and the **transductive** accuracy within **0.04 pp** (97.37% re-run vs. 97.41% reference). The residual is floating-point sensitivity of the iterative `lsqr` LDA solve across library versions, far below the episode std (0.71 pp). The centroid/cosine baselines involve no iterative solve and reproduce exactly. We report the canonical reference values for the 11-class regime to stay consistent with the paired-significance and sensitivity analyses.

---

## 5. Explicit corrections to the original table (honesty statement)

| Original entry | Problem | Correction |
|---|---|---|
| AgriFewNet 88.45 / 86.12 (11-class) | Matches nothing published; wrong regime | As-reported **87.3 / 94.8 / 97.1 / 98.2** at 1/5/10/15-shot (Table `tab:literature`) |
| VSF 90.51 in the **11-class** column | 90.51 is VSF's **5-way 5-shot** PlantVillage number | Moved to literature table under the correct regime |
| Zero-Shot CLIP 21.20 / Tip-Adapter 79.40 / CLIP+LoRA 86.10 | Author-supplied; no reproducible backing | Re-run on identical features: **19.18 / 49.97**; CLIP+LoRA dropped (GPU training, not re-run, no verified plant-specific number) |
| Zhao 54.24 (PlantDoc 5-way 5-shot) | Regime imprecise | As-reported **42.69 / 54.24** (1/5-shot, cross-domain unified protocol) |

The classifier itself is unchanged (97.41% ± 0.71% 11-class mean); the corrections concern **comparison fairness and provenance**, not the proposed method's accuracy.

---

## 6. Scope and limitations (stated plainly)

- **Controlled re-runs are on PlantVillage.** The full PlantDoc dataset is not redistributed here, so PlantDoc literature values are quoted as-reported and our PlantDoc results use the documented protocol; we do not claim a controlled PlantDoc head-to-head.
- **In-regime ≠ head-to-head.** Our 5-way K-shot numbers and the literature values share a regime but differ in encoder, class subset, and split; we explicitly avoid direct superiority claims across them.
- **Numerical tolerance.** Transductive accuracy reproduces within 0.04 pp across library versions (lsqr floating-point sensitivity); inductive and all baselines reproduce exactly.
- **No backbone retraining.** The comparison isolates the gradient-free adaptation algorithm; it is not a comparison of encoders.

---

*All controlled figures are traceable to `controlled_baseline_results.json` (re-run by `run_controlled_baselines.py`) and to the revised manuscript Tables `tab:comparative` / `tab:literature` and Appendix `app:repro`. Literature values are verified against the cited publications (DOIs: 10.3390/app152312787, 10.3390/agriengineering8100407, 10.3390/s26123632).*