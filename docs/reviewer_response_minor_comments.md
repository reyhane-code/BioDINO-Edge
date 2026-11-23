# Response to Reviewer — Minor Comments #1–#5 (Mathematical Clarifications, Units, and Terminology)

**Manuscript:** BioDINO-Edge: Gradient-Free Cascades of Frozen Vision Foundation Models for On-Device Few-Shot Plant Disease Diagnosis
**Target journal:** Engineering Applications of Artificial Intelligence (Elsevier)
**Source of truth (kappa):** `experiments/augmentation_sensitivity/results/augmentation_sensitivity.json` (condition `views8`, fields `cov_cond_mean` / `cov_cond_std`, 30 episodes, seed 42+17e)
**Revised manuscript:** `paper/main_eaai_elsarticle.tex` (20 pages, clean compile). Equation numbering is preserved: the gate remains Eq. (13); the new whitening-spread definition occupies Eq. (10), replacing the withdrawn variance-multiplier align that previously held that number, so all downstream equation numbers are unchanged.

---

## Reviewer comments (restated)

1. **Eq. (13) explanation:** the text after Eq. (13) suggests \(1-\mathcal{H}_{\text{norm}}\) can be negative, although \(\mathcal{H}_{\text{norm}}\in[0,1]\).
2. **Units:** the text reports 9,856 MACs while the complexity table reports "9.8K FLOPs"; the MAC/FLOP convention is inconsistent.
3. **Figure 2 badges:** the per-sample "Accuracy" labels need explanation, or relabeling if they denote confidence.
4. **Variance multipliers:** the claimed 31× background-variance reduction and 2.44× lesion-variance amplification lack precise metric definitions and empirical evidence.
5. **Terminology:** cross-domain few-shot adaptation must be distinguished from evaluation with labeled target-domain support, consistently throughout.

---

## Summary of response

We thank the reviewer for these precise corrections. All five are addressed by text revisions; no new experiment was required because the replacement quantity for comment 4 was already measured in the augmentation-sensitivity study. Two earlier over-claims (the direction-specific variance multipliers and the per-sample figure badges) are **withdrawn**, not re-derived.

---

## 1. Boundedness of \(1-\mathcal{H}_{\text{norm}}\) — comment 1

The reviewer is correct: by construction \(\mathcal{H}(\mathbf{p}) \le \log C\), hence \(\mathcal{H}_{\text{norm}} \in [0,1]\) and \(1-\mathcal{H}_{\text{norm}} \ge 0\) in exact arithmetic. The \(\epsilon\)-regularized logarithm can only *underestimate* the entropy, so it cannot push the normalized entropy above 1 either.

**Revision:** immediately after the entropy definition (Eq. (14)) we added: *"Because Shannon entropy is bounded by \(\log C\), \(\mathcal{H}_{\text{norm}} \in [0,1]\) and \(1-\mathcal{H}_{\text{norm}}\) is non-negative in exact arithmetic (the \(\epsilon\)-regularized logarithm can only underestimate the entropy); the \((\cdot)_+\) operator in Eq. (13) is thus a floating-point safeguard against round-off, not a theoretical correction."* The gate equation itself is unchanged.

## 2. MAC vs FLOP convention — comment 2

**Revision:** at first use (Section 3, Tier-2) the cost is now stated as: *"9,856 multiply-accumulate operations (one MAC = one multiply plus one add = 2 FLOPs, hence \(\approx 19.7\) kFLOPs)"*, i.e. \(C \cdot d = 11 \times 896 = 9{,}856\) MACs \(= 19{,}712\) FLOPs. The legacy complexity table that carried the inconsistent "9.8K FLOPs" label no longer exists in the manuscript, so a single convention now governs all cost statements. The Persian documentation set was aligned to the same convention.

## 3. Per-sample "Accuracy" badges in Figure 2 — comment 3

The badges were neither per-sample accuracy nor calibrated confidence; they were unbacked annotations. Rather than relabel them, the entire saliency figure was **removed** from the manuscript (together with its post-hoc attribution equation and subsection), because its qualitative content added no quantitative evidence and the frozen-encoder attention cannot be modified by the downstream LDA. The Limitations section now states explicitly that the framework makes no lesion-localization or attention-modification claim. All remaining figures report episode-level or binned quantities (e.g., the reliability diagram plots binned empirical accuracy against mean confidence), so no per-sample "accuracy" label survives anywhere in the paper.

## 4. Withdrawal of 31× / 2.44× and replacement by a measured, defined quantity — comment 4

We agree these multipliers were not defined metrics and had no empirical backing: the eigen-directions of the within-class covariance carry no semantic "background"/"lesion" annotation, so direction-specific suppression/amplification factors cannot be measured as claimed.

**Revision:** all occurrences (abstract, introduction, Section 3 motivation, conclusion, contributions) are replaced by the **whitening spread**, now defined as Eq. (10):
\[\kappa(\hat{\mathbf{\Sigma}}) = \lambda_{\max}(\hat{\mathbf{\Sigma}}) / \lambda_{\min}(\hat{\mathbf{\Sigma}}),\]
the ratio between the largest and smallest inverse-variance whitening weights. It is **measured** as \(1435.6 \pm 310.2\) over the 30 canonical episodes on the 8-view augmented support (source: `augmentation_sensitivity.json`, condition `views8`), and is cross-referenced to the augmentation-sensitivity table, whose \(v=8\) row reports the same quantity as \(1.4\times10^3\). The text states explicitly that eigen-directions carry no semantic labels and that this defined spectral spread is reported *instead of* direction-specific multipliers.

## 5. Adaptation-setting terminology — comment 5

**Revision:** a new setup bullet, *"Adaptation Settings Defined"*, distinguishes three settings:
(i) **few-shot adaptation with labeled target-domain support** — support and queries from the same target domain (all PlantVillage regimes and our PlantDoc runs; PlantDoc support is drawn from the PlantDoc train split);
(ii) **cross-domain few-shot adaptation** — source-domain support, target-domain queries (the PlantVillage→PlantDoc protocol of Zhao et al. in the literature table), **which we do not run**;
(iii) **transductive recalibration** — unlabeled query anchors with pseudo-labels.
The only cross-domain element of our PlantDoc protocol is the transfer of hyperparameters selected on PlantVillage. All manuscript mentions of our PlantDoc evaluation were renamed from "cross-domain" to **"field-domain"** (abstract, results, discussion, appendix, provenance table, contributions); the Zhao et al. row retains its correct cross-domain label. The Persian documentation and the earlier response letters were aligned to the same terminology.

---

## Provenance and reproducibility

| Claim in revision | Backing artifact |
|---|---|
| \(\kappa = 1435.6 \pm 310.2\) (30 episodes, 8-view support) | `experiments/augmentation_sensitivity/results/augmentation_sensitivity.json`, `views8.cov_cond_mean/std`; Table (aug sensitivity) \(v=8\) row |
| 9,856 MACs = 19,712 FLOPs | Arithmetic identity \(C \cdot d = 11 \times 896\); convention 1 MAC = 2 FLOPs stated in text |
| \(\mathcal{H}_{\text{norm}} \in [0,1]\) | Definition \(\mathcal{H}_{\text{norm}} = \mathcal{H}/\log C\) and \(\mathcal{H} \le \log C\) |
| Figure/badge withdrawal, terminology renames | Editorial; no numeric claim involved |

No number in this response or in the revised manuscript lacks a backing artifact. The Persian guides (`algorithm_guide.md`, `mathematics_foundation_guide.md`, `code_math_algorithm_deep_dive.md`) and `reviewer_response_comment8.md` were updated to the same formulations (κ whitening spread, MAC convention, field-domain terminology, post-hoc attribution framing).