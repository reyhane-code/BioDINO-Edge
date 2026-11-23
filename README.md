# BioDINO-Edge: Uncertainty-Gated Cascaded Few-Shot Foliar Pathology on Mobile CPUs

[![Paper Status: EAAI (Elsevier)](https://img.shields.io/badge/Journal-Elsevier%20EAAI-blue.svg)](paper/BioDINO_Edge_EAAI.pdf)
[![Platform: Edge CPU](https://img.shields.io/badge/Platform-ARM64%20CPU%20%7C%20Android%2017-brightgreen.svg)]()
[![Model Size: INT8 ONNX](https://img.shields.io/badge/Package%20Size-22.8%20MB-orange.svg)]()
[![License: Academic Research](https://img.shields.io/badge/License-Research%20Only-lightgrey.svg)]()

This repository contains the complete research artifacts, reproduction code, edge deployment pipeline, and LaTeX manuscript for **BioDINO-Edge**: a dual-tier, cascaded few-shot diagnostic architecture for precision agricultural pathology deployed on commodity mobile CPUs without cloud access or GPU backpropagation.

---

## Key Highlights & Validated Benchmarks

- **Dual-Tier Dynamic Cascade:**
  - **Tier-1 Fast Path:** 8-bit integer (INT8) quantized DINOv2 Small visual encoder (384-D) scoring nearest-centroid cosine similarity in **59.78 ms** (~16.7 FPS burst rate).
  - **Shannon Uncertainty Gate:** Queries with high entropy and small top-2 margins are selectively routed to Tier-2; confident leaves (14.02% of the stream) exit immediately with **99.69% accuracy** and **0.25% false-negative rate (FNR)**.
  - **Tier-2 Deep Escalation:** BioCLIP ViT-B (512-D) companion encoder concatenated with DINOv2 (896-D) fused via regularized Fisher Linear Discriminant Analysis (Fisher-LDA) with taxonomic text priors (**164.52 ms**, 6.1 FPS).
  - **Cascaded Stream Average:** Full-stream mean latency of **149.84 ms** (~6.7 FPS), satisfying real-time scouting bounds with 97.36% accuracy.
- **Closed-Form Algebraic Adaptation:** Resolves small-sample covariance singularity ( \ll d$) via analytical **Ledoit--Wolf shrinkage** in **533.99 ms** on mobile CPU, eliminating backpropagation graphs, GPU memory spikes, and learning rate tuning.
- **Streaming Transductive Recalibration:** FIFO buffer tracks high-confidence field queries ($\ge 0.70$), updating prototypes and class boundaries on-the-fly (**97.41% ± 0.71%** accuracy on PlantVillage, +4.13 pp over nearest centroids).
- **Physical Handheld Deployment:** Validated on a physical Samsung Galaxy S24 FE (Exynos 2400e, Android 17 / API 37) with **497.45 MB peak PSS RAM** and standalone **22.8 MB APK**.

---

## Repository Directory Structure

``	ext
D:\Research\BioDINO-Edge\
├── paper/                                  # Final LaTeX publication assets
│   ├── BioDINO_Edge_EAAI.tex               # Standalone final LaTeX manuscript (Elsevier EAAI format)
│   ├── BioDINO_Edge_EAAI.pdf               # Compiled 34-page publication PDF
│   ├── main.tex / main.pdf                 # Symlinks / convenience aliases
│   ├── references.bib                      # Complete BibTeX bibliography
│   ├── elsarticle.cls / elsarticle-num.bst # Elsevier LaTeX class & citation style
│   ├── fig_architecture.pdf               # Fig 1: Two-tier architecture diagram
│   ├── fig_risk_coverage.pdf              # Fig 3: Risk-coverage curves & AURC
│   ├── fig_tau_sensitivity.pdf            # Fig 4: Threshold sweep & trade-off curve
│   └── fig_confusion.pdf                   # Fig 5: Row-normalized confusion matrix
│
├── pipeline/                               # Core Python inference & edge execution
│   ├── biodino_edge_pipeline.py            # Primary edge pipeline harness (evaluation & streaming)
│   ├── core_math.py                        # Core algebra: Ledoit-Wolf shrinkage, Shannon gating, Fisher-LDA
│   ├── data_engine.py                      # Data loaders, few-shot episode generator & multi-view augmentation
│   ├── taxonomic_prompts.py                # Botanical taxonomic prompts & text prototype extractor
│   ├── export_edge_models.py               # ONNX export and INT8 quantization script
│   ├── demo_predict.py                     # Single-image and batch diagnostic CLI runner
│   ├── benchmark_engine.py                 # Multi-episode evaluation engine
│   └── generate_confusion_matrix.py        # Reproduces Fig 5 from model predictions
│
├── experiments/                            # Exact experiment runners reproducing paper tables
│   ├── augmentation_sensitivity/           # Table 12: View count (v in {1,2,4,8}) & augmentation modes
│   ├── baseline_comparison/                # Table 1 & Table 2: Controlled baselines on identical features
│   ├── dinov2_ablation_gate_scenario/      # Table 9: Foundation encoder ablation & gating scenarios
│   ├── edge_device_benchmark/              # Table 11, 13, 14, 15: Physical Android latency & CPU breakdown
│   ├── paper_rerun/                        # Table 3: Sequential streaming vs LaplacianShot, canonical reruns
│   ├── quantization_accuracy/              # Table 10 & 16: INT8 vs FP32 precision & PlantDoc ladder
│   ├── routing_calibration_validation/     # Table 18, Fig 3 & 4: Gating threshold sweep & AURC curves
│   └── transductive_validation/            # Table 5 & 10: Paired t-test/Wilcoxon & transductive sensitivity
│
├── cache_artifacts/                        # Pre-extracted embeddings & quantized ONNX models
│   ├── pv_novel_embeddings.pt              # Cached PlantVillage 11-class embeddings (instant experiments)
│   ├── biodino_mobile_int8.onnx            # Tier-1 DINOv2 Small quantized model (~22 MB)
│   └── biodino_companion_int8.onnx         # Tier-2 BioCLIP ViT-B quantized model (~85 MB)
│
├── models/                                 # Dedicated model storage
│   ├── biodino_mobile_int8.onnx
│   └── biodino_companion_int8.onnx
│
├── results/                                # Raw JSON execution logs and benchmark outputs
│   ├── biodino_edge_results.json
│   ├── biodino_hierarchical_results.json
│   └── biodino_v3_results.json
│
├── demo_data/few_shot_demo_corn_rust/      # Real foliar photos for quick verification demo
├── plantdoc/                               # PlantDoc field test benchmark (252 test images, 27 classes)
├── plantvillage/                           # PlantVillage foliar dataset (11 classes, 9,385 images)
│
├── android_app/                            # Complete native Android Kotlin project
│   ├── app/src/main/java/...               # Clean Kotlin code: ONNX Runtime + EJML Cholesky
│   ├── BioDINO-Edge-v1.0.1-release.apk     # Pre-compiled standalone signed APK ready to install
│   ├── build.gradle.kts                    # Gradle build script
│   └── settings.gradle.kts
│
│
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
`

---

## Environment Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- PyTorch >= 2.0 with torchvision
- ONNX Runtime >= 1.18

### 2. Installation
`ash
# Clone and navigate to the repository
cd \BioDINO-Edge
# Create and activate a clean virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
`

---

## How to Run Diagnostic Inference & Pipeline

### 1. Quick Foliar Demo Prediction (CLI)
Classify sample leaves from the included demo set:
`ash
python pipeline/demo_predict.py --image demo_data/few_shot_demo_corn_rust/6_test_infected_corn_rust.jpg
`

### 2. Run Full Edge Evaluation Pipeline
Run the cascaded two-tier pipeline across 30 Monte Carlo episodes on PlantVillage using cached embeddings:
`ash
python pipeline/biodino_edge_pipeline.py --dataset plantvillage --episodes 30
`
To evaluate on the unconstrained in-the-wild PlantDoc benchmark:
`ash
python pipeline/biodino_edge_pipeline.py --dataset plantdoc --episodes 50
`

### 3. Re-export Quantized ONNX Edge Models
Export frozen PyTorch backbones to static INT8 ONNX binaries:
`ash
python pipeline/export_edge_models.py --format int8 --output_dir models/
`

---

## How to Reproduce Paper Experiments & Tables

All scripts read from cache_artifacts/pv_novel_embeddings.pt and run deterministically within seconds.

| Table / Figure in Paper | Research Objective | Reproduction Command |
|---|---|---|
| **Table 1 & Table 2** | Controlled baseline comparison on identical 896-D features | python experiments/baseline_comparison/run_controlled_baselines.py |
| **Table 3** | Sequential streaming vs LaplacianShot under varying batch sizes | python experiments/paper_rerun/run_batchsize_sensitivity.py |
| **Table 5** | Paired statistical significance tests (t-test, Wilcoxon, Cohen's $) | python experiments/transductive_validation/run_transductive_validation.py |
| **Table 8** | Step-by-step architectural component ablation | python experiments/paper_rerun/run_canonical_rerun.py |
| **Table 9** | Foundation encoder feature dimension ablation (384-D, 512-D, 896-D) | python experiments/dinov2_ablation_gate_scenario/run_feature_ablation.py |
| **Table 10** | Sensitivity to transductive support weights, thresholds & buffer capacity | python experiments/transductive_validation/run_transductive_validation.py |
| **Table 12** | Multi-view augmentation count & covariance conditioning $\kappa(\hat{\mathbf{\Sigma}})$ | python experiments/augmentation_sensitivity/run_augmentation_sensitivity.py |
| **Table 16** | Numerical parity between FP32 and INT8 quantization | python experiments/quantization_accuracy/run_quantization_accuracy.py |
| **Table 18** | Tier-separated latency, coverage & gating threshold sweep | python experiments/routing_calibration_validation/run_routing_validation.py |
| **Fig 3 & Fig 4** | Selective Risk--Coverage curves (AURC) & $\tau$ sweep plots | python experiments/routing_calibration_validation/run_routing_validation.py |
| **Fig 5** | Row-normalized confusion matrix across 9,330 PlantVillage leaves | python pipeline/generate_confusion_matrix.py |

---

## Inspecting and Compiling the Paper

The complete, final LaTeX manuscript is stored in paper/:
- **Manuscript File:** paper/BioDINO_Edge_EAAI.tex
- **Compiled PDF:** paper/BioDINO_Edge_EAAI.pdf

To recompile the PDF from source with MiKTeX or TeX Live:
`ash
cd paper
pdflatex -interaction=nonstopmode BioDINO_Edge_EAAI.tex
bibtex BioDINO_Edge_EAAI
pdflatex -interaction=nonstopmode BioDINO_Edge_EAAI.tex
pdflatex -interaction=nonstopmode BioDINO_Edge_EAAI.tex
`
*The document builds cleanly in 34 pages with zero warnings, zero overfull hboxes, and zero undefined citations.*

---

## Citation

If you use BioDINO-Edge in your research, please cite our paper:

`ibtex
@article{biodino_edge_2026,
  title={Cascaded Two-Tier Few-Shot Foliar Pathology with Uncertainty-Gated Foundation Models on Commodity Edge Processors},
  author={Alsaadi, Reyhane Javidi and Collaborators},
  journal={Engineering Applications of Artificial Intelligence},
  year={2026},
  publisher={Elsevier}
}
`
