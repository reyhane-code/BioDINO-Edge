# BioDINO-Edge

> **Gradient-Free Cascades of Frozen Vision Foundation Models for On-Device Few-Shot Plant Disease Diagnosis**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch%20%7C%20ONNX-orange.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20Format-red.svg)](paper/BioDINO-Edge.pdf)

BioDINO-Edge is a gradient-free, two-tier few-shot diagnostic framework for foliar crop disease identification on resource-constrained edge devices. By coupling self-supervised representations from **DINOv2** with organismal botanical priors from **BioCLIP**, the pipeline adapts to novel crop pathologies via analytical **Fisher Linear Discriminant Analysis (Fisher-LDA)** regularized by **Ledoit–Wolf shrinkage**—eliminating backpropagation, GPU requirements, and catastrophic forgetting during on-device deployment.

---

## Key Highlights

- **Gradient-Free Closed-Form Adaptation:** Projection weights are solved analytically in closed form ($<22$ ms on an edge CPU), completely circumventing gradient descent.
- **Adaptive Two-Tier Confidence Gating:** A joint Shannon entropy and prediction margin gate routes routine foliage to a fast Tier-1 model ($21.12$ ms, $>45$ FPS on a quad-core ARM Cortex-A76) and escalates ambiguous queries to a multimodal Tier-2 classifier.
- **Soft-Anchored Transductive Recalibration:** Confident field predictions are buffered as reference anchors to periodically update projection boundaries in the background, mitigating finite-sample prototype drift.
- **Lightweight On-Device Packaging:** Packaged into INT8 ONNX binaries with a base mobile footprint of $<23$ MB (total storage $108.03$ MB including the on-demand companion model).
- **Strong Diagnostic Fidelity:** 
  - **PlantVillage (11 Classes):** $97.37\% \pm 0.73\%$ accuracy with a $0.604\%$ false-negative disease escape rate.
  - **PlantDoc (In-the-Wild):** $83.61\%$ (5-way 5-shot) and $56.35\%$ (27-way all-class) without foundation-model fine-tuning.

---

## Repository Structure

```text
Bio-DINO-Edge/
├── cache_artifacts/                 # Precomputed support embeddings & intermediate models
├── paper/                           # Complete IEEE conference paper source & figures
│   ├── BioDINO-Edge.tex             # Main LaTeX source
│   ├── BioDINO-Edge.pdf             # Compiled camera-ready PDF
│   ├── IEEEtran.cls                 # IEEE conference template class
│   ├── references.bib               # BibTeX bibliography database
│   ├── fig_architecture.pdf        # Architecture schematic diagram
│   ├── fig_saliency.pdf            # Spatial explainability & attention figure
│   └── fig_calibration.pdf         # Uncertainty calibration curves
├── pipeline/                        # Core Python algorithms and execution modules
│   ├── biodino_edge_pipeline.py     # Main benchmark evaluation pipeline
│   ├── biodino_hierarchical_pipeline.py # Hierarchical benchmark pipeline
│   ├── core_math.py                 # Ledoit-Wolf shrinkage, Fisher-LDA & entropy gating
│   ├── data_engine.py               # Dataset ingestion & few-shot episode generators
│   ├── demo_predict.py              # Single-image live inference demo
│   ├── export_edge_models.py        # ONNX export and INT8 dynamic quantization
│   ├── generate_architecture_diagram.py # Generates architecture schematics
│   ├── generate_saliency_and_calibration.py # Generates paper figures & metrics
│   └── taxonomic_prompts.py         # Botanical prompt templates and binomial descriptors
├── biodino_mobile_int8.onnx         # Quantized Tier-1 fast-path edge model (~23 MB)
├── biodino_companion_int8.onnx      # Quantized Tier-2 deep escalation model (~89 MB)
├── biodino_edge_results.json        # Benchmark evaluation metrics & episode outputs
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Environment Setup

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/reyhane-code/Bio-DINO-Edge.git
cd Bio-DINO-Edge

# Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Dataset Preparation

The benchmark evaluates on two foliar datasets:
1. **PlantVillage:** Structured laboratory images under uniform conditions. Place dataset images under:
   ```text
   plantvillage/plantvillage dataset/color/<Crop___Pathology>/
   ```
2. **PlantDoc (In-the-Wild):** Challenging field images with canopy clutter. Samples are included under `pipeline/plantdoc_samples/` or can be linked to the full dataset directory.

---

## Running the Pipeline

### Quick Single-Image Inference Demo
Test the two-tier uncertainty-gated pipeline on a single query leaf:

```bash
# Run prediction on an input leaf image
python pipeline/demo_predict.py --image path/to/leaf.jpg --device cpu

# Use custom confidence gating threshold (tau)
python pipeline/demo_predict.py --image path/to/leaf.jpg --tau 0.35
```

### Reproducing Benchmark Experiments
To evaluate the 5-shot adaptation pipeline across PlantVillage and PlantDoc:

```bash
# Evaluate on PlantVillage (11-class Apple, Grape, Potato benchmark)
python pipeline/biodino_edge_pipeline.py --dataset plantvillage --k_shot 5

# Evaluate on PlantDoc (In-the-Wild benchmark)
python pipeline/biodino_edge_pipeline.py --dataset plantdoc --k_shot 5

# Run combined benchmark with latency profiling
python pipeline/biodino_edge_pipeline.py --dataset both --k_shot 5 --profile_latency
```

### Generating Paper Saliency & Calibration Figures
To reproduce the attention rollout overlays and Expected Calibration Error (ECE) curves:

```bash
python pipeline/generate_saliency_and_calibration.py
```

Outputs are automatically saved as publication-ready PDFs in the `paper/` directory.

---

## Edge Model Build & Quantization

To compile and export the dual-tier models to INT8 ONNX binaries for on-device (Android / ARM CPU) deployment:

```bash
python pipeline/export_edge_models.py
```

This generates:
- **Tier 1 (`biodino_mobile_int8.onnx` ~22 MB):** Standalone DINOv2 Small backbone with fused prototype projection for real-time mobile triage ($>45$ FPS).
- **Tier 2 (`biodino_companion_int8.onnx` ~89 MB):** Companion multimodal BioCLIP ViT-B model with regularized Fisher-LDA alignment for ambiguous queries.

---

## Compiling the Paper (LaTeX)

The camera-ready manuscript is written in standard IEEEtran format. To compile the paper:

### Using `pdflatex` & `bibtex`:
```bash
cd paper
pdflatex -interaction=nonstopmode BioDINO-Edge.tex
bibtex BioDINO-Edge
pdflatex -interaction=nonstopmode BioDINO-Edge.tex
pdflatex -interaction=nonstopmode BioDINO-Edge.tex
```

### Using `latexmk`:
```bash
cd paper
latexmk -pdf BioDINO-Edge.tex
```

The resulting document is generated as `paper/BioDINO-Edge.pdf`.

---

## Citation

If you use BioDINO-Edge or find this codebase helpful in your research, please cite:

```bibtex
@inproceedings{alsaadi2026biodino,
  title     = {BioDINO-Edge: Gradient-Free Cascades of Frozen Vision Foundation Models for On-Device Few-Shot Plant Disease Diagnosis},
  author    = {Javidi Alsaadi, Reyhane and Samadzadeh, Mohammad Reza},
  booktitle = {IEEE Conference on Precision Agriculture and Edge AI},
  year      = {2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

