# Cover Letter

**[Date]**

To the Editor-in-Chief
**Engineering Applications of Artificial Intelligence**
Elsevier

Dear Editor,

We are pleased to submit our original research manuscript entitled **“BioDINO-Edge: Gradient-Free Cascades of Frozen Vision Foundation Models for On-Device Few-Shot Plant Disease Diagnosis”** for consideration as an **Original Research article** in *Engineering Applications of Artificial Intelligence*.

**Contribution to artificial intelligence.** The manuscript proposes a gradient-free, closed-form adaptation framework for frozen multi-scale vision foundation models. It resolves the small-sample covariance singularity of Fisher linear discriminant analysis under severe few-shot constraints ($K=5$, $N \ll d$) through analytical Ledoit--Wolf shrinkage, and combines spatial DINOv2 patch representations with taxonomic BioCLIP embeddings in a single 896-dimensional descriptor. No backbone is fine-tuned and no on-device backpropagation is performed.

**Contribution to engineering.** The framework is deployed as a standalone, fully offline Android application (approximately 22.8 MB) with a two-tier Shannon-entropy/margin routing gate, INT8 ONNX inference on a mobile CPU, and closed-form transductive recalibration. It is validated on a physical Android device and on the in-the-wild PlantDoc benchmark, and it reports latency, memory, calibration, selective risk, and false-negative disease-escape rates relevant to practical precision-agriculture deployment.

We believe the work fits the scope of the journal because it addresses a real-world engineering application of AI under explicit hardware, latency, and data-scarcity constraints, and because all reported results are reproducible from released code and public datasets.

We confirm that:

- the manuscript is original, has not been published previously, and is not under consideration elsewhere;
- all authors have approved the manuscript and its submission;
- the authors declare no competing financial or personal interests;
- the study used only public datasets (PlantVillage and PlantDoc) in accordance with their terms;
- the use of AI-assisted tools in manuscript preparation is declared in the manuscript, before the reference list.

The complete anonymized manuscript, a separate title page, highlights, figures, and declarations are provided with this submission. Code and models will be released in a public repository upon acceptance.

Thank you for considering our work. We look forward to the reviewers’ comments.

Sincerely,

**Reyhane Javidi Alsaadi** (corresponding author)  
Department of Computer Engineering, Dr. Shariaty Technical University, Tehran, Iran  
Email: reyhane.code@gmail.com  
ORCID: https://orcid.org/0009-0000-0296-7467

On behalf of all authors: Reyhane Javidi Alsaadi and Mohammad Reza Samadzadeh.
