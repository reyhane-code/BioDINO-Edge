# Response to Reviewer — Comment #7 (Empirical Evidence on Real Edge Hardware)

**Manuscript:** BioDINO-Edge: Gradient-Free Cascades of Frozen Vision Foundation Models for On-Device Few-Shot Plant Disease Diagnosis
**Target journal:** Engineering Applications of Artificial Intelligence (Elsevier)
**Source of truth:** `experiments/edge_device_benchmark/results/benchmark_report.json`
**Run status:** `complete` · timestamp `2026-11-13T21:13:00Z` · schema `biodino_edge_device_benchmark_v1`
**Revised manuscript location:** Section *Edge Hardware Latency and Memory Footprint* (`sec:edge_hardware`); Tables `tab:device_spec`, `tab:device_perf`, `tab:adaptation_breakdown`.

---

## Reviewer comment (restated)

> The manuscript does not provide sufficient empirical evidence of real performance on edge devices. Specifically:
> **(a)** Report measurements on physical edge hardware — end-to-end inference time, peak RAM, model-load time, and energy consumption if possible.
> **(b)** Decompose the claimed "<22 ms projection-update" into its constituent stages: feature extraction, covariance estimation, shrinkage-coefficient computation, and matrix decomposition.
> **(c)** Provide the per-stage computational cost.

---

## Summary of response

We thank the reviewer for this decisive comment. We agree that the earlier latency/memory figures were estimates rather than physical-device measurements, and that the "<22 ms" projection-update claim was not a measured full-recalibration cost. To address this, we:

1. Built a dedicated on-device benchmark harness (Android `benchmark` build type, `isDebuggable=false`) and ran it on a **production Samsung Galaxy S24 FE (SM-S721B)**, CPU-only INT8, with no GPU/NNAPI delegate.
2. Report **measured** end-to-end latency, peak process RAM (PSS), one-time model-load time, and a **full per-stage adaptation-cost breakdown** with computational complexities (Sections 1–3 below).
3. **Explicitly withdraw three earlier over-claims** that the measurements contradict ("<22 ms", ">45 FPS real-time", and "22.05 MB resident RAM / 110 MB RAM budget"), and rescope the contribution to *interactive handheld scouting* rather than continuous real-time video (Section 5).

Every number below comes from a single clean run; all additive identities are internally consistent (Section 4). Energy could not be attributed to the app from `batterystats` and is honestly reported as **not measured**.

---

## 1. Physical edge-device testbed — answers (a)

*Manuscript Table `tab:device_spec`.*

| Property | Value |
|---|---|
| Device | Samsung SM-S721B (Galaxy S24 FE), codename `r12s` |
| SoC | Samsung s5e9945 (Exynos 2400e) |
| OS | Android 17 (API 37) |
| ABI / CPU cores | arm64-v8a / 10 |
| Total RAM | 7224.3 MB |
| Inference runtime | ONNX Runtime 1.18.0 |
| Threads (intra / inter) | 4 / 2 |
| Precision / delegate | INT8 / CPU (NNAPI **off**, GPU **off**) |
| Linear algebra | pure-Java EJML 0.43.1 |

**Measurement protocol.** Latency timed over **100 iterations after 10 warm-up passes** using `SystemClock.elapsedRealtimeNanos`; memory is the process **PSS** (`Debug.getPss()`); a **600 s sustained soak** runs *last* to capture thermal throttling; the adaptation algebra is timed after a **60 s cool-down** at stable thermal state. Android thermal status (0 none … 4 critical) is logged at every phase. Phase ordering is latency (cool) → adaptation (with cool-down) → sustained soak (heat) so that the reported algebra/latency figures are not thermally inflated.

---

## 2. Measured end-to-end performance — answers (a)

*Manuscript Table `tab:device_perf` (mean ± std over N=100; median and p95 in ms).*

| Stage / metric | Mean ± std | Median | p95 |
|---|---|---|---|
| Tier-1 model load (one-time) | 323.92 ms | — | — |
| Tier-2 model load (one-time) | 447.71 ms | — | — |
| **Peak RAM, PSS** | **497.45 MB** | — | — |
| Peak native heap | 385.02 MB | — | — |
| Idle PSS (before load) | 22.04 MB | — | — |
| Preprocess | 2.59 ± 0.67 | 2.36 | 4.36 |
| Tier-1 DINOv2 encoder | 57.18 ± 9.13 | 51.50 | 68.84 |
| Shannon gate / route | <0.01 | 0.00 | 0.00 |
| Tier-2 incremental (BioCLIP + Fisher) | 104.74 ± 17.39 | 94.20 | 124.97 |
| Decision | <0.01 | 0.00 | 0.00 |
| Cascade end-to-end (device, 18% fast-exit) | 145.37 ± 47.61 | 147.77 | 196.36 |
| Tier-2-on-all end-to-end | 164.52 ± 26.77 | 148.73 | 196.36 |
| Sustained throughput (600 s soak) | **4.26 FPS** | — | — |
| Energy (mJ / inference) | **not measured** | — | — |

**Headline rates.**
- **Tier-1 fast path: 59.78 ms ≈ 16.7 FPS** (cool-start burst).
- **Fully escalated Tier-2 query: 164.52 ms ≈ 6.1 FPS**; the BioCLIP encoder adds **104.74 ms** to the cached Tier-1 pass.
- **Full-stream routing blend: 149.84 ms ≈ 6.7 FPS** at the canonical 14.0% fast-exit rate (manuscript Table `tab:tier_routing`). The device cascade above (145.37 ms) is measured on the 11-image on-device set, whose 18% fast-exit fraction differs from the full-stream 14%; the two are kept distinct.
- **Sustained: 4.26 FPS** as the SoC reaches thermal status 3 — so 16.7 FPS is a *burst*, not a sustained rate.

**Energy.** Per-inference energy could not be attributed to the app from Android `batterystats` (whole-device counter, no per-UID isolation on this build). Rather than report an unverifiable number, we mark energy as **not measured** and leave it to future work with an external power monitor.

---

## 3. Per-stage adaptation cost breakdown — answers (b) and (c)

*Manuscript Table `tab:adaptation_breakdown` (N=440 views, d=896, C=11; algebra over 5 repeats after a 60 s cool-down, thermal status 2/2/2).*

The reviewer is correct that "<22 ms" cannot represent the full recalibration. We decompose the closed-form Fisher recalibration into the exact stages requested:

| # | Stage | Mean ± std (ms) | Complexity |
|---|---|---|---|
| 1 | **Feature extraction** (440 views, DINOv2 + BioCLIP) | 88055.6 (200.13 / view) | O(N·(C_dino + C_bio)) |
| 2 | **Covariance estimation** (centered scatter S = XᵀX/N) | 146.19 ± 2.15 | O(N·d²) |
| 3 | **Shrinkage-coefficient computation** (Ledoit–Wolf α) | 4.49 ± 0.85 | O(N·d) |
| 4 | **Matrix decomposition** (Cholesky Σ = LLᵀ) | 332.00 ± 2.52 | O(d³/3) |
| 5 | Projection solve (triangular solves for W, b) | 51.31 ± 3.68 | O(C·d²) |
| | **Algebraic recalibration total (stages 2–5)** | **533.99 ± 6.29 (≈0.53 s)** | — |
| | Full adaptation (extraction + algebra) | 88589.6 (≈88.6 s) | — |

**Mapping to the reviewer's four named stages:** feature extraction = stage 1; covariance estimation = stage 2; shrinkage-coefficient computation = stage 3; matrix decomposition = stage 4 (Cholesky), with the triangular projection solve (stage 5) reported for completeness.

**Key clarifications.**
- **Feature extraction (stage 1) is the dominant cost (≈88 s) and is reported separately.** It consists of 440 augmented support views pushed through both frozen encoders; it is a *background, one-time* support-set build, not a per-query cost. The **closed-form recalibration itself (stages 2–5) is sub-second: ≈0.53 s, gradient-free.**
- **Ledoit–Wolf is O(N·d), not O(N·d²).** We use the closed-form Frobenius identity
  Σᵢ‖xᵢxᵢᵀ − S‖_F² = Σᵢ‖xᵢ‖⁴ − N·‖S‖_F²,
  which avoids materializing N rank-1 d×d matrices. The measured 4.49 ms confirms the linear-in-d cost.
- **Cholesky (≈332 ms) dominates the algebra**, consistent with O(d³/3) ≈ 2.4×10⁸ ops at d=896.
- Measured shrinkage intensity **α = 0.0182**; algebra timed at **stable thermal status 2** (after cool-down), so the 0.53 s figure is a genuine cool-state measurement, not a throttled one.

---

## 4. Internal consistency and reproducibility

All additive identities hold exactly on the reported means:

- **Tier-1 total** = preprocess + encoder + gate + decision = 2.5924 + 57.1828 + 0.0009 + 0.0008 = **59.777 ms** ✓
- **Tier-2-on-all** = Tier-1 + incremental = 59.777 + 104.7388 = **164.5158 ms** ✓
- **Algebra total** = covariance + LW + Cholesky + solve = 146.192 + 4.4947 + 331.9965 + 51.3061 = **533.9893 ms** ✓
- **Full adaptation** = extraction + algebra = 88055.6153 + 533.9893 = **88589.6046 ms** ✓
- **Cascade blend** ≈ 0.18·59.78 + 0.82·164.52 = 145.66 ≈ 145.37 ms ✓ (rounding of the 18% fast-exit fraction)

Dispersion is tight where it matters: algebra total CV ≈ 1.2% (Cholesky ≈ 0.76%) over 5 repeats. Thermal status was stable at 2 during both latency and algebra; only the final sustained soak climbs to 3, as expected.

**Reproduction.** Harness: `android_app/app/src/benchmark/.../AdaptationBenchmark.kt` (and the latency/memory harness), `benchmark` build type, ONNX Runtime 1.18.0 CPU INT8 (4 intra / 2 inter), EJML 0.43.1. Driver: `run_device_benchmark.ps1`. Output: `experiments/edge_device_benchmark/results/benchmark_report.json`; manuscript tables generated by `make_tables.py` → `edge_device_tables.tex`.

---

## 5. Explicit corrections to earlier claims (honesty statement)

The measurements contradict three earlier headline claims. We have corrected them throughout the revised manuscript (abstract, introduction, complexity analysis, algorithm box, results, discussion, conclusion) rather than defend them:

| Earlier claim | Measured reality | Disposition |
|---|---|---|
| "<22 ms projection-update" | Projection solve alone = 51.31 ms; **full closed-form recalibration = 533.99 ms ≈ 0.53 s** (extraction ≈ 88 s reported separately) | **Withdrawn / rescoped** to sub-second gradient-free algebra |
| "21.12 ms Tier-1, >45 FPS real-time" | **Tier-1 = 59.78 ms ≈ 16.7 FPS** cool burst; **4.26 FPS sustained** (thermal 3) | **Withdrawn**; rescoped to *interactive handheld scouting*, not continuous video |
| "22.05 MB resident RAM" / "110 MB RAM budget" | **Peak PSS = 497.45 MB** (385.02 MB native heap); 107.31 MB is the **on-disk/OTA storage** footprint, not a RAM bound | **Corrected**; storage vs. runtime RAM distinguished |
| BioCLIP "42.90 ms" | **Tier-2 incremental = 104.74 ms** | **Corrected** |

The gate's primary benefit is therefore **safety, not throughput**: the fast-exited subset is resolved at 99.7% accepted accuracy and 0.25% false-negative rate, while ambiguous queries receive the full multimodal analysis. The standalone APK remains **≈22.8 MB** and operates entirely offline; accuracy is unchanged (97.41% ± 0.71% 30-episode mean), since the corrections concern latency/memory reporting, not the classifier.

---

## 6. Scope and limitations (stated plainly)

- **Single device / single SoC.** Results are from one production Exynos 2400e handset; we do not claim cross-device generality. A multi-device sweep (e.g., Raspberry Pi 5, mid-range Snapdragon) is future work.
- **Energy not measured** (not attributable from `batterystats`); no unverifiable power number is reported.
- **Sustained thermal throttling is real** (4.26 FPS at thermal 3 vs. 16.7 FPS cool burst); the system targets interactive per-frame scouting.
- **Full adaptation is extraction-dominated** (≈88 s for 440 support views, background/one-time); the *recalibration algebra* is the sub-second (≈0.53 s) gradient-free cost.
- **Device vs. full-stream fast-exit** differ (18% on the 11-image device set vs. 14.0% canonical); both are reported and kept distinct.

---

*All figures in this response are traceable to `benchmark_report.json` (status `complete`, 2026-11-13T21:13:00Z) and to the revised manuscript Section `sec:edge_hardware` and its three tables.*