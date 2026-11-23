# CORRECTION — LaplacianShot must be the FAITHFUL implementation

## What was wrong

`run_fair_baselines.py` implemented LaplacianShot as **soft label propagation**
(`F <- a*S@F + (1-a)*Y`, Zhou-style). That is not LaplacianShot.

The published method (Ziko, Dolz, Granger, Ben Ayed, **ICML 2020**,
arXiv:2006.15486) minimises a quadratic **binary assignment** with a pairwise
Laplacian term, solved by majorize-minimise:

    min_{Y in {0,1}^{n x C}, Y1=1}  <Y,D> - (lambda/2) <Y, W Y>
    y_i <- onehot(argmin_c [ D[i,c] - lambda * (W Y)[i,c] ])

The correct solver is **stronger**, so the earlier paper text understated the
baseline. All LaplacianShot numbers have been regenerated.

## Corrected numbers (canonical episodes)

Script: `run_laplacianshot_canonical.py`
Artifact: `results/laplacianshot_canonical_2026-11-18.json`
lambda = 2.0, selected on a DISJOINT seed stream (42+1000+17e).

| Regime | LaplacianShot | BioDINO-Edge | delta | p | W/L |
|---|---|---|---|---|---|
| 11-class, 5-shot, 30 ep | **98.49 ± 0.59** (F1 98.70) | 97.41 ± 0.73 (F1 97.09) | **-1.08** | 1.4e-7 | 3/27 |
| 5-way 5-shot, 50 ep | **99.50 ± 0.82** (F1 99.46) | 98.78 ± 1.69 (F1 98.51) | **-0.72** | 2.7e-3 | 11/39 |

Previously reported: -0.53 pp (11-class) and -0.49 pp (5-way). Both were wrong
because of the stand-in solver. The paper now says -1.08 pp.

## Deployment strategies (faithful solver)

Script: `run_laplacianshot_streaming.py`
Artifact: `results/laplacianshot_streaming_2026-11-18.json`
8 episodes x 600 queries, paired, same query ordering.

| Strategy | LaplacianShot | BioDINO-Edge | delta | p |
|---|---|---|---|---|
| one leaf at a time | 93.12 ± 1.47 | **96.12 ± 1.39** | +3.00 | 2.9e-4 |
| growing buffer 25 | 86.50 ± 1.60 | **96.12** | +9.62 | 1.1e-7 |
| growing buffer 50 | 88.17 ± 1.64 | **96.12** | +7.96 | 7.8e-7 |
| growing buffer 100 | 90.17 ± 1.40 | **96.12** | +5.96 | 3.7e-6 |
| growing buffer 200 | 93.48 ± 1.18 | **96.12** | +2.65 | 1.8e-3 |
| whole query set | **96.38 ± 1.31** | 96.12 | -0.25 | 0.68 (ns) |

Reading: the fixed-pool advantage is real (-1.08 pp). Under any per-leaf or
buffered deployment it inverts and every row is significant. But note the
growing buffer NARROWS the gap (9.62 -> 2.65 as the buffer grows), so the honest
claim is "the gap depends on batching", not "propagation cannot be deployed".

Note also: the whole-pool row is a tie here (96.38 vs 96.12, p=0.68) even though
the full 9330-query run gives LaplacianShot -1.08 pp, because a 600-query subset
is a much smaller graph.

## Fisher + propagation, faithful

Script: `run_fisher_laplacian_faithful.py`
Artifact: `results/fisher_laplacian_faithful_2026-11-18.json`

| Feature space | Acc |
|---|---|
| raw 896-D | **98.57 ± 0.64** |
| Fisher-projected | 92.84 ± 3.24 |
| raw + projected concat | 98.53 ± 0.68 |

Projecting through the discriminant destroys the neighbourhood structure the
pairwise term needs. The `both` variant ties raw, so it adds nothing.

## Why LaplacianShot is the right comparator

1. It is a transductive **inference rule on top of any frozen embedding** — the
   same object we are comparing against. The official README says explicitly
   "can be utilized during inference on top of any learned feature embeddings".
2. It is training-free: no backbone gradient, no fine-tuning. Comparing it to
   our gradient-free adapter is apples to apples.
3. It is the strongest published training-free transductive baseline; ICML 2020,
   still the reference point in the FSL literature.
4. Both methods consume the identical support set and the identical 896-D
   features, so the only difference is the adaptation rule.

Not including it would be a *worse* methodological error than losing to it.

## PlantDoc: the dataset the paper actually deploys on

Script: `run_laplacianshot_plantdoc.py`
Artifact: `results/laplacianshot_plantdoc_2026-11-18.json`
Deployed INT8 binaries, same 135 support leaves (8-view -> 1080), same 896-D
features, PlantDoc constants (w_sup=1, 2 anchors/class, no threshold).

| Regime | LaplacianShot | BioDINO-Edge | delta |
|---|---|---|---|
| 27-way, 252 field queries | 32.14 | **50.79** | +18.65 |
| 5-way 5-shot, 50 episodes | 62.72 | **85.01** | +22.29 (p=7.5e-16, 50/0) |

The ranking INVERTS on the field benchmark. Reason: LaplacianShot matches to
visual prototypes by cosine only and has no botanical text branch; on PlantDoc
the text prior carries a large share of the signal (see the ladder: text-only
34.92, BioCLIP 51.98, blend 53.17).

Note: this run's "ours" 5-way entry is 85.01 vs the pipeline reference 84.22.
The 0.8 pp difference comes from restricting episodes to classes with >= 2 test
leaves. Both are reported; the table note says so.

## Historical note for reviewers

LaplacianShot: ICML 2020 (PMLR v119), arXiv June 2020. Not a 2026 method; it is
five years old and still standard. Losing to a five-year-old method by ~1 pp in
its own protocol is a fair result to report, and defensible because the protocol
requires a batch that a phone does not have.
