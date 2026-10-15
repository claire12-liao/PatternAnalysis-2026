# Confidence-aware Facebook Page Classification with a GCN

COMP3710, Semester 2 2026. Selected project: multi-layer GNN node classification, Normal difficulty.

## Current status - 15 October 2026

- The student selected the Normal GNN project.
- The course repository has been cloned from `topic-recognition`, starting at commit `552b902`. The course remote is named `upstream`.
- Dataset specifications and the GCN reference have been checked against primary sources.
- The course data mirror was located in the PDF's embedded link in Appendix A.6: https://github.com/gayanku/facebook_page-page_dataset . The original PyG download domain is unavailable from the current environment.
- The pinned course archive has been downloaded and audited. It has 22,470 nodes, 128 finite feature values per node, four classes and 342,004 raw edge entries. See `reports/data_audit.json` for the checksum, provenance and complete audit.
- Split seed 42 gives 13,481 training, 4,493 validation and 4,496 test nodes. The splits cover every node exactly once.
- `dataset.py` implements numeric loading, validation, stratified splitting, training-only feature scaling and conversion to a PyG Data object. Four CPU software tests passed. The PyG conversion itself has not yet been executed because local Torch/PyG dependencies are not installed.
- Models have not been implemented or trained; there are no classification-performance results yet.
- The supplied course announcement offers a quota-limited Virtual GPU Lab at https://gpu-lab.cloud.eait.uq.edu.au/ . Rangpur is also an option. Current hardware, package versions and available quota still need checking. This data audit used no GPU time.

## Problem and engineering question

Build a prototype for a reviewer who classifies Facebook pages. Compare a feature-only MLP with a graph convolutional network (GCN) that also uses page relationships. Investigate whether the GCN improves classification and confidence reliability sufficiently to justify its computational cost. Evaluate an operating threshold that automatically accepts confident predictions and routes uncertain cases to a reviewer.

## Dataset and evaluation design

The downloaded course archive matches the documented graph dimensions. The supplied features are a 128-dimensional SVD compression of 4,714 description-token indicators. This provided representation is used in a transductive setting; it is not a feature transform fitted solely on training nodes.

The mirror uses `X`, `y`, `edges_x` and `edges_y`, while the upstream FacebookPagePage loader expects different names. `dataset.py` therefore constructs native PyG Data explicitly. It does not load the optional pickled `page_name` objects. The audit found 179 duplicate edge entries and 358 raw self-loop entries; graph construction deduplicates and removes existing loops so the GCN can later add exactly one self-loop per node.

1. Use the course-linked mirror, pinned at commit `ec01d02c39da4385052023ed28e36f0794f61dc8`, with SHA-256 `7a1b8a0687515e9859d1ca649dc276825f5db44568894dfbb6b55ebc1cf6a9c1`. Do not substitute raw feature-index lists for dense vectors.
2. Audit feature shapes, finite values, label values, edge endpoints, self-loops, duplicate edges, and class balance. Record whether the edge count includes both directions.
3. Create and save one stratified 60/20/20 train/validation/test split with split seed 42. Use the same split for every model and initialization seed. Fit any feature standardization only on training nodes.
4. Use a clearly documented transductive setting: all node features and graph edges are available; only training labels enter optimization. Validation and test labels are never model inputs or messages. Test labels do not determine hyperparameters or confidence thresholds.
5. Choose model settings and the confidence threshold using validation data. Reserve a separate part of the validation set for temperature calibration if calibration is added. Evaluate the locked procedure on the held-out test set.

## Initial models and experiment scope

- Baseline: two-layer MLP, initially `128 -> 64 -> 4`, ReLU and dropout.
- Selected model: two-layer GCN, initially `128 -> 64 -> 4`, with self-loops and symmetric degree normalization as in Kipf and Welling.
- Keep initial hidden width, dropout, maximum epochs and selection rules comparable. Select checkpoints using validation macro-F1 and use early stopping.
- Start with a short run that verifies forward propagation, finite loss, backward propagation, and decreasing training loss. This is a feasibility check, not a final performance claim.
- After feasibility is confirmed, compare three initialization seeds on the fixed split. Only expand experiments when needed to resolve an observed question.

## Evidence required in the final report

| Requirement | Planned evidence |
| --- | --- |
| Classification | Accuracy, per-class precision/recall/F1, macro-F1, multiclass AUROC, confusion matrix |
| Confidence dilemma | Confidence versus correctness, reliability diagram/ECE, coverage versus accepted-case error, validation-selected reject threshold |
| Embeddings | A reproducible t-SNE or UMAP plot with ground-truth colours and interpretation |
| Failure analysis | 3-5 actual misclassified nodes, including confidence, degree and neighbourhood context where informative |
| Resource trade-off | Parameter count, measured peak GPU memory and synchronized inference time; record hardware and batch/full-graph scope |
| Reproducibility | Split and initialization seeds, exact dependency versions, commands, checkpoints stored outside Git |
| Recommendation | An evidence-based assessment of GCN versus MLP and when review is still needed |

Do not invent missing metrics, claim the GCN outperforms the MLP before evaluation, or treat a proposed resource budget as a measurement.

## Implementation milestones

1. Dataset acquisition and audit are complete. Review the feasibility draft and complete the tutor check-off.
2. Implement `dataset.py`, `modules.py` and an MLP training check.
3. Implement the GCN and comparable experiments in `train.py`.
4. Add checkpoint loading and test examples in `predict.py`; generate plots and inspect failures.
5. Complete this README, export it for Turnitin, and open a PR to upstream `topic-recognition` from the student's fork.
6. Address the later teaching-team code review within its announced response window.

Keep meaningful commits as each real milestone is reached. Do not commit datasets, model weights, private keys or generated dependency environments.

## Dates to verify

The 2026 course profile lists the Pattern Analysis Project assessment period as **19-30 October 2026**. It does not establish a precise individual submission time or the new PDF's midpoint check-off date. Check Assessment Summary, the actual submission item and course announcements before setting a final schedule.

## Artificial Intelligence Usage Disclosure

Codex assisted with interpreting the supplied specification, checking primary references and drafting this project plan and feasibility outline. No model results have been generated or verified at this stage. The disclosure must be updated to describe actual implementation assistance and the student's audits and tests as development progresses.

Codex also drafted the data-loading/audit scripts and four synthetic software tests. The scripts were run against the actual course archive with NumPy 2.3.5 on Python 3.12.14; all four tests passed. Synthetic tests check software behaviour only and are not evidence of classification performance. The student still needs to review and understand the implementation.

## References

- Supplied course specification: `COMP3710_Report_2026_draft_v2.3.pdf`, particularly Sections 2.1, 3, 4, 5 and 6.
- PyG FacebookPagePage documentation: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.FacebookPagePage.html
- PyG dataset loader and original download URL: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/facebook.html
- Kipf and Welling, *Semi-Supervised Classification with Graph Convolutional Networks*: https://arxiv.org/abs/1609.02907
- PyG download issue: https://github.com/pyg-team/pytorch_geometric/issues/10525
- Course-linked data mirror and preprocessing description: https://github.com/gayanku/facebook_page-page_dataset
- 2026 course profile: https://course-profiles.uq.edu.au/course-profiles/COMP3710-60675-7660
