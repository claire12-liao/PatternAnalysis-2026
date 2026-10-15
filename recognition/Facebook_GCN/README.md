# Confidence-aware Facebook Page Classification with a GCN

COMP3710, Semester 2 2026. Selected project: multi-layer GNN node classification, Normal difficulty.

## Current status - 15 October 2026

- The student selected the Normal GNN project.
- The course repository has been cloned from `topic-recognition`, starting at commit `552b902`. The course remote is named `upstream`.
- Dataset specifications and the GCN reference have been checked against primary sources.
- The course data mirror was located in the PDF's embedded link in Appendix A.6: https://github.com/gayanku/facebook_page-page_dataset . The original PyG download domain is unavailable from the current environment.
- The pinned course archive has been downloaded and audited. It has 22,470 nodes, 128 finite feature values per node, four classes and 342,004 raw edge entries. See `reports/data_audit.json` for the checksum, provenance and complete audit.
- Split seed 42 gives 13,481 training, 4,493 validation and 4,496 test nodes. The splits cover every node exactly once.
- The required `modules.py`, `dataset.py`, `train.py` and `predict.py` are implemented. Actual PyG conversion, five CPU training epochs for each model, checkpoint saving/reloading, validation predictions, confidence plots and a GCN t-SNE plot have run successfully.
- All six software tests passed, including graph normalization and permutation equivariance. See `reports/feasibility_smoke.json` for the preliminary evidence and exact software versions.
- These are short feasibility runs, not final performance estimates. The test set has not been evaluated. Formal experiments, GPU resource measurements, manual interpretation and the final report remain outstanding.
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
5. The 4,493 validation nodes are split into 2,245 model-selection and 2,248 calibration/threshold-fitting nodes using role seed 43. Select checkpoints with the former; fit scalar temperature and an empirical reject threshold with the latter. Evaluate the locked procedure on the held-out test set.

## Reproducing the current feasibility stage

Use a project environment with a suitable PyTorch build, then install the remaining dependencies in `requirements.txt`. `requirements-verified-cpu.txt` records the exact direct versions used locally; GPU runs need an appropriate CUDA-enabled PyTorch build and their own recorded versions.

From this folder:

```bash
python audit_dataset.py --download
python -m unittest discover -v
python train.py --smoke --device cpu --epochs 5 --models mlp gcn
python predict.py runs/smoke/gcn_seed42/model.pt --split validation --embedding
```

The smoke mode caps training at five epochs and refuses test evaluation. Checkpoints, predictions and figures are stored under `runs/`, which is excluded from Git. The short runs have 8,516 parameters per model and declining training losses; they do not establish the final comparative quality of the methods.

The one-page feasibility draft is in `outputs/feasibility_review.pdf`; its editable text is `feasibility_review_draft.md`. A tutor consultation must still be completed and recorded.

## Formal experiment procedure - not yet executed

After the compulsory check-off and before viewing any test results, lock the data split, model settings, seed list and operating-point fitting rule. The initial matched settings are hidden width 64, dropout 0.5, Adam learning rate 0.01, weight decay 0.0005, up to 200 epochs and patience 30, with initialization seeds 42, 43 and 44.

```bash
python train.py --models mlp gcn --seeds 42 43 44 --device cuda --epochs 200 --evaluate-test
python predict.py runs/final/gcn_seed42/model.pt --split test --device cuda --embedding
```

The final command evaluates a locked procedure; do not use its test scores to choose settings or select the most flattering seed. Use the prespecified seed 42 for qualitative plots and case studies, and report aggregate performance over all three initialization seeds. The fixed split does not measure variation across different data splits. Calibration/threshold fitting statistics are in-sample for that fitting stage; independent estimates come from the final test set.

Record the actual GPU, package versions, peak allocated/reserved memory and synchronized full-graph inference time. No GPU resource measurement is available from the CPU feasibility stage. In the quota-limited Virtual GPU Lab, preserve code and results before ending the instance; check the environment rather than assuming the supplied COMP4703 environment includes every GNN dependency.

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

1. Complete: course data acquisition/audit, reproducible splits and six software tests.
2. Complete: MLP/GCN prototype, five-epoch CPU checks and saved-model validation inference.
3. Next: review the one-page feasibility draft and complete the tutor check-off.
4. Run the locked matched experiments, measure actual GPU use and evaluate the held-out test set.
5. Inspect 3-5 failures, interpret embedding/confidence plots and write the evidence-based recommendation.
6. Complete this README, export it for Turnitin, and open a PR to upstream `topic-recognition` from the student's fork.
7. Address the later teaching-team code review within its announced response window.

Keep meaningful commits as each real milestone is reached. Do not commit datasets, model weights, private keys or generated dependency environments.

## Dates to verify

The 2026 course profile lists the Pattern Analysis Project assessment period as **19-30 October 2026**. It does not establish a precise individual submission time or the new PDF's midpoint check-off date. Check Assessment Summary, the actual submission item and course announcements before setting a final schedule.

## Artificial Intelligence Usage Disclosure

Codex assisted with specification interpretation, primary-reference checks, project planning, the feasibility draft, and drafting/refining the data pipeline, MLP/GCN implementation, training/evaluation scripts and software tests. Validation included an actual archive audit; six synthetic software checks; five CPU training epochs for both models; checkpoint reloading; validation predictions; and confidence/t-SNE figure generation. Exact run versions are recorded in `reports/feasibility_smoke.json`. The final test set and school GPU quota were not used during these checks.

Synthetic tests check software behaviour and the short runs check execution feasibility; neither establishes final comparative performance. The student still needs to review, understand and verify the implementation, complete the tutor consultation and update this disclosure as formal experiments progress.

## References

- Supplied course specification: `COMP3710_Report_2026_draft_v2.3.pdf`, particularly Sections 2.1, 3, 4, 5 and 6.
- PyG FacebookPagePage documentation: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.FacebookPagePage.html
- PyG dataset loader and original download URL: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/facebook.html
- Kipf and Welling, *Semi-Supervised Classification with Graph Convolutional Networks*: https://arxiv.org/abs/1609.02907
- PyG download issue: https://github.com/pyg-team/pytorch_geometric/issues/10525
- Course-linked data mirror and preprocessing description: https://github.com/gayanku/facebook_page-page_dataset
- 2026 course profile: https://course-profiles.uq.edu.au/course-profiles/COMP3710-60675-7660
