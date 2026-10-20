# Confidence-aware Facebook Page Classification with a GCN

**Yuhan Liao - 49839991 | COMP3710, Semester 2, 2026 | Final experiment report: 21 October 2026**

**Selected project:** Multi-layer GNN, Normal difficulty.

## 1. Purpose and principal findings

This Facebook-page classification assistant combines 128-dimensional page features and connections using a two-layer graph convolutional network (GCN). It predicts television-show, government, company or politician labels. A matched feature-only multilayer perceptron (MLP) tests graph context's value. The engineering question is whether classification and confidence gains justify additional cost, and whether uncertainty can guide human review.

Across three initialisations on one fixed split, GCN reaches **93.84% test accuracy**, versus **80.77% for MLP**, a **13.07 percentage-point** gain. GCN Macro-F1 is 0.9333 versus 0.7884. Nevertheless, very confident mistakes remain. The fitted GCN threshold is zero, accepting every case: it meets the original empirical accuracy target but fails to provide useful ambiguity routing. The recommendation is a stronger classification prototype with human oversight, rather than automatic deployment under the present policy.

![Test accuracy and Macro-F1 across three initialisation seeds](figures/model_comparison.png)

## 2. Data, preprocessing and leakage controls

The course-linked [Facebook Page-Page mirror](https://github.com/gayanku/facebook_page-page_dataset) provides **22,470 nodes and 128 finite features per node**. Class counts are tvshow 6,495; government 6,880; company 5,768; politician 3,327. Features are a supplied SVD compression of 4,714 description-token indicators; their upstream construction is not claimed as training-only.

The archive is pinned to commit `ec01d02c39da4385052023ed28e36f0794f61dc8`, SHA-256 `7a1b8a0687515e9859d1ca649dc276825f5db44568894dfbb6b55ebc1cf6a9c1`. Because its numeric keys differ from the standard PyG loader, `dataset.py` constructs `torch_geometric.data.Data` explicitly, without loading pickled page names. The audit validates dimensions, labels, finite values, endpoints and edges.

There are 342,004 raw edge entries, including 179 duplicates and 358 self-loops. Deduplication, symmetrisation and loop removal produce 341,646 non-self directed entries (both directions included); GCN then adds one self-loop per node. Duplicate links thus do not receive extra weight. Standardisation uses training-node mean/standard deviation only; near-zero deviations use scale one.

| Fixed stratified split | Nodes | Use |
| --- | ---: | --- |
| Training | 13,481 | Cross-entropy optimisation |
| Validation: model selection | 2,245 | Checkpoint and early stopping |
| Validation: calibration | 2,248 | Temperature and threshold fitting |
| Test | 4,496 | Final evaluation and post-evaluation inspection |

The 60/20/20 split uses per-class flooring and seed 42; role seed 43 partitions validation by class. Tests verify exhaustive, disjoint, reproducible splits and class coverage.

This is **transductive classification**: all features/edges are available, including unlabelled validation/test nodes. Labels never enter inputs or messages. Only training labels enter optimisation; selection/calibration labels serve separate roles. Test labels do not choose settings, checkpoints or thresholds. Inductive performance on new pages/graphs is untested.

## 3. Algorithm and implementation

GCN follows [Kipf and Welling](https://arxiv.org/abs/1609.02907). For binary symmetric adjacency A, define `S = D^(-1/2) (A + I) D^(-1/2)`, where D contains the row sums of `A + I`. Self-loops preserve the node's own signal; degree normalisation moderates neighbour contributions. The network computes `H = ReLU(S X W0 + b0)` and `Z = S Dropout(H) W1 + b1`. Two aggregation layers expose a node to two-hop feature context; neighbourhood labels are never supplied. Biases are added after aggregation.

Both networks use **128 -> 64 -> 4**, ReLU, dropout 0.5 and **8,516 parameters**. MLP removes S, treating nodes independently; aggregation is the main architectural difference. `modules.py` uses native PyTorch layers and sparse multiplication, without NumPy or prebuilt PyG convolutions. PyG supplies the data container; NumPy/scikit-learn support preprocessing/evaluation.

| File | Responsibility |
| --- | --- |
| `modules.py` | MLP, GCN and binary normalised sparse adjacency |
| `dataset.py` | Checksum-verified acquisition, validation, splits and training-only scaling |
| `train.py` | Explicit training loop, selection, calibration, metrics, checkpoint and profiling |
| `predict.py` | Reload saved weights, export predictions, embeddings and diagnostic figures |
| `evaluation.py` | Multiclass metrics, temperature fitting, ECE and selective prediction |
| `audit_dataset.py` | Numeric dataset and split-provenance audit |
| `test_data_pipeline.py`, `test_models.py` | Six data/model correctness checks |

English docstrings explain purposes, functions and protocol boundaries. Six checks passed before formal runs, covering splits/class coverage, training-only scaling, invalid arrays, duplicate/self-loop handling, adjacency correctness, and permutation equivariance/finite gradients. Software checks do not establish predictive quality.

## 4. Training, model selection and saved-model inference

Matched settings are Adam, learning rate 0.01, weight decay 0.0005, maximum 200 epochs, patience 30 and initialisation seeds **42, 43, 44**. Full-graph forwards compute training-masked cross-entropy. Selection Macro-F1 determines the checkpoint, restored before calibration/testing; losses and gradients must be finite.

![MLP seed-42 training and selection histories](figures/mlp_training.png)

![GCN seed-42 training and selection histories](figures/gcn_training.png)

Loss falls for both models. GCN seed 42 selects epoch 167 and stops at 197; the last epoch is not necessarily best. All three runs are reported. Seeds change initialisation/stochastic training, **not the split**. Means/sample standard deviations describe optimisation variability, not confidence intervals or variation across independent splits.

Checkpoints save architecture, weights, checksum, seeds, scaling, temperature and threshold. `predict.py` verifies checksum/scaling, restores weights and uses evaluation mode without gradients; non-final checkpoints cannot evaluate test labels. Both seed-42 checkpoints were reloaded, exporting 4,496 predictions each; CSV accuracy agrees with saved evaluation.

## 5. Held-out classification results

Metrics below use raw softmax scores; positive temperature scaling preserves argmax labels, so accuracy, F1 and confusion counts are unchanged. AUROC is macro one-versus-rest; ranking scores can change slightly after multiclass probability rescaling.

| Model | Test accuracy (%) | Macro-F1 | Macro OVR AUROC |
| --- | --- | --- | --- |
| MLP | 80.77 +/- 0.45 | 0.7884 +/- 0.0054 | 0.94685 +/- 0.00029 |
| GCN | 93.84 +/- 0.32 | 0.9333 +/- 0.0032 | 0.99005 +/- 0.00038 |

| Model / seed | Epochs / best epoch | Test accuracy (%) | Macro-F1 | AUROC |
| --- | --- | --- | --- | --- |
| MLP / 42 | 200 / 175 | 80.63 | 0.7849 | 0.94658 |
| MLP / 43 | 148 / 118 | 80.40 | 0.7857 | 0.94683 |
| MLP / 44 | 200 / 194 | 81.27 | 0.7946 | 0.94715 |
| GCN / 42 | 197 / 167 | 93.75 | 0.9323 | 0.99007 |
| GCN / 43 | 144 / 114 | 93.57 | 0.9308 | 0.98966 |
| GCN / 44 | 200 / 190 | 94.19 | 0.9369 | 0.99042 |

### Per-class results and confusion analysis

The **prespecified seed 42** provides P/R/F1 (precision/recall/F1) below; overall accuracy can obscure smaller-class weaknesses.

| Class | Test n | MLP P / R / F1 | GCN P / R / F1 |
| --- | --- | --- | --- |
| tvshow | 1299 | 0.8336 / 0.8483 / 0.8409 | 0.9392 / 0.9269 / 0.9330 |
| government | 1376 | 0.7940 / 0.8685 / 0.8296 | 0.9323 / 0.9615 / 0.9467 |
| company | 1155 | 0.8165 / 0.8165 / 0.8165 | 0.9519 / 0.9602 / 0.9560 |
| politician | 666 | 0.7490 / 0.5781 / 0.6525 | 0.9190 / 0.8694 / 0.8935 |

![Seed-42 test confusion matrices; rows are true and columns predicted classes](figures/confusion_matrices.png)

Matrix numbers are counts; colour is normalised within each true-class row. GCN improves all four class F1 scores over MLP. Politician is still weakest: recall 0.8694 means **87 of 666 politicians are missed**, including 54 predicted as tvshow. Conversely, 45 tvshow nodes are predicted as politicians. Each misclassification is a false negative for its true class and a false positive for its predicted class. High aggregate accuracy therefore does not eliminate consequential class-specific errors.

## 6. Confidence, calibration and the reject-option limitation

Confidence is the maximum softmax probability. Scalar temperature T is fitted by minimising cross-entropy on calibration nodes only, then probabilities are `softmax(Z/T)`. T is positive (clamped to 0.05-20), so it adjusts sharpness without changing the predicted class. ECE uses ten equal-width confidence bins, weighted by their population; it depends on binning and can conceal sparse-region errors.

| Model | Test NLL: raw -> scaled | Test ECE: raw -> scaled | Fitted T (42 / 43 / 44) |
| --- | --- | --- | --- |
| MLP | 0.5597 -> 0.5533 | 0.0345 -> 0.0400 | 1.163 / 1.087 / 1.171 |
| GCN | 0.2227 -> 0.2223 | 0.0250 -> 0.0161 | 0.866 / 0.847 / 0.875 |

These are held-out means. GCN ECE improves; MLP ECE worsens despite improved mean NLL. GCN seed-42 NLL slightly worsens (0.222336 -> 0.222389). Calibration optimisation guarantees no universal held-out improvement.

![MLP seed-42 reliability and risk-coverage curves](figures/mlp_confidence.png)

![GCN seed-42 reliability and risk-coverage curves](figures/gcn_confidence.png)

Reliability compares confidence with correctness; the diagonal is ideal. Risk-coverage plots sort predictions by confidence, showing accepted-case error against coverage. These test diagnostics do not define a new test-tuned policy. Near-diagonal regions coexist with near-certain errors.

The implemented threshold grid is 0, 0.01, ..., 1.00. Candidates must accept at least `max(1, floor(0.05 * N_calibration))` nodes and achieve at least **90% empirical calibration accepted-case accuracy**. The objective maximises coverage, breaking ties towards the smallest threshold. If none qualifies, 1.01 rejects everything. The rule is empirical, without a statistical risk guarantee.

| Model / seed | Threshold | Test coverage (%) | Accepted test accuracy (%) |
| --- | --- | --- | --- |
| MLP / 42 | 0.63 | 71.82 | 90.06 |
| MLP / 43 | 0.63 | 71.22 | 90.54 |
| MLP / 44 | 0.62 | 73.58 | 89.66 |
| GCN / 42 | 0.00 | 100.00 | 93.75 |
| GCN / 43 | 0.00 | 100.00 | 93.57 |
| GCN / 44 | 0.00 | 100.00 | 94.19 |

**All GCN thresholds are zero.** Its full calibration set already exceeds 90%, so 100% coverage maximises the objective. This is the expected result of the chosen objective, not a numerical fitting fault. It nevertheless accepts low-confidence cases as well as confident mistakes and fails the intended ambiguity-routing goal. MLP rejects roughly 26-29%, but seed 44 accepted test accuracy is 89.66%, below the fitting target: calibration performance does not guarantee held-out performance. The locked rule is reported unchanged after test inspection. A stricter risk bound, explicit review budget or ambiguity margin would need separate validation; none is claimed as implemented or tested here.

## 7. Embedding visualisation and interpretation

![MLP seed-42 hidden representations with ground-truth colours](figures/mlp_embedding.png)

![GCN seed-42 hidden representations with ground-truth colours](figures/gcn_embedding.png)

These t-SNE plots use test-node 64-dimensional hidden representations, ground-truth colours, up to 500 randomly selected points per class, seed 42, PCA initialisation, automatic learning rate and perplexity 30. Both show broad class-associated regions, including a company-dominated region and substantial mixed central areas. The GCN plot shows a more distinct politician-dominated region, but tvshow/politician and other boundary mixing remains, consistent with the confusion matrix. This is qualitative support only. Separate projections make coordinates and distances incomparable between models; t-SNE can distort global geometry, the class-balanced sample changes apparent density, and visual separation does not measure accuracy or prove the mechanism of improvement.

## 8. Five real failure autopsies

Cases come from the fixed seed-42 GCN test predictions. The selection rule samples high confidence, low degree, low margin and high degree before filling unique cases; it is not a random error sample. Confidence below is temperature-scaled. All five are accepted by GCN because its threshold is zero.

| Node | True -> GCN | GCN confidence | Degree | MLP prediction / confidence / accepted |
| --- | --- | --- | --- | --- |
| 16052 | government -> company | 1.000000 | 266 | company / 0.906357 / True |
| 9254 | tvshow -> politician | 1.000000 | 16 | politician / 0.993158 / True |
| 22382 | politician -> tvshow | 0.815955 | 1 | company / 0.563093 / False |
| 18785 | politician -> government | 0.263498 | 2 | government / 0.408311 / False |
| 953 | tvshow -> company | 0.999799 | 4 | politician / 0.344795 / False |

1. **16052: confident neighbourhood conflict.** Degree 266; all 161 training-labelled neighbours are company, and 99.6% of neighbours are predicted company. This suggests topology can reinforce an incorrect category. MLP is also wrong, so graph aggregation alone is not proven responsible; feature ambiguity or label mismatch is possible but unverified. Review original content before diagnosing a wrong label.
2. **9254: near-certain error.** All eight training-labelled neighbours are politicians, although the node is labelled tvshow. Both models confidently predict politician. Shared feature/context ambiguity is plausible. Temperature scaling raises confidence rather than correcting the class. This case refutes treating a high score as a correctness guarantee.
3. **22382: sparse local support.** Its single non-self neighbour has a training tvshow label; GCN predicts tvshow at 0.816. MLP also fails, predicting company, but its threshold sends the case to review. Sparse context motivates checking degree-sensitive error patterns; a single case does not establish that low degree causes failure.
4. **18785: explicit ambiguity missed by policy.** Government is predicted at 0.2635, with top-two margin 0.0015. Degree is two and the one training-labelled neighbour is politician. MLP also predicts government but refers it. The GCN decision is almost tied and should attract human attention, yet the current maximum-coverage rule accepts it.
5. **953: confident error despite mixed labelled context.** Degree four, with one tvshow and one company training neighbour; all neighbours' predicted labels agree with company. GCN predicts company at 0.9998. MLP predicts politician at 0.3448 and refers it. Local prediction agreement can reinforce error rather than establish truth.

![Training-neighbour label context for the five GCN errors](figures/failure_neighbourhood.png)

These are numeric hypotheses, not causal explanations. Without page text/names, semantic triggers and annotation mistakes remain unverified; controlled ablation/content review is needed. Cases support reporting, not further tuning.

## 9. Computational cost and measured scope

Formal experiments ran on **Rangpur, NVIDIA A100-PCIE-40GB**, driver 590.48.01, Slurm job 680694, partition `a100-test`, shared allocation `shard:1` and two requested CPUs. These are actual measurements, not the preliminary budget.

| Measured quantity | MLP | GCN |
| --- | --- | --- |
| Parameters | 8,516 | 8,516 |
| Peak allocated GPU memory (MiB) | 48.27 | 82.98 |
| Peak reserved GPU memory (MiB), seeds 42/43/44 | 56 / 64 / 64 | 96 / 96 / 96 |
| Full-graph forward median (ms), mean +/- SD | 0.293 +/- 0.006 | 0.780 +/- 0.001 |
| Training loop seconds, mean +/- SD | 2.992 +/- 1.414 | 2.949 +/- 0.589 |

Each run's forward time is the median of ten full-graph passes after three warm-ups, with CUDA synchronisation, evaluated on all 22,470 nodes. It excludes loading and adjacency construction; it is neither per-node nor end-to-end request latency. Reported mean/SD aggregates those three medians. GCN costs about 2.66 times the forward time and 1.72 times the allocated memory of MLP, while both remain small in absolute terms here.

PyTorch peaks cover training, calibration and forward profiling before test metrics; they exclude other users' allocations and are not total device occupancy. Training time includes epoch validation/checkpoint copying, excluding data/adjacency setup, calibration and plotting. Different stopping epochs, cold starts and sharing prevent claiming faster GCN training. The running job snapshot is historical; later terminal evidence shows the session ended and the user's queue empty after download.

## 10. Reproduction and example usage

Run commands from `recognition/Facebook_GCN/`. First install an appropriate PyTorch CPU/CUDA build, then `python -m pip install -r requirements.txt`. The actual formal environment was **Python 3.11.15; PyTorch 2.13.0 (`+cu130` reported at runtime); CUDA 13.0; torch-geometric 2.8.1; NumPy 2.4.6; scikit-learn 1.9.1; matplotlib 3.11.2**, Linux x86-64/glibc 2.28. `requirements.txt` permits a wider compatible range; for matching runs pin these direct versions with a suitable PyTorch wheel. Hardware/library changes may affect numerical results and timing; seed settings do not guarantee bitwise equality.

```bash
python audit_dataset.py --download
python -m unittest discover -v
python train.py --smoke --device cpu --epochs 5 --models mlp gcn
python predict.py runs/smoke/gcn_seed42/model.pt --split validation --embedding
```

Smoke mode caps training at five epochs and refuses test evaluation. For the unchanged formal protocol, use the same preset without tuning on the published test results:

```bash
python train.py --models mlp gcn --seeds 42 43 44 --device cuda --epochs 200 --evaluate-test
python predict.py runs/final/mlp_seed42/model.pt --split test --device cuda --embedding
python predict.py runs/final/gcn_seed42/model.pt --split test --device cuda --embedding
```

Input is the checksum-verified numeric archive, converted to `x: [22470,128]` and canonical `edge_index: [2,341646]`. Outputs include `model.pt`, `summary.json`, `history.csv`, `training.png`, then `predictions_test/predictions.csv`, `evaluation.json`, `confidence.png` and optional `embedding.png`. CSV fields are node ID, true/predicted numeric label, confidence and accepted flag; labels 0-3 follow tvshow/government/company/politician. For example, node 18785 yields predicted label 1, true label 3, confidence approximately 0.263498, accepted True.

Datasets, weights, environments and generated run folders are excluded from Git. Reproduce checkpoints locally or retain private backups; do not expect weights to be bundled in the PR. The Colab demo reads saved evidence by default; regenerating displayed tables/plots is not a new formal training run. Fresh prediction requires the saved checkpoint and dataset.

## 11. Engineering recommendation and remaining limitations

Choose GCN for the next assisted-classification prototype: its accuracy/F1 gain is substantial on this split for modest absolute inference/memory cost. Keep human review rather than approve the present all-accept policy. Prioritise a separately validated review budget/risk requirement, class-specific politician errors, and explicit ambiguous-case handling. MLP is a cheaper fallback when graph access is unavailable, though its quality is substantially lower here.

External validity remains limited by one graph, one random split, globally supplied features, no inductive/time-shift evaluation, one small architecture and no extensive hyperparameter search. Three initialisations do not replace repeated splits. Small-bin ECE and five selected failures cannot establish deployment safety. New policy development must reserve independent evaluation data after the present test inspection. No proposed improvement in this section is represented as a completed experiment.

## 12. Historical one-page feasibility review - 15 October 2026

The following preserves the **preliminary** review; its unfinished-experiment statements describe 15 October, not current results. The original editable version is [feasibility_review_draft.md](feasibility_review_draft.md). The [updated review dated 21 October](feasibility_review_current.md) accurately discloses the completed experiments. The student reports posting the review and check-off request on Ed on 21 October; tutor confirmation is still pending. Neither document nor posting establishes a recorded check-off.

**Project:** Confidence-aware Facebook Page Classification - Normal GNN

**Working draft - 15 October 2026.** Data and short CPU checks are complete. Tutor check-off and formal experiments remain outstanding.

**Scope and user need.** Build a Normal-difficulty node-classification prototype that helps a reviewer categorize Facebook pages and refer uncertain predictions for manual review. Investigate whether a multi-layer GCN provides more useful and reliable predictions than a feature-only MLP at an acceptable resource cost.

**Proposed data and model.** Use the course-linked FacebookPagePage mirror, with 128-dimensional SVD features and four classes, converted to native PyG Data. Compare a two-layer GCN with a feature-only MLP; both use hidden width 64 and have 8,516 parameters. GCN adds normalized neighbour aggregation, linking the project to graph structure, matrix operations and semi-supervised learning. Use fixed stratified train/validation/test splits. Split validation further into model-selection and calibration roles. The transductive model uses all node features and edges, but only training labels enter optimization; labels are never messages or input features.

**Acceptance criteria.**

1. Verify dataset dimensions and edge validity, and demonstrate that the saved split is reproducible, exhaustive and disjoint.
2. Train both models and report per-class precision/recall/F1, macro-F1 and AUROC on the same held-out split. Diagnose the outcome whether or not GCN improves on MLP.
3. Select a confidence threshold using validation data and report held-out coverage and accepted-case error. Compare confidence reliability between the models.
4. Provide an interpreted t-SNE/UMAP visualization and explain 3-5 actual failure cases.
5. Measure parameter count, inference time and GPU memory. The planning budget is one GPU with a 4 GiB peak-memory target; this remains unmeasured and does not assume particular hardware availability.

**Preliminary evidence.** Data audit: 22,470 nodes, 128 finite float32 features, four classes and 342,004 raw edge entries. Seed 42 gives 13,481/4,493/4,496 training/validation/test nodes without overlap; validation contains 2,245 selection and 2,248 calibration nodes. All six software tests passed. Five CPU epochs reduced MLP training loss from 1.421 to 1.082 and GCN loss from 1.393 to 1.069. Both saved models were reloaded successfully. These checks establish execution feasibility, not final quality. The 4,496-node test set remains unevaluated. Source checksums, splits, versions and short-run evidence are saved with the project.

**Risks, budget and fallback.** Main risks are leakage, oversmoothing and overconfident errors. Keep labels out of message passing, use training-only feature scaling, and use MLP as the operational fallback if graph context adds insufficient value. Canonicalize duplicate/self edges before GCN normalization. The new Virtual GPU Lab has a finite quota; Rangpur is another option. CPU feasibility checks used no school GPU time. Check the actual GPU and package environment before longer jobs.

**Next steps and schedule.** Discuss this review with the tutor and record check-off, then run matched experiments across three initialization seeds, inspect confidence and failures, and complete the report. The course profile lists 19-30 October 2026; confirm the individual lab slot, midpoint requirement and precise submission time. Include this review in the final README.

**References:** Kipf and Welling (2017), GCN; course-linked data mirror `gayanku/facebook_page-page_dataset` (Appendix A.6).

## 13. Submission status

The modular implementation, formal experiments and local result backups are complete. This README includes the final evidence and historical feasibility review. A progressive Git history, a working PR against upstream `topic-recognition`, its later review response, the tutor-recorded check-off and the Turnitin PDF are separate assessment requirements. Their completion must be checked independently; this report does not claim submission or check-off. The selected Normal problem does not qualify for the Hard-only implementation criterion.

## Artificial Intelligence Usage Disclosure

**ChatGPT and Codex** provided substantial assistance with interpreting requirements, finding/checking references and the course data link, planning experiments, scaffolding and refining data/model/training/evaluation code and tests, debugging environment/SSH issues, preparing plotting and reporting material, interpreting results, and producing the Colab/tutor preparation material. This is broader than autocomplete; it is not represented as wholly independent manual coding. The GCN formulation is attributed to Kipf and Welling, and dataset handling to the cited course/PyG sources.

Verification included the checksum/data audit, leakage and graph-normalisation checks, six passing software tests, five-epoch CPU feasibility runs, actual six-model Rangpur experiments, checkpoint reloading, prediction-CSV consistency checks and inspection of saved plots/errors. These checks verify particular properties, not every possible defect or deployment safety. The student remains responsible for understanding and explaining every submitted operation, loss and metric; AI-generated explanations must be checked against code and evidence. The threshold-zero limitation and calibration regressions are disclosed rather than replaced with an untested improvement.

## References

1. COMP3710 teaching team. *Final Report Assessment*, draft v2.3, 2026, Sections 2.1 and 3-6; supplied assessment document.
2. T. N. Kipf and M. Welling. *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR, 2017. [Paper](https://arxiv.org/abs/1609.02907).
3. C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger. *On Calibration of Modern Neural Networks*. ICML, 2017. [Paper](https://arxiv.org/abs/1706.04599).
4. PyTorch Geometric. [FacebookPagePage documentation](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.FacebookPagePage.html) and [loader source](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/facebook.html).
5. Course-linked [Facebook Page-Page numeric dataset and preprocessing description](https://github.com/gayanku/facebook_page-page_dataset), pinned commit and checksum in Section 2.
