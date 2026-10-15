# Feasibility Review: Confidence-aware Facebook Page Classification

**Working draft - data audit complete; review and understand this plan before discussing it with the tutor. No tutor check-off is recorded here.**

**Scope and user need.** Build a Normal-difficulty node-classification prototype that helps a reviewer categorize Facebook pages and refer uncertain predictions for manual review. Investigate whether a multi-layer GCN provides more useful and reliable predictions than a feature-only MLP at an acceptable resource cost.

**Proposed data and model.** Use PyG FacebookPagePage with the specified 128-dimensional features and four target classes. Begin with a two-layer GCN and a two-layer MLP with hidden width 64. The GCN combines node features with normalized neighbour aggregation, linking the project to graph structure, matrix operations and semi-supervised learning. Use a fixed stratified 60/20/20 split. The transductive model may use all node features and edges, but only training labels enter optimization; held-out labels are never messages or input features.

**Acceptance criteria.**

1. Verify dataset dimensions and edge validity, and demonstrate that the saved split is reproducible, exhaustive and disjoint.
2. Train both models and report per-class precision/recall/F1, macro-F1 and AUROC on the same held-out split. Diagnose the outcome whether or not GCN improves on MLP.
3. Select a confidence threshold using validation data and report held-out coverage and accepted-case error. Compare confidence reliability between the models.
4. Provide an interpreted t-SNE/UMAP visualization and explain 3-5 actual failure cases.
5. Measure parameter count, inference time and GPU memory. The initial planning budget is one Rangpur GPU and a 4 GiB peak-memory target; this is a target to test, not a measured requirement or a guarantee of available hardware.

**Preliminary evidence.** The course-linked mirror has been downloaded and audited: 22,470 nodes, 128 finite float32 features per node, four classes and 342,004 raw edge entries. Fixed split seed 42 gives 13,481/4,493/4,496 training/validation/test nodes with no overlap. The numeric feature array occupies 10.97 MiB; this is not peak training memory. Four synthetic software tests passed, covering split validity, training-only scaling, graph-edge cleaning and malformed inputs. The archive checksum and source commit are recorded in `reports/data_audit.json`. There are no model-training or classification-performance results yet.

**Risks, budget and fallback.** Main risks are label leakage, oversmoothing, overconfident errors and compute availability. Keep labels separate from model inputs, start with a shallow GCN and use the MLP as the comparison and operational fallback. The mirror uses SVD features and custom array names, so construct PyG Data explicitly rather than assuming the stock loader is compatible. Run short checks before longer jobs. A new quota-limited Virtual GPU Lab is available according to the course announcement; Rangpur is another option. Confirm the precise submission and check-off dates; the course profile gives an assessment period of 19-30 October 2026.

**Next experiment.** Run short MLP/GCN training and backpropagation checks, recording the actual environment and resource use, before longer GPU experiments. Incorporate this review into the final README and record the required tutor consultation separately.
