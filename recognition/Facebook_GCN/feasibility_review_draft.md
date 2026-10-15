# Feasibility Review: Confidence-aware Facebook Page Classification

**Draft - complete the empirical feasibility evidence before presenting this review to the tutor.**

**Scope and user need.** Build a Normal-difficulty node-classification prototype that helps a reviewer categorize Facebook pages and refer uncertain predictions for manual review. Investigate whether a multi-layer GCN provides more useful and reliable predictions than a feature-only MLP at an acceptable resource cost.

**Proposed data and model.** Use PyG FacebookPagePage with the specified 128-dimensional features and four target classes. Begin with a two-layer GCN and a two-layer MLP with hidden width 64. The GCN combines node features with normalized neighbour aggregation, linking the project to graph structure, matrix operations and semi-supervised learning. Use a fixed stratified 60/20/20 split. The transductive model may use all node features and edges, but only training labels enter optimization; held-out labels are never messages or input features.

**Acceptance criteria.**

1. Verify dataset dimensions and edge validity, and demonstrate that the saved split is reproducible, exhaustive and disjoint.
2. Train both models and report per-class precision/recall/F1, macro-F1 and AUROC on the same held-out split. Diagnose the outcome whether or not GCN improves on MLP.
3. Select a confidence threshold using validation data and report held-out coverage and accepted-case error. Compare confidence reliability between the models.
4. Provide an interpreted t-SNE/UMAP visualization and explain 3-5 actual failure cases.
5. Measure parameter count, inference time and GPU memory. The initial planning budget is one Rangpur GPU and a 4 GiB peak-memory target; this is a target to test, not a measured requirement or a guarantee of available hardware.

**Preliminary evidence and remaining check.** The course repository branch and dataset documentation have been checked. The documented dataset contains 22,470 nodes; its float32 feature matrix alone would occupy approximately 11 MiB. This is a calculation and does not estimate total training memory. The original download domain failed a DNS check on 15 October 2026. The lecturer's mirror is still required. Before the tutor consultation, add the real dataset checksum, shape/class/split audit and a short MLP/GCN execution result or measured resource estimate. No training result is currently available.

**Risks, budget and fallback.** Main risks are inaccessible or mismatched data, label leakage, oversmoothing and overconfident errors. Confirm the 128-dimensional dataset before implementation, keep labels separate from model inputs, start with a shallow GCN and use the MLP as the comparison and operational fallback. Run a short feasibility experiment before longer Rangpur jobs. Confirm the submission and check-off dates in the course assessment pages; the public profile currently gives an assessment period of 19-30 October 2026.

**Next experiment.** Acquire and audit the approved dataset, then run a short training/backpropagation check and record the actual environment and resource use. Incorporate this review into the final README after the required tutor consultation.
