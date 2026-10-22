# Post-consultation documentation and threshold decision record

This record documents the completed feasibility check-off and tutor feedback as reported by the student, alongside observed results and the pending Ed clarification.

## Consultation and document history

- The feasibility check-off is completed. No exact completion date is assigned in this record.
- The tutor suggested including failure analysis in the experimental record; the student described the other consultation matters as satisfactory.
- The 15 October preliminary review and the 21 October updated review remain dated snapshots. Their then-pending status statements are not retrospectively rewritten. The final README now explains the later reported completion and the completed report/PR work.
- The initial Ed review/check-off request was reported posted on 21 October. A separate follow-up question about the unresolved final requirement interpretation has been prepared; sending is not confirmed and no reply has been recorded. The two communications must not be conflated.

## Locked experimental finding

The unchanged threshold rule scans 0, 0.01, ..., 1.00 on calibration nodes. It requires at least `max(1, floor(0.05 * N_calibration))` accepted nodes and at least 90% empirical accepted-case accuracy, maximises coverage, and breaks ties towards the smaller threshold. If no candidate qualifies, 1.01 rejects all predictions.

All three GCN runs select threshold 0 because the full calibration set already meets the 90% target. Test coverage is 100%; the five reported failures, including a low-confidence near-tie, are accepted. This satisfies the fitting objective while failing the intended ambiguity-routing purpose. Completing the analysis does not establish that the operational goal has been met.

## Decision and next actions

- Retain the original code, model weights, calibration settings, result files and figures. This documentation update neither retrains a model nor replaces an inconvenient result.
- Do not retune on the already inspected test labels or selected errors. The cases support evidence-led diagnosis, not a new independent validation claim.
- Preserve the confidence analysis and five failure autopsies while requirement clarification is unresolved.
- Possible future risk bounds, review budgets, ambiguity margins, content audits and graph ablations are proposals only. They require a new validation protocol and independent evaluation data before any efficacy claim.
- The Turnitin receipt has not been provided, and the later formal PR review response remains outstanding.

## Scope of this update

Only report documentation and the current status record are changed. Reported numerical results, nine existing figures, the six software checks and the original dated reviews are preserved. Failure discussions now distinguish evidence, hypothesis, limits and future action. These changes improve traceability and interpretation without asserting new experimental evidence.
