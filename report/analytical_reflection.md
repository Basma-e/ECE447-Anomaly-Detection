# Analytical Reflection - Anomaly Detection Project

## Context
This project treats credit-card fraud detection as a semi-supervised anomaly detection problem. Models are trained on normal-heavy data and evaluated on validation/test sets containing both normal and fraudulent transactions. Because fraud is rare, operating-point decisions are driven by the precision-recall-false-positive trade-off rather than accuracy.

## Trade-offs in threshold selection
All three methods (robust statistical z-score, distance-based kNN, and Isolation Forest) output continuous anomaly scores. The key deployment decision is where to place the threshold.

At lower thresholds, recall increases because more transactions are flagged as suspicious, but false positives rise quickly. In fraud operations, this can overwhelm analysts, increase customer friction, and create investigation backlog. At higher thresholds, precision and false-positive rate usually improve, but true frauds are missed. Missing fraud has direct monetary and trust costs.

In this work, thresholds are selected on the validation split by sweeping score quantiles and maximizing F1, while still examining false-positive rate. This gives a practical compromise:
- F1 protects against optimizing precision or recall in isolation.
- Quantile sweeps are robust across score scales and model families.
- FPR is reported explicitly so threshold decisions can map to operational capacity.

From an engineering perspective, the threshold should ultimately be tied to business constraints: maximum daily alert volume, tolerated customer-review latency, and cost asymmetry between false negatives (fraud loss) and false positives (manual review burden). A defensible policy is to choose the highest-recall threshold that keeps FPR under a pre-agreed budget.

## Impact of imbalance
Class imbalance is severe in credit-card fraud data, so naive metrics can be misleading. A model can achieve very high accuracy while detecting little fraud. This is why we prioritize:
- Precision, recall, and F1 at the selected threshold.
- PR curves (more informative than ROC in rare-event settings).
- Confusion matrices to expose absolute false-positive and false-negative counts.

Imbalance also affects calibration and threshold stability. Small score shifts in the majority class can create large absolute increases in false positives. Distance-based methods can become especially sensitive in sparse regions, and unsupervised methods may over-flag benign but uncommon legitimate patterns.

Another consequence is data-split design: the training split should represent normal behavior reliably, while validation/test must include fraud examples to tune and evaluate threshold behavior. The shared team pipeline enforces this split logic consistently across methods, reducing accidental leakage and making comparisons fairer.

## Failure scenarios in production
Even well-performing offline models can fail after deployment. Likely failure scenarios include:

1. **Concept drift and behavior drift**  
   Customer purchasing behavior changes over time (seasonality, geographic shifts, new merchant categories). Score distributions drift, and fixed thresholds become stale.

2. **Adversarial adaptation**  
   Fraud tactics evolve in response to controls. Attackers mimic normal behavior, reducing separability and recall.

3. **Data quality failures**  
   Missing values, schema changes, scaling mismatches, delayed feature pipelines, or duplicate event ingestion can distort scores and trigger false spikes.

4. **Operational overload**  
   If alert volume exceeds analyst capacity, effective recall drops in practice because triage quality declines, even if model metrics appear acceptable.

5. **Feedback-loop bias**  
   Labels from investigated alerts may be biased toward previously flagged patterns, reducing visibility into novel fraud types.

These scenarios emphasize that anomaly detection quality is not only model quality; it depends on data reliability, process capacity, and human-in-the-loop workflows.

## Monitoring and retraining strategy
A practical monitoring plan should combine model, data, and operations signals:

- **Data monitoring:** track feature completeness, null rates, range shifts, and schema integrity.
- **Score monitoring:** monitor score distribution drift and alert-rate drift by method.
- **Performance monitoring:** once labels arrive, track precision/recall/F1/FPR and confusion-matrix trends.
- **Operations monitoring:** track queue length, review latency, and analyst throughput to keep thresholds operationally realistic.

Retraining should be triggered by measurable conditions instead of fixed calendar dates alone:
- Significant drift in score distributions or feature statistics.
- Sustained precision/recall degradation after labels mature.
- Major product or transaction-pattern changes.

Recommended retraining process:
1. Rebuild splits with the shared pipeline and latest labeled window.
2. Re-run threshold sweeps on validation to re-select operating points.
3. Log all experiments and artifacts in MLflow for auditability.
4. Compare candidate models against current production baseline under the same FPR budget.
5. Roll out with shadow or staged deployment and monitor early-warning metrics.

This approach keeps the anomaly system adaptive while preserving traceability and governance.

## Evidence linkage
Experiment evidence is packaged in `artifacts/submission_evidence/`, including exported run metadata and evaluation plots. Additional MLflow UI screenshots are placed under `artifacts/submission_evidence/mlflow_ui/` to provide visual tracking proof for submission.
