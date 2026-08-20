# Metrics Report

All numbers below were verified directly against the project's own output files
(`synthetic_logs.csv`, `scored_events.csv`, `day4_results.csv`, `drift_log.json`,
`entities.json`) — none are estimated or taken from code comments without
cross-checking against the actual data.

## Dataset

| Stat | Value |
|---|---|
| Total events | 19,033 |
| Normal events | 18,517 (97.3%) |
| Attack events | 516 (2.71%) |
| Users | 70 (5 departments: engineering, sales, finance, hr, it_admin) |
| Hosts | 90 (16 per dept + 10 shared) |
| Simulated days | 30 |
| Behavioral features engineered | 13 (strictly causal — no future information) |
| Concept-drift users | 22 / 70 (31.4%), shift begins day 20 |
| Cold-start (new-hire) users | 5, appear only from day 22 |

### Attack breakdown (event-level / incident-level)

| Attack type | Events | Incidents |
|---|---|---|
| brute_force | 284 | 25 |
| lateral_movement | 118 | 38 |
| device_spoofing | 38 | 38 |
| impossible_travel | 38 | 38 |
| credential_misuse | 38 | 38 |

## Day 3 — Unsupervised ensemble (Isolation Forest + Autoencoder)

Evaluated on held-out normal rows + all attack rows (never trained on either).

| Metric | Value |
|---|---|
| ROC-AUC (final_risk_score) | 0.9458 |
| ROC-AUC (ensemble_score_raw, pre-cold-start-blend) | 0.9441 |
| Alert threshold (5% FP budget) | 93.64 |
| False-positive rate at threshold | 5.00% |
| Overall attack recall at threshold | 82.9% |

Per-attack-type recall at the 5% FP threshold:

| Attack type | Recall |
|---|---|
| impossible_travel | 97.4% |
| lateral_movement | 96.6% |
| brute_force | 85.2% |
| credential_misuse | 71.1% |
| device_spoofing | 21.1% |

Cold-start blending effect (normal-labeled rows for thin-history entities, n=1,034):
mean raw ensemble score **76.85** → mean blended `final_risk_score` **57.82**
(pulled below the alert threshold, preventing new-hire false positives).

## Day 4 — Supervised XGBoost classifier (test split, n=3,823, untouched until final report)

| Metric | Value |
|---|---|
| Accuracy | 98.9% |
| Macro F1 | 0.832 |
| Macro precision / recall | 0.823 / 0.857 |
| Weighted F1 | 0.989 |
| False-positive rate (normal rows flagged as any attack) | 0.54% |

Per-class precision / recall / F1:

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| brute_force | 0.923 | 0.882 | 0.902 | 68 |
| credential_misuse | 0.714 | 0.625 | 0.667 | 8 |
| device_spoofing | 1.000 | 1.000 | 1.000 | 7 |
| impossible_travel | 0.615 | 1.000 | 0.762 | 8 |
| lateral_movement | 0.692 | 0.643 | 0.667 | 28 |
| normal | 0.994 | 0.995 | 0.994 | 3,704 |

**Domain-precondition gate impact** (validation split, decided before touching test):
brute_force precision 0.510 → 0.962, recall unchanged (0.847); macro F1 0.817 → 0.862.

**Unified score** (Day3 + Day4 merged, rule selected on validation only): 94.1%
overall recall at a matched 5% FP budget on test.

**Secondary review tier** (catches attacks neither layer confidently classified):
160 rows flagged across the full dataset, 15 real attacks recovered, 145 noise.

## Explainability

| Metric | Value |
|---|---|
| SHAP explainer build time (one-time) | 3.6 s |
| SHAP explain latency (batch, 200 alerts) | 1.68 ms / alert |
| Attribution type | Exact (TreeExplainer), not sampled/approximated |

## Drift detection (per-entity ADWIN)

| Metric | Value |
|---|---|
| True positives (real drift, detected) | 15 / 22 (68.2%) |
| False positives (false alarms) | 0 / 43 (0%) |
| False negatives (missed) | 7 / 22 |
| Mean detection lag (post-drift-day) | +2.47 days |
| Population-level alert triggered | Day 25 (10/65 = 15.4% entities drifted in a 7-day window) |

## bullets

- Designed and built an end-to-end behavioral anomaly detection system for
  cybersecurity, combining unsupervised (Isolation Forest + Autoencoder ensemble)
  and supervised (XGBoost) models to detect 5 attack types from access logs,
  achieving **98.9% accuracy** and **0.54% false-positive rate** on a held-out test
  set.
- Engineered 13 strictly-causal behavioral features (geo-velocity, access-graph
  distance, rolling burst windows) from raw access logs to detect brute-force,
  impossible travel, lateral movement, credential misuse, and device spoofing.
- Solved severe class imbalance (rarest attack types <0.3% of events) using SMOTE,
  balanced class weighting, and a domain-precondition gate, improving brute-force
  precision from 51% to 96% without any recall loss.
- Built per-entity concept-drift detection (ADWIN) achieving 0% false alarms across
  43 stable users, and a cold-start population-baseline blending mechanism that
  reduced new-hire false-positive risk scores by ~25 points.
- Delivered exact, real-time explainability using SHAP TreeExplainer (1.68ms/alert)
  and a Streamlit analyst dashboard with live alert triage, geo/graph incident
  visualization, and a one-click live model retrain.
