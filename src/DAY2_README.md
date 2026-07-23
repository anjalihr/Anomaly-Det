# Day 2 — Feature Engineering (DONE)

## What was built
`src/features.py` — converts raw `synthetic_logs.csv` into `data/features.csv`,
26 columns (16 original + ~15 engineered features), one row per event.

## The core rule enforced throughout
**Strict causality**: every feature for an event is computed using only that
entity's history *before* that event's timestamp. Verified explicitly (see
validation below) — every user's very first-ever event shows zero history and
is flagged `is_cold_start_entity=1`, exactly as it should be in a real
real-time system that has no future information.

## Bug found and fixed during validation
`is_foreign_dept` initially always returned 0, even for lateral_movement
attacks. Root cause: the log's `dept` column (from Day 1) always stores the
*user's own* department, not the department that owns the host being
accessed — so it can never reveal a foreign-department access by
construction. Fixed by deriving the host's owning department from the
access-topology graph (`entities.json` -> `dept_hosts`) instead. This is
exactly the kind of thing worth mentioning if asked about your process —
we validated against ground truth and caught a real bug rather than
assuming the first version worked.

## Validation: feature-to-attack-type separation (mean values)

| attack_type | hour_zscore | geo_velocity_kmh | is_new_device | fingerprint_mismatch | is_new_host | is_foreign_dept | foreign_dept_burst | new_host_burst | failure_burst |
|---|---|---|---|---|---|---|---|---|---|
| brute_force | 1.79 | 903 | 0.00 | 0.0 | 0.01 | 0.0 | 0.00 | 0.12 | **7.95** |
| credential_misuse | **2.84** | 11 | **1.00** | 0.0 | 0.00 | 0.0 | 0.00 | 0.00 | 0.00 |
| device_spoofing | 1.00 | 52 | 0.00 | **0.9** | 0.00 | 0.0 | 0.00 | 0.00 | 0.00 |
| impossible_travel | 1.26 | **23,479** | 0.00 | 0.0 | 0.05 | 0.0 | 0.00 | 0.00 | 0.00 |
| lateral_movement | 1.14 | 420 | 0.00 | 0.0 | **1.00** | **1.0** | **1.58** | **2.33** | 0.00 |
| normal (NaN) | 0.88 | 126 | 0.01 | 0.0 | 0.03 | 0.0 | 0.00 | 0.01 | 0.00 |

Every attack type lights up strongly on its *intended* feature(s) and stays
near-baseline on the others — this is what makes the Day 3-4 models' job
tractable and interpretable, rather than throwing raw logs at a black box.

## Why this design (recap for the pitch)
- **Per-entity, not global** baselines throughout — the reason false
  positives stay low is that "normal" is defined relative to each user's
  own behavior, not a single company-wide rule.
- **Cold-start blending**: personal and population (department-level)
  baselines are combined with a weight proportional to how much history
  the entity actually has — a brand-new user isn't judged against an
  unreliable 2-event personal average, nor flagged as 100% anomalous by
  default either.
- **Graph-based lateral movement**, not a host-count threshold — uses the
  Day 1 access-topology graph to detect crossing into departments/clusters
  the entity has never touched, which is what keeps legitimately
  broad-access roles (e.g. IT admins) from triggering false alarms.

## Next (Day 3)
`src/train_models.py` — Isolation Forest + Autoencoder ensemble trained on
`features.csv`, giving every event a real-time anomaly score without
needing any attack labels (handles cold-start and novel attacks by design).
