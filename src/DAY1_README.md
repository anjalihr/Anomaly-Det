# Day 1 — Synthetic Data Generation (DONE)

## What was built
`src/generate_logs.py` — generates the full synthetic access-log dataset plus
ground-truth metadata needed by every later stage.

## Outputs
- `data/synthetic_logs.csv` — 13,346 events, 16 columns
- `data/entities.json` — user profiles, device fingerprints, access-topology graph edges, config (drift day, cold-start day)
- `data/attack_sessions.json` — ground truth of every injected attack session (for scoring detection accuracy later)

## Dataset stats (this run, seed=42)
- 50 users, 5 departments, 90 hosts (16/dept + 10 shared), organized as a real graph (not a flat list)
- 30 simulated days
- 12,801 normal events / 545 attack events → **4.08% attack ratio** (event-level).
  At the *attack-type* level the imbalance is much sharper (brute_force: 365 events
  vs credential_misuse/device_spoofing/impossible_travel: 20 each) — this is the
  imbalance the Day 4 classifier has to handle with SMOTE + class weights.
- 105 distinct attack sessions across 5 types, all verified structurally correct:
  - **brute_force**: 6-25 rapid failed logins from one source IP, occasional success
  - **impossible_travel**: geographically distant logins (verified: normal login → later login from a far city within 5-90 min, physically infeasible)
  - **lateral_movement**: verified hosts touched are outside the user's home department (e.g. an HR user hitting finance-host, it_admin-host, sales-host in rapid succession)
  - **credential_misuse**: odd-hour login from an unrecognized device_id
  - **device_spoofing**: known device_id but fingerprint (OS/browser/screen) mismatches what that device has ever presented before
- **Cold-start** simulated: 5 users (`user_045`-`user_049`) have zero history until day 22 — verified, first event exactly on schedule
- **Concept drift** simulated: 15/50 users shift home city + working hours starting day 20 (policy change) — verified via weekly city breakdown for a sample user

## Why this design (recap for the pitch)
- The access-topology **graph** (not a flat host list) is what makes lateral
  movement detection non-trivial and defensible later — Day 2's feature
  engineering will compute "is this an edge/cluster the user has ever
  traversed before" rather than a naive "host count > threshold" rule.
- Cold-start and drift are **structurally baked into the data**, not
  hand-waved — we can point at exact user IDs and exact days in the live
  demo to prove the system handles both.
- Attack ratio and per-type imbalance are realistic enough to make SMOTE/
  class-weighting a genuine necessity, not decoration.

## Next (Day 2)
`src/features.py` — per-entity rolling baselines, geo-velocity for impossible
travel, graph-traversal features for lateral movement, device-fingerprint
consistency scores.
