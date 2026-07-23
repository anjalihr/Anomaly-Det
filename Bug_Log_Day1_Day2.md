**Behavioral Anomaly Detection System**

Bug & Issue Log --- Day 1 (Data Generation) & Day 2 (Feature
Engineering)

Purpose of This Log

This document records every issue found and fixed during development,
along with how each was caught. It exists to make the build process
auditable and defensible under questioning --- rather than presenting
the system as if it worked perfectly on the first attempt, this log
shows the actual validate-then-fix process used at each stage.

Day 1 --- Synthetic Log Generation

Outcome

No functional bugs were found in the generator itself. Every validation
check (schema correctness, cold-start timing, drift-group city/hour
shift, and structural correctness of all 5 attack types) passed on the
first full run. This was verified explicitly --- not assumed --- by
inspecting sample rows for each attack type and cross-checking
cold-start/drift users against the exact days configured in the script.

Process Issue Handled (not a code bug)

An earlier, less complete draft of the generator already existed in the
working environment (different schema, no access-topology graph, weaker
separation between cold-start and drift logic). Rather than edit it
incrementally, it was archived and replaced with the finalized version
to avoid a schema mismatch propagating silently into Day 2's feature
engineering. This is a design decision worth mentioning if asked about
iteration --- the first attempt was not treated as final just because it
ran without errors.

Validation Checks Performed

- Confirmed dataset shape and column schema matched design (13,346 rows,
  16 columns).

- Confirmed attack ratio (4.08%) and per-attack-type event counts were
  realistic and imbalanced.

- Spot-checked impossible_travel rows: verified two logins from
  geographically distant cities within a short time gap.

- Spot-checked lateral_movement rows: verified hosts touched belonged to
  departments other than the user\'s home department.

- Spot-checked credential_misuse rows: verified odd-hour timestamp +
  unrecognized device_id.

- Spot-checked device_spoofing rows: verified a known device_id paired
  with a mismatched fingerprint.

- Confirmed cold-start users (user_045--user_049) had zero events before
  day 22, exactly as configured.

- Confirmed drift-group users showed a city/working-hour shift starting
  exactly on day 20 (verified via weekly city breakdown).

Day 2 --- Feature Engineering

Bug Found: is_foreign_dept always returned 0

  ----------- ----------------------------- ----------------------- ------------
  **Field**   **Root Cause**                **Fix**                 **Status**

  Symptom     The lateral_movement attack   N/A --- caught via      Fixed &
              type showed is_foreign_dept = mandatory               Verified
              0.0 and                       feature-separation      
              foreign_dept_burst_distinct = validation step         
              0.0 in the per-attack-type    (comparing mean feature 
              feature summary --- despite   values grouped by       
              lateral_movement being        attack_type) before     
              defined specifically as       proceeding to model     
              accessing hosts in foreign    training.               
              departments.                                          
  ----------- ----------------------------- ----------------------- ------------

Root Cause

The synthetic log\'s \`dept\` column (set in Day 1\'s generator) always
stores the USER\'s own home department --- it is copied directly from
the user\'s profile at event creation time and never changes, regardless
of which host the event actually accesses. The Day 2 feature code was
comparing this column against itself, which by construction can never
produce a mismatch, so the "foreign department" signal was silently dead
for every single event, including genuine lateral-movement attacks.

Fix

Replaced the comparison with the HOST\'s actual owning department,
derived from the access-topology graph built in Day 1 (entities.json →
dept_hosts mapping), instead of the log\'s static per-user dept column.
is_foreign_dept is now: (host\'s owning department ≠ user\'s home
department) AND (host\'s department ≠ 'shared'). The rolling
burst-window tracker (foreign_dept_burst_distinct) and the
department-novelty tracker (user_depts_seen) were both updated to key
off the corrected host department as well, since they depended on the
same broken value.

Verification After Fix

- Re-ran the per-attack-type feature summary: is_foreign_dept jumped
  from 0.0 to 1.0 for lateral_movement, and stayed at 0.0 for every
  other category (normal traffic included).

- foreign_dept_burst_distinct rose from 0.0 to 1.58 for lateral_movement
  specifically, confirming the rolling-window burst logic was also
  repaired, not just the single-event flag.

- Re-confirmed no other attack type was affected by the fix (all other
  feature means were unchanged before/after).

Secondary Checks Performed (no bugs found, but explicitly verified)

- Causality / no future leakage: confirmed every user\'s first-ever
  event has entity_history_event_count = 0 and is_cold_start_entity = 1,
  proving no feature used data from before an entity\'s own first
  appearance.

- NaN audit: confirmed the only NaN values in the output are
  geo/time-gap features on each user\'s very first event (expected ---
  there is no "previous event" to compare against) and metadata columns
  (attack_type/attack_id) on normal rows.

- Circular-hour handling: verified the hour-of-day distance function
  correctly treats 23:00 and 00:00 as close together rather than 23
  hours apart, avoiding false positives for users who habitually work
  past midnight.

Why This Log Matters for the Pitch

Rather than presenting a system that "just worked," this log
demonstrates an actual engineering process: build → validate against
ground truth → catch a real, non-obvious bug → fix it → re-verify. The
is_foreign_dept bug in particular is a good example of why explicit
validation against labeled ground truth matters --- the code ran without
throwing any errors and produced a plausible-looking column; only a
deliberate feature-separation check against the known attack labels
revealed that the signal was silently useless.
