
"""
coldstart.py
============
Day 3 — Cold-start handling.
 
Problem this solves
--------------------
Isolation Forest and the autoencoder score events purely on THIS event's
feature values. For a brand-new hire (or any entity with thin history),
several features are structurally "unusual" simply because there is no
personal history yet to compare against - e.g. every host they touch is
technically their first, hour_zscore is being computed against a
population prior rather than a personal one, etc. Left alone, this makes
new/low-history entities look artificially risky and would flood a real
analyst dashboard with false positives on day one of someone's job -
exactly the "cold-start" failure mode described in the problem statement.
 
Fix: population-baseline fallback
-----------------------------------
features.py already tracks, per row, `cold_start_blend_weight` - a 0..1
value (0 = no personal history at all, 1 = fully established, >=
MIN_HISTORY_EVENTS of personal history). We reuse that same signal here,
one layer up, to calibrate the MODEL'S risk score rather than just the
input features:
 
    final_risk_score = w * raw_ensemble_score
                        + (1 - w) * dept_population_reference_score
 
where `dept_population_reference_score` is the median ensemble anomaly
score observed for ESTABLISHED (non-cold-start), NORMAL-labeled entities
in the same department. In plain language: "if we barely know this
person yet, don't fully trust a high anomaly score from the model - pull
it toward what's typical for their department until they've built up
enough history for the model to trust their personal baseline."
 
This is a deliberate design choice, not a way to silently suppress real
attacks: cold-start entities are, by construction in generate_logs.py,
never used as attack victims (`eligible_victims` excludes them), so this
blending only ever protects genuinely new/thin-history normal behavior,
and the blend weight moves toward 1.0 (full trust in the raw score)
automatically as an entity accumulates real history - no manual retraining
needed.
"""
 
import numpy as np
import pandas as pd
 
GLOBAL_FALLBACK_KEY = "__global__"
 
 
def compute_population_reference(scored_normal_established: pd.DataFrame,
                                   score_col: str = "ensemble_score_raw",
                                   dept_col: str = "dept") -> dict:
    """
    Build a dept -> median-anomaly-score lookup, computed ONLY from
    NORMAL-labeled rows belonging to ESTABLISHED (non-cold-start) entities.
    This is exactly the "pre-existing organizational knowledge" baseline
    features.py's population hour-priors already use the same philosophy
    for - typical behavior for a department, not leaked from any one
    individual's future.
    """
    ref = scored_normal_established.groupby(dept_col)[score_col].median().to_dict()
    ref[GLOBAL_FALLBACK_KEY] = float(scored_normal_established[score_col].median())
    return {k: float(v) for k, v in ref.items()}
 
 
def blend_cold_start_scores(df: pd.DataFrame,
                             population_reference: dict,
                             raw_score_col: str = "ensemble_score_raw",
                             weight_col: str = "cold_start_blend_weight",
                             dept_col: str = "dept") -> pd.Series:
    """
    Returns the calibrated `final_risk_score` column (0-100), blending each
    row's raw ensemble score toward its department's population-normal
    median by (1 - cold_start_blend_weight).
    """
    dept_ref = df[dept_col].map(population_reference).fillna(population_reference[GLOBAL_FALLBACK_KEY])
    w = df[weight_col].clip(0, 1)
    blended = w * df[raw_score_col] + (1 - w) * dept_ref
    return blended.clip(0, 100).round(2)
 
