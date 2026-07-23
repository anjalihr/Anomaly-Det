
"""
explain.py
==========
Day 5 — Per-alert explainability via SHAP TreeExplainer.
 
WHY THIS MATTERS FOR THIS PROBLEM SPECIFICALLY (not generic ML polish)
--------------------------------------------------------------------------
This is a system that can flag a real employee for investigation. "The
model said 94% confidence" is not an answer an analyst - or the flagged
employee - can act on or contest. TreeExplainer gives an EXACT (not
sampled/approximated) per-feature attribution for XGBoost specifically,
fast enough to run inline on every alert (see benchmark below), which is
what makes this an accountability mechanism rather than a nice-to-have:
every flag ships with the specific, auditable reasons behind it.
 
WHY TreeExplainer, not KernelExplainer/generic SHAP
--------------------------------------------------------
KernelExplainer is model-agnostic but approximates via sampling - slow
(would blow the real-time latency budget from the Day 4/7 benchmark) and
noisy across repeated calls on the same row. TreeExplainer computes EXACT
Shapley values for tree ensembles in polynomial time by walking the
actual trees, because XGBoost is a tree model - so there's no accuracy
being traded away by picking the fast option here.
 
OUTPUT
-------
For a scored row, returns the top-K features (signed SHAP value = how
much that feature pushed the prediction toward the PREDICTED class,
positive = toward, negative = away) rendered as one plain-language
sentence each, e.g.:
    "5 failed login attempts in the last 5 minutes (strongly increased
     suspicion of brute_force)"
 
Run:
    python3 explain.py          # demo: explains a handful of real alerts
"""
 
import json
import os
import time
 
import joblib
import numpy as np
import pandas as pd
import shap
 
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
 
# ---------------------------------------------------------------------------
# Plain-language templates. Each returns a sentence describing WHAT the
# feature's actual value means, independent of direction - direction
# ("increased"/"reduced" suspicion) is appended separately from the SHAP
# sign, so the sentence stays accurate regardless of which way it pushed.
# ---------------------------------------------------------------------------
def _fmt_hour_zscore(v):
    return f"login timing is {v:.1f} standard deviations from this user's typical hours"
 
def _fmt_geo_distance_km(v):
    return f"{v:,.0f} km from the previous login's location"
 
def _fmt_geo_velocity_kmh(v):
    return f"implied travel speed of {v:,.0f} km/h between consecutive logins"
 
def _fmt_time_since_last_event_min(v):
    return f"only {v:.1f} minutes since this user's last activity"
 
def _fmt_is_new_device(v):
    return "device never seen for this user before" if v >= 1 else "device this user has used before"
 
def _fmt_fingerprint_mismatch(v):
    return "device's OS/browser/screen fingerprint doesn't match its known history" if v >= 1 \
        else "device fingerprint matches its known history"
 
def _fmt_is_new_host(v):
    return "host never accessed by this user before" if v >= 1 else "host this user has accessed before"
 
def _fmt_is_foreign_dept(v):
    return "host belongs to a department outside the user's home department" if v >= 1 \
        else "host is within the user's home department (or shared infrastructure)"
 
def _fmt_is_first_time_in_this_dept(v):
    return "user's first-ever access to this specific department" if v >= 1 \
        else "user has accessed this department before"
 
def _fmt_foreign_dept_burst_distinct(v):
    return f"{int(v)} distinct foreign departments touched in the last 15 minutes"
 
def _fmt_new_host_burst_count(v):
    return f"{int(v)} never-before-seen hosts touched in the last 15 minutes"
 
def _fmt_graph_dist_from_history(v):
    return f"this host is {int(v)} network hop(s) from anywhere this user has been before" if v > 0 \
        else "this host is directly adjacent to (or is) somewhere this user has been before"
 
def _fmt_failure_burst_count(v):
    return f"{int(v)} failed login attempts in the last 5 minutes"
 
FEATURE_TEMPLATES = {
    "hour_zscore": _fmt_hour_zscore,
    "geo_distance_km": _fmt_geo_distance_km,
    "geo_velocity_kmh": _fmt_geo_velocity_kmh,
    "time_since_last_event_min": _fmt_time_since_last_event_min,
    "is_new_device": _fmt_is_new_device,
    "fingerprint_mismatch": _fmt_fingerprint_mismatch,
    "is_new_host": _fmt_is_new_host,
    "is_foreign_dept": _fmt_is_foreign_dept,
    "is_first_time_in_this_dept": _fmt_is_first_time_in_this_dept,
    "foreign_dept_burst_distinct": _fmt_foreign_dept_burst_distinct,
    "new_host_burst_count": _fmt_new_host_burst_count,
    "graph_dist_from_history": _fmt_graph_dist_from_history,
    "failure_burst_count": _fmt_failure_burst_count,
}
 
 
def _direction_phrase(shap_val: float) -> str:
    mag = abs(shap_val)
    strength = "strongly" if mag > 1.0 else "moderately" if mag > 0.3 else "slightly"
    return f"{strength} increased suspicion" if shap_val > 0 else f"{strength} reduced suspicion"
 
 
class AlertExplainer:
    """
    Wraps a trained XGBoost classifier + SHAP TreeExplainer. Built once
    (model load + explainer construction is the expensive part), reused
    per-alert (see benchmark in main() - per-alert cost after warmup).
    """
 
    def __init__(self, model=None, label_encoder=None, feature_cols=None):
        if model is None:
            model = joblib.load(os.path.join(MODELS_DIR, "xgb_classifier.joblib"))
        if label_encoder is None:
            label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
        if feature_cols is None:
            with open(os.path.join(MODELS_DIR, "classifier_feature_cols.json")) as f:
                feature_cols = json.load(f)["feature_cols"]
        self.model = model
        self.label_encoder = label_encoder
        self.feature_cols = feature_cols
        self.classes = list(label_encoder.classes_)
        self.explainer = shap.TreeExplainer(model)
 
    def explain_row(self, feature_row: np.ndarray, predicted_label: str, top_k: int = 3) -> dict:
        """
        feature_row: 1D array of length len(self.feature_cols), in that order.
        Returns {"predicted_label", "top_factors": [ {feature, value, shap_value, sentence}, ... ]}
        """
        X = feature_row.reshape(1, -1)
        shap_values = self.explainer.shap_values(X)  # shape (1, n_features, n_classes)
        class_idx = self.classes.index(predicted_label)
        sv_for_class = shap_values[0, :, class_idx]
 
        order = np.argsort(-np.abs(sv_for_class))[:top_k]
        factors = []
        for i in order:
            fname = self.feature_cols[i]
            fval = feature_row[i]
            sval = float(sv_for_class[i])
            what = FEATURE_TEMPLATES.get(fname, lambda v: f"{fname}={v}")(fval)
            factors.append({
                "feature": fname,
                "value": float(fval),
                "shap_value": round(sval, 4),
                "sentence": f"{what} ({_direction_phrase(sval)} of {predicted_label})",
            })
        return {"predicted_label": predicted_label, "top_factors": factors}
 
    def explain_batch(self, X: np.ndarray, predicted_labels: np.ndarray, top_k: int = 3) -> list:
        """Vectorized SHAP call for many rows at once (much cheaper per-row than
        calling explain_row in a loop - use this for dashboard batch rendering)."""
        shap_values = self.explainer.shap_values(X)  # (n, n_features, n_classes)
        out = []
        for i in range(X.shape[0]):
            class_idx = self.classes.index(predicted_labels[i])
            sv = shap_values[i, :, class_idx]
            order = np.argsort(-np.abs(sv))[:top_k]
            factors = []
            for j in order:
                fname = self.feature_cols[j]
                fval = X[i, j]
                sval = float(sv[j])
                what = FEATURE_TEMPLATES.get(fname, lambda v: f"{fname}={v}")(fval)
                factors.append({
                    "feature": fname, "value": float(fval), "shap_value": round(sval, 4),
                    "sentence": f"{what} ({_direction_phrase(sval)} of {predicted_labels[i]})",
                })
            out.append({"predicted_label": predicted_labels[i], "top_factors": factors})
        return out
 
 
def main():
    print("Loading model + building SHAP TreeExplainer...")
    t0 = time.time()
    ax = AlertExplainer()
    build_time = time.time() - t0
    print(f"  Explainer ready in {build_time*1000:.0f} ms (one-time cost)")
 
    df = pd.read_csv(f"{DATA_DIR}/day4_results.csv", low_memory=False)
    test = df[df["clf_split"] == "test"]
 
    demo_rows = []
    for atype in ["brute_force", "device_spoofing", "impossible_travel", "lateral_movement", "credential_misuse"]:
        hit = test[(test["attack_type"] == atype) & (test["predicted_attack_type"] == atype)]
        if len(hit):
            demo_rows.append(hit.iloc[0])
 
    print(f"\nExplaining {len(demo_rows)} real caught alerts (one per attack type):\n")
    for row in demo_rows:
        feat_row = row[ax.feature_cols].astype(float).fillna(0).to_numpy(dtype=float)
        result = ax.explain_row(feat_row, row["predicted_attack_type"])
        print(f"--- ALERT: {row['event_id'][:8]}...  predicted={result['predicted_label']}  "
              f"(true={row['attack_type']}, confidence={row['classifier_confidence']:.3f}) ---")
        for f in result["top_factors"]:
            print(f"    - {f['sentence']}")
        print()
 
    # --- latency benchmark (per-alert cost, the number that matters for the dashboard) ---
    sample = test.sample(min(200, len(test)), random_state=1)
    X = sample[ax.feature_cols].astype(float).fillna(0).to_numpy(dtype=float)
    labels = sample["predicted_attack_type"].fillna("normal").to_numpy()
    t0 = time.time()
    _ = ax.explain_batch(X, labels)
    elapsed = time.time() - t0
    print(f"Batch explain latency: {len(sample)} alerts in {elapsed*1000:.1f} ms "
          f"=> {elapsed/len(sample)*1000:.3f} ms/alert (after one-time {build_time*1000:.0f} ms explainer build)")
 
 
if __name__ == "__main__":
    main()
 
