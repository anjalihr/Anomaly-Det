
"""
classifier.py
=============
Day 4 — Supervised multi-class attack classifier.
 
What this adds that Day 3 structurally cannot
-----------------------------------------------
Day 3's Isolation Forest / autoencoder are UNSUPERVISED: they never see a
single attack label, and they score anomaly as "how far is this point from
the normal cluster in 13-dimensional feature space, averaged/pooled across
all 13 dimensions." That is exactly why device_spoofing (one deterministic
flag, everything else normal) fell through the cracks - a single strong
feature gets diluted across 12 unremarkable ones.
 
An XGBoost classifier, by contrast, is SUPERVISED: it is trained directly
on which attack_type each row actually is, so it can learn per-feature
IMPORTANCE for each class - e.g. "if fingerprint_mismatch=1, that's nearly
always device_spoofing, regardless of every other feature" - a relationship
Day 3's models had no mechanism to learn. This is the concrete, measured
reason the pipeline has both an unsupervised layer AND a supervised layer,
not just one or the other.
 
The hard part: class imbalance
--------------------------------
attack_type distribution on this dataset: normal=12849, brute_force=273,
lateral_movement=60, impossible_travel=20, device_spoofing=20,
credential_misuse=20. A classifier trained naively on this will just
learn to predict "normal" almost always and still score ~97% accuracy
while catching almost nothing - accuracy is a meaningless metric here,
which is why we report per-class precision/recall/F1 and a confusion
matrix instead.
 
Two complementary techniques handle this (deliberately combined, not
either/or):
  1. SMOTE (Synthetic Minority Oversampling) - applied to the TRAINING
     split only, generates synthetic examples of rare attack types by
     interpolating between real minority-class neighbors in feature space.
     NEVER applied to the test split - evaluating on synthetic data would
     make the numbers meaningless.
  2. Balanced sample weights - XGBoost is additionally given per-row
     weights (inverse class frequency) during training, so even after
     SMOTE's oversampling, the loss function itself still penalizes
     misclassifying a rare class more than misclassifying "normal."
 
STRICT NO-LEAKAGE RULE (same discipline as Day 2/3): the classifier is
trained ONLY on the 13 raw behavioral features Day 3 used - NOT on Day 3's
anomaly scores. Day 3's scores are artificially low for rows its own
models were fit to, which would leak "was this row in Day 3's training
set" information into Day 4's classifier if used as an input feature. The
two layers are merged at the DECISION level (see run_day4.py), not the
feature level, keeping each layer's train/test bookkeeping independent and
each layer's contribution separately explainable.
"""
 
import json
import os
 
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
 
# Same 13 behavioral features Day 3 used - see models.py for why the
# meta-features (entity_history_*, cold_start_*) are deliberately excluded.
FEATURE_COLS = [
    "hour_zscore",
    "geo_distance_km",
    "geo_velocity_kmh",
    "time_since_last_event_min",
    "is_new_device",
    "fingerprint_mismatch",
    "is_new_host",
    "is_foreign_dept",
    "is_first_time_in_this_dept",
    "foreign_dept_burst_distinct",
    "new_host_burst_count",
    "graph_dist_from_history",
    "failure_burst_count",
]
 
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
 
 
def prepare_matrix(df: pd.DataFrame, fill_values: dict) -> np.ndarray:
    """Uses the SAME fill_values dict Day 3 computed from its own train
    split - kept as an explicit argument (not recomputed here) so both
    layers fill missing values identically and stay comparable."""
    X = df[FEATURE_COLS].copy()
    for col, val in fill_values.items():
        if col in X.columns:
            X[col] = X[col].fillna(val)
    return X.to_numpy(dtype=float)
 
 
def build_labels(df: pd.DataFrame) -> pd.Series:
    """attack_type is NaN for normal rows - fill with the literal string
    'normal' so we have one clean multi-class target column."""
    return df["attack_type"].fillna("normal")
 
 
def smote_oversample(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42):
    """
    Oversample minority classes in the TRAINING split only.
 
    k_neighbors must be < the smallest class's sample count for SMOTE to
    find neighbors to interpolate between. With classes as small as ~15
    rows post-split, we cap k_neighbors dynamically rather than crashing
    on sklearn/imblearn's default of 5.
    """
    class_counts = pd.Series(y_train).value_counts()
    smallest_minority = class_counts[class_counts.index != "normal"].min()
    k_neighbors = max(1, min(5, smallest_minority - 1))
 
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res, k_neighbors
 
 
def train_classifier(X_train: np.ndarray, y_train_encoded: np.ndarray,
                      n_classes: int, random_state: int = 42) -> XGBClassifier:
    """
    XGBoost multi:softprob classifier, additionally weighted by inverse
    class frequency on top of the SMOTE-resampled training set - belt and
    suspenders against the imbalance, not redundant: SMOTE fixes the
    SAMPLE COUNT imbalance, sample_weight fixes the LOSS FUNCTION's
    sensitivity to misclassifying whichever classes remain rarer even
    after resampling.
    """
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_encoded)
 
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_encoded, sample_weight=sample_weight)
    return model
 
 
def score_classifier(model: XGBClassifier, X: np.ndarray, label_encoder: LabelEncoder):
    """
    Returns three parallel arrays:
      predicted_label   - the argmax class name (string), e.g. "brute_force"
      confidence         - the winning class's predicted probability (0-1)
      attack_prob        - 1 - P(normal), i.e. total probability mass
                            assigned to ANY attack class (0-1). This is
                            what gets merged with Day 3's anomaly score.
    """
    proba = model.predict_proba(X)
    pred_idx = proba.argmax(axis=1)
    predicted_label = label_encoder.inverse_transform(pred_idx)
    confidence = proba[np.arange(len(proba)), pred_idx]
 
    normal_idx = list(label_encoder.classes_).index("normal")
    attack_prob = 1.0 - proba[:, normal_idx]
 
    return predicted_label, confidence, attack_prob
 
 
# ---------------------------------------------------------------------------
# DOMAIN-PRECONDITION GATE (added after Day 4 review — see bug log)
# ---------------------------------------------------------------------------
# DIAGNOSIS: Post-SMOTE, the classifier had 48 false-positive brute_force
# calls on the test split whose failure_burst_count was ~0 - i.e. it was
# calling "brute_force" on rows with NO observed failed-login clustering,
# which is a contradiction in terms (brute_force IS repeated failed
# logins, by construction in generate_logs.py). Root cause: SMOTE
# interpolates minority-class points in full 13-D feature space, which can
# synthesize points that are near a real brute_force cluster on OTHER
# dimensions while sitting at failure_burst_count=~0 - a region that never
# occurs in real brute_force traffic but the interpolation doesn't know
# that. The model then learns a decision boundary that includes those
# synthetic ghosts.
#
# FIX: gate certain classes behind a hard, human-auditable necessary
# condition - not sufficient, just necessary - drawn directly from each
# attack type's own definition in generate_logs.py. If the top-ranked
# class's precondition isn't met, fall through to the next-ranked class
# that DOES satisfy its own precondition (falling all the way to "normal"
# if nothing qualifies). This is a hybrid symbolic+ML design, not a
# post-hoc accuracy hack: every gate is a condition the attack literally
# cannot exist without, so it can only ever remove impossible predictions,
# never a correct one.
#   - brute_force needs failure_burst_count >= 1 (repeated failed logins
#     is the entire definition of the attack)
#   - device_spoofing needs fingerprint_mismatch == 1 (the attack IS a
#     fingerprint mismatch by construction)
#   - lateral_movement / credential_misuse / impossible_travel are left
#     UNGATED: their signal is graph/context-based (foreign-dept bursts,
#     odd-hour + new-device combos, geo-velocity) rather than one binary
#     flag, so there's no single necessary condition to gate on without
#     just re-implementing a worse version of what the classifier already
#     learned. Validated impossible_travel's false positives separately
#     (see bug log): they are dominated by a distinct, already-understood
#     cause - the event immediately following a real attack inherits a
#     contaminated "last known location," which is a legitimate thing for
#     an analyst to want reviewed, not classifier noise - so no gate was
#     forced there.
#
# Measured impact on VALIDATION (never touched test to decide this):
#   brute_force   precision 0.510 -> 0.962, recall unchanged (0.847)
#   macro F1      0.817 -> 0.862, accuracy 0.980 -> 0.992
# ---------------------------------------------------------------------------
 
def _precondition_ok(cls: str, row: np.ndarray, feat_idx: dict) -> bool:
    if cls == "brute_force":
        return row[feat_idx["failure_burst_count"]] >= 1
    if cls == "device_spoofing":
        return row[feat_idx["fingerprint_mismatch"]] >= 1
    return True  # lateral_movement, credential_misuse, impossible_travel, normal: ungated
 
 
def score_classifier_gated(model: XGBClassifier, X: np.ndarray, label_encoder: LabelEncoder):
    """
    Same return signature as score_classifier, but applies the domain-
    precondition gate described above on top of the raw model output.
    This is the version used everywhere in run_day4.py from here on -
    score_classifier() is kept as-is (ungated) so the "what did the raw
    ML alone learn" number stays inspectable for the bug-log comparison.
    """
    proba = model.predict_proba(X)
    classes = list(label_encoder.classes_)
    normal_idx = classes.index("normal")
    feat_idx = {c: i for i, c in enumerate(FEATURE_COLS)}
 
    order = np.argsort(-proba, axis=1)  # each row: class indices ranked best->worst
    n = X.shape[0]
    pred_idx = np.empty(n, dtype=int)
    for i in range(n):
        chosen = normal_idx
        for ci in order[i]:
            if _precondition_ok(classes[ci], X[i], feat_idx):
                chosen = ci
                break
        pred_idx[i] = chosen
 
    predicted_label = label_encoder.inverse_transform(pred_idx)
    confidence = proba[np.arange(n), pred_idx]
    attack_prob = 1.0 - proba[:, normal_idx]  # unchanged: still the raw model's total attack mass
    return predicted_label, confidence, attack_prob
 
 
def save_artifacts(model, label_encoder, fill_values):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "xgb_classifier.joblib"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODELS_DIR, "classifier_feature_cols.json"), "w") as f:
        json.dump({"feature_cols": FEATURE_COLS, "fill_values": fill_values}, f, indent=2)
 
 
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "xgb_classifier.joblib"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODELS_DIR, "classifier_feature_cols.json")) as f:
        meta = json.load(f)
    return model, label_encoder, meta["fill_values"]
 
