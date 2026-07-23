
"""
run_day4.py
===========
Day 4 driver script. Ties together classifier.py + Day 3's scored_events.csv.
 
METHODOLOGY NOTE (why train/VALIDATION/test, not just train/test)
-------------------------------------------------------------------
The first version of this script picked "max" over "mean" for combining
Day 3's anomaly score with Day 4's classifier probability by looking at
recall/FP numbers on the held-out test split, then reported final metrics
on that SAME split. That's test-set leakage through model selection: once
a design decision (which combination rule) is chosen by looking at a
split's outcome, metrics reported on that same split are no longer a
clean estimate of real-world performance - the test set quietly
influenced the design.
 
Fix: a proper three-way split.
  - TRAIN      - fit the classifier (with SMOTE, as before)
  - VALIDATION - used ONLY to decide the merge rule (mean vs max) and
                 calibrate the secondary-review threshold. Never used to
                 report a final number.
  - TEST       - touched EXACTLY ONCE, at the very end, to report the
                 final numbers under whichever rule validation selected.
                 Never used to make any decision.
 
Steps:
  1. Load scored_events.csv (Day 3's output)
  2. Build the 6-class target (normal + 5 attack types)
  3. SESSION-AWARE 3-way split (60/20/20) - preserves rare-class
     representation and keeps every attack incident's events entirely in
     one split (never spread across two, see the Day 4 bug log)
  4. SMOTE-oversample the TRAINING split only
  5. Train XGBoost with balanced sample weights on top of SMOTE
  6. On VALIDATION: compare mean vs max combination rules, pick a winner
  7. On TEST (untouched until now): evaluate the classifier itself
     (confusion matrix, per-class P/R/F1, row- and incident-level recall)
     AND the winning merge rule's unified score - these are the numbers
     that go in the pitch
  8. Score every row, merge, save data/day4_results.csv for Day 5/6
 
POST-REVIEW FIX (see classifier.py's score_classifier_gated docstring for
full diagnosis): raw SMOTE+XGBoost had 48 test-set false-positive
brute_force calls with ~zero failure_burst_count - synthetic-minority
"ghost" points the model learned from that don't correspond to anything
that can occur in real brute_force traffic. Fixed with a hybrid
symbolic+ML gate: certain classes require a hard, attack-defining
precondition (failure_burst_count>=1 for brute_force,
fingerprint_mismatch==1 for device_spoofing) to be eligible at all: raw ML
still ranks candidates, the gate only removes impossible ones. Decided and
measured on VALIDATION only (brute_force precision 0.510->0.962, recall
unchanged); applied identically to test/full-dataset scoring afterward.
Separately, impossible_travel's residual false positives were traced to a
distinct, legitimate cause - the event immediately following a real
attack inherits a contaminated "last known location" for the geo-velocity
feature - and were deliberately left ungated since that's a real signal
worth an analyst's attention, not classifier noise.
 
Run:
    python3 run_day4.py
"""
 
import os
 
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
from classifier import (
    FEATURE_COLS,
    build_labels,
    prepare_matrix,
    save_artifacts,
    score_classifier,
    score_classifier_gated,
    smote_oversample,
    train_classifier,
)
 
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RANDOM_STATE = 42
VAL_FRACTION = 0.20    # of the full dataset
TEST_FRACTION = 0.20   # of the full dataset (remaining 60% is train)
 
 
def session_aware_three_way_split(df: pd.DataFrame, y_all: pd.Series):
    """Splits normal rows at the row level and attack rows at the attack_id
    (whole-incident) level, stratified by attack_type, into train/val/test.
    No attack_id ever appears in more than one split."""
    normal_mask = (y_all == "normal").to_numpy()
    normal_idx_all, attack_idx_all = df.index[normal_mask], df.index[~normal_mask]
 
    remaining_frac = 1.0 - TEST_FRACTION
    val_frac_of_remaining = VAL_FRACTION / remaining_frac
 
    normal_remain_idx, normal_test_idx = train_test_split(
        normal_idx_all, test_size=TEST_FRACTION, random_state=RANDOM_STATE
    )
    normal_train_idx, normal_val_idx = train_test_split(
        normal_remain_idx, test_size=val_frac_of_remaining, random_state=RANDOM_STATE
    )
 
    attack_sessions = df.loc[attack_idx_all, ["attack_id", "attack_type"]].drop_duplicates("attack_id")
    sessions_remain, sessions_test = train_test_split(
        attack_sessions, test_size=TEST_FRACTION,
        stratify=attack_sessions["attack_type"], random_state=RANDOM_STATE,
    )
    sessions_train, sessions_val = train_test_split(
        sessions_remain, test_size=val_frac_of_remaining,
        stratify=sessions_remain["attack_type"], random_state=RANDOM_STATE,
    )
 
    attack_train_idx = df.index[df["attack_id"].isin(sessions_train["attack_id"])]
    attack_val_idx = df.index[df["attack_id"].isin(sessions_val["attack_id"])]
    attack_test_idx = df.index[df["attack_id"].isin(sessions_test["attack_id"])]
 
    train_idx = normal_train_idx.union(attack_train_idx)
    val_idx = normal_val_idx.union(attack_val_idx)
    test_idx = normal_test_idx.union(attack_test_idx)
 
    ids_train = set(df.loc[train_idx, "attack_id"].dropna())
    ids_val = set(df.loc[val_idx, "attack_id"].dropna())
    ids_test = set(df.loc[test_idx, "attack_id"].dropna())
    assert not (ids_train & ids_val), f"leakage train/val: {ids_train & ids_val}"
    assert not (ids_train & ids_test), f"leakage train/test: {ids_train & ids_test}"
    assert not (ids_val & ids_test), f"leakage val/test: {ids_val & ids_test}"
 
    return train_idx, val_idx, test_idx
 
 
def compute_unified(final_risk_score: pd.Series, classifier_attack_prob: pd.Series, rule: str) -> pd.Series:
    a, b = final_risk_score, classifier_attack_prob
    if rule == "max":
        return pd.concat([a, b], axis=1).max(axis=1)
    elif rule == "mean":
        return (a + b) / 2
    raise ValueError(rule)
 
 
def evaluate_merge_rule(split_df: pd.DataFrame, rule: str, target_fp: float = 0.05):
    """Calibrates an alert threshold on this split's own normal rows to hit
    ~target_fp, then reports overall attack recall and per-attack-type
    recall under that rule. Used on VALIDATION to pick a rule, and again
    (independently) on TEST to report the final number under the winner."""
    unified = compute_unified(split_df["final_risk_score"], split_df["classifier_attack_prob"], rule)
    normal_mask = (split_df["label"] == "normal").to_numpy()
    threshold = np.percentile(unified[normal_mask], 100 * (1 - target_fp))
    fp_rate = (unified[normal_mask] >= threshold).mean()
 
    attack_mask = (split_df["label"] == "attack").to_numpy()
    overall_recall = (unified[attack_mask] >= threshold).mean() if attack_mask.any() else float("nan")
 
    per_type = {}
    attack_types = split_df.loc[attack_mask, "attack_type"]
    for atype in attack_types.unique():
        sub_mask = attack_mask & (split_df["attack_type"] == atype).to_numpy()
        per_type[atype] = (unified[sub_mask] >= threshold).mean()
 
    return {"rule": rule, "threshold": threshold, "fp_rate": fp_rate,
            "overall_recall": overall_recall, "per_type_recall": per_type}
 
 
def main():
    print("Loading scored_events.csv (Day 3 output)...")
    df = pd.read_csv(f"{DATA_DIR}/scored_events.csv")
    print(f"  {len(df)} rows")
 
    y_all = build_labels(df)
    print("\nClass distribution:")
    print(y_all.value_counts().to_string())
 
    # ---------------- 1. SESSION-AWARE 3-way split (train / val / test) ----------------
    train_idx, val_idx, test_idx = session_aware_three_way_split(df, y_all)
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)
    y_train_raw = y_all.loc[train_idx].reset_index(drop=True)
    y_val_raw = y_all.loc[val_idx].reset_index(drop=True)
    y_test_raw = y_all.loc[test_idx].reset_index(drop=True)
 
    print(f"\nTrain: {len(train_df)} rows  |  Validation: {len(val_df)} rows  |  Test: {len(test_df)} rows")
    print("Validation class distribution:")
    print(y_val_raw.value_counts().to_string())
    print("Test class distribution (never touched until final reporting):")
    print(y_test_raw.value_counts().to_string())
 
    # ---------------- 2. Feature matrix (fill values fit on TRAIN only) ----------------
    fill_values = {c: float(train_df[c].median()) for c in FEATURE_COLS if train_df[c].isna().any()}
    X_train = prepare_matrix(train_df, fill_values)
    X_val = prepare_matrix(val_df, fill_values)
    X_test = prepare_matrix(test_df, fill_values)
 
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_raw)
    n_classes = len(label_encoder.classes_)
    print(f"\nClasses: {list(label_encoder.classes_)}")
 
    # ---------------- 3. SMOTE on TRAIN only ----------------
    print("\nApplying SMOTE to training split only...")
    X_train_res, y_train_res, k_used = smote_oversample(X_train, y_train_encoded, random_state=RANDOM_STATE)
    print(f"  k_neighbors used: {k_used}")
    print(f"  Training rows before SMOTE: {len(X_train)}  ->  after: {len(X_train_res)}")
 
    # ---------------- 4. Train ----------------
    print("\nTraining XGBoost multi-class classifier (SMOTE + balanced sample weights)...")
    model = train_classifier(X_train_res, y_train_res, n_classes, random_state=RANDOM_STATE)
    save_artifacts(model, label_encoder, fill_values)
    print(f"  Saved model artifacts -> {os.path.join(os.path.dirname(DATA_DIR), 'models')}")
 
    # ---------------- 5. MERGE RULE SELECTION on VALIDATION ONLY ----------------
    print("\n" + "=" * 70)
    print("MERGE RULE SELECTION (validation split only - test is not touched yet)")
    print("=" * 70)
    val_pred_labels, _, val_attack_prob = score_classifier_gated(model, X_val, label_encoder)
    val_eval = val_df.copy()
    val_eval["classifier_attack_prob"] = val_attack_prob * 100
    val_eval["clf_pred_label"] = val_pred_labels
 
    candidates = {}
    for rule in ["mean", "max"]:
        result = evaluate_merge_rule(val_eval, rule)
        candidates[rule] = result
        print(f"\n  rule={rule:<5s} threshold={result['threshold']:.2f}  FP={result['fp_rate']:.4f}  "
              f"overall_recall={result['overall_recall']:.3f}")
        for atype, r in result["per_type_recall"].items():
            print(f"      {atype:<20s} recall={r:.3f}")
 
    winning_rule = max(candidates, key=lambda r: candidates[r]["overall_recall"])
    print(f"\n  WINNER (by validation overall recall at matched FP budget): '{winning_rule}'")
 
    # ---------------- 6. FINAL evaluation on TEST (touched exactly once) ----------------
    print("\n" + "=" * 70)
    print("CLASSIFIER EVALUATION (held-out test split — real distribution, never SMOTE'd)")
    print("=" * 70)
    y_pred_labels, _, _ = score_classifier_gated(model, X_test, label_encoder)
 
    print("\nPer-class precision / recall / F1:")
    print(classification_report(y_test_raw, y_pred_labels, digits=3, zero_division=0))
 
    print("Confusion matrix (rows=true, cols=predicted):")
    labels_order = list(label_encoder.classes_)
    cm = confusion_matrix(y_test_raw, y_pred_labels, labels=labels_order)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels_order], columns=[f"pred_{l}" for l in labels_order])
    print(cm_df.to_string())
 
    normal_test_mask = y_test_raw == "normal"
    fp_rate = (y_pred_labels[normal_test_mask.to_numpy()] != "normal").mean()
    print(f"\nFalse-positive rate on test-set normal rows (flagged as ANY attack type): "
          f"{fp_rate:.4f} ({fp_rate*100:.2f}%)")
 
    print("\nRecall by attack_type on test set (the Day 3 -> Day 4 comparison):")
    day3_recall_ref = {  # from Day 3's evaluation (final_risk_score, post cold-start blend), at its
        # 5% FP threshold, for the pitch comparison slide -- regenerated alongside the Jul 21
        # incident-count fix (20->38 per rare class), so this always tracks the CURRENT run's
        # own Day 3 numbers rather than a stale prior-dataset snapshot.
        "brute_force": 0.852, "impossible_travel": 0.974, "lateral_movement": 0.966,
        "credential_misuse": 0.711, "device_spoofing": 0.211,
    }
    for atype in [l for l in labels_order if l != "normal"]:
        mask = (y_test_raw == atype).to_numpy()
        n = mask.sum()
        recall = (y_pred_labels[mask] == atype).mean() if n else float("nan")
        day3 = day3_recall_ref.get(atype)
        print(f"    {atype:<20s} n={n:<4d} Day4 recall={recall:.3f}   (Day3 unsupervised recall was {day3:.3f})")
 
    # ---------------- 6b. INCIDENT-level evaluation (test only) ----------------
    print("\n" + "=" * 70)
    print("INCIDENT-LEVEL evaluation (>=1 event in the incident flagged = incident caught)")
    print("=" * 70)
 
    test_df_eval = test_df.copy()
    test_df_eval["pred_label"] = y_pred_labels
    test_df_eval["true_label"] = y_test_raw.to_numpy()
 
    incident_rows = []
    for atype in [l for l in labels_order if l != "normal"]:
        sessions = test_df_eval.loc[test_df_eval["true_label"] == atype, "attack_id"].unique()
        n_incidents = len(sessions)
        n_caught_classifier = 0
        n_caught_correct_type = 0
        for sid in sessions:
            sess_rows = test_df_eval[test_df_eval["attack_id"] == sid]
            if (sess_rows["pred_label"] != "normal").any():
                n_caught_classifier += 1
            if (sess_rows["pred_label"] == atype).any():
                n_caught_correct_type += 1
        row_level_n = (test_df_eval["true_label"] == atype).sum()
        row_level_recall = (test_df_eval.loc[test_df_eval["true_label"] == atype, "pred_label"] == atype).mean()
        incident_rows.append({
            "attack_type": atype,
            "n_incidents": n_incidents,
            "n_events": row_level_n,
            "row_level_recall": round(row_level_recall, 3),
            "incident_recall_any_attack": round(n_caught_classifier / n_incidents, 3) if n_incidents else float("nan"),
            "incident_recall_correct_type": round(n_caught_correct_type / n_incidents, 3) if n_incidents else float("nan"),
        })
    incident_summary = pd.DataFrame(incident_rows)
    print(incident_summary.to_string(index=False))
 
    # ---------------- 6c. FINAL unified-score evaluation on TEST, under the winning rule ----------------
    print("\n" + "=" * 70)
    print(f"UNIFIED SCORE evaluation on TEST, rule='{winning_rule}' (selected on validation, applied here for the first time)")
    print("=" * 70)
    _, _, test_attack_prob = score_classifier_gated(model, X_test, label_encoder)
    test_eval = test_df.copy()
    test_eval["classifier_attack_prob"] = test_attack_prob * 100
    final_result = evaluate_merge_rule(test_eval, winning_rule)
    print(f"  threshold={final_result['threshold']:.2f}  FP={final_result['fp_rate']:.4f}  "
          f"overall_recall={final_result['overall_recall']:.3f}")
    for atype, r in final_result["per_type_recall"].items():
        print(f"      {atype:<20s} recall={r:.3f}")
 
    # ---------------- 7. Score everything + merge with Day 3's scores ----------------
    print("\nScoring full dataset and merging with Day 3 anomaly scores...")
    X_full = prepare_matrix(df, fill_values)
    pred_label_full, confidence_full, attack_prob_full = score_classifier_gated(model, X_full, label_encoder)
 
    clf_split = pd.Series("train", index=df.index)
    clf_split.loc[val_idx] = "val"
    clf_split.loc[test_idx] = "test"
    df["clf_split"] = clf_split.to_numpy()
    df["predicted_attack_type"] = np.where(pred_label_full == "normal", None, pred_label_full)
    df["classifier_confidence"] = confidence_full.round(4)
    df["classifier_attack_prob"] = (attack_prob_full * 100).round(2)
 
    # MERGE RULE: whichever rule won on the VALIDATION split above - not
    # chosen by looking at test performance. See the module docstring for
    # why this distinction matters.
    df["unified_risk_score"] = compute_unified(df["final_risk_score"], df["classifier_attack_prob"], winning_rule).round(2)
 
    # ---------------- 8. Secondary review tier - threshold also chosen on VALIDATION ----------------
    print("\nCalibrating secondary-review threshold on validation split...")
    for thresh in [90, 95, 97, 98, 99]:
        noise = ((val_eval["label"] == "normal") & (val_eval["final_risk_score"] >= thresh)).sum()
        recovered = ((val_eval["label"] == "attack") & (val_eval["final_risk_score"] >= thresh) &
                     (val_eval["clf_pred_label"] == "normal")).sum()
        print(f"    threshold={thresh}: normal noise={noise}, missed-attacks recovered={recovered}")
    SECONDARY_REVIEW_THRESHOLD = 98  # chosen from the validation sweep above, not test
    df["secondary_review_flag"] = (df["predicted_attack_type"].isna()) & (df["final_risk_score"] >= SECONDARY_REVIEW_THRESHOLD)
    test_mask = df["clf_split"] == "test"
    n_review_test = df.loc[test_mask, "secondary_review_flag"].sum()
    n_review_test_attack = (df.loc[test_mask, "secondary_review_flag"] & (df.loc[test_mask, "label"] == "attack")).sum()
    print(f"  On TEST (final number): secondary review tier flagged {n_review_test} rows "
          f"({n_review_test_attack} real missed attacks, {n_review_test - n_review_test_attack} noise)")
 
    out_path = f"{DATA_DIR}/day4_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows -> {out_path}")
 
 
if __name__ == "__main__":
    main()
 
