
"""
run_day3.py
===========
Day 3 driver script. Ties together models.py + coldstart.py:
 
  1. Load features.csv (output of Day 2's features.py)
  2. Split NORMAL rows into train / held-out-normal (attacks are NEVER
     trained on - only used for evaluation)
  3. Fit Isolation Forest + Autoencoder on train-normal only
  4. Score every row (train-normal, held-out-normal, attacks) with both
     models, percentile-normalize each to 0-100, average into
     `ensemble_score_raw`
  5. Apply cold-start population-baseline blending -> `final_risk_score`
  6. Evaluate on held-out-normal + attacks ONLY (never on train-normal,
     which the models were literally fit to and would trivially "pass")
  7. Save data/scored_events.csv (feeds Day 4's classifier + Day 6's
     dashboard) and print a metrics summary
 
Run:
    python3 run_day3.py
"""
 
import os
 
import numpy as np
import pandas as pd
 
from coldstart import blend_cold_start_scores, compute_population_reference
from models import (
    FEATURE_COLS,
    prepare_matrix,
    save_artifacts,
    score_autoencoder,
    score_isolation_forest,
    train_autoencoder,
    train_isolation_forest,
)
 
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.30   # fraction of NORMAL rows held out from training, for honest FP-rate eval
 
 
def evaluate(df_eval: pd.DataFrame, score_col: str, alert_threshold: float):
    """Prints ROC-AUC, alert-threshold FP rate on normal, and per-attack-type
    recall at that threshold. df_eval must contain only held-out-normal +
    attack rows (never train-normal)."""
    from sklearn.metrics import roc_auc_score
 
    y_true = (df_eval["label"] == "attack").astype(int)
    auc = roc_auc_score(y_true, df_eval[score_col])
    print(f"\n[{score_col}] ROC-AUC (held-out normal vs. all attacks): {auc:.4f}")
 
    normal_mask = df_eval["label"] == "normal"
    fp_rate = (df_eval.loc[normal_mask, score_col] >= alert_threshold).mean()
    print(f"[{score_col}] Alert threshold: {alert_threshold:.2f}  ->  "
          f"False-positive rate on held-out normal: {fp_rate:.4f} ({fp_rate*100:.2f}%)")
 
    print(f"[{score_col}] Recall (detection rate) by attack_type at this threshold:")
    attack_df = df_eval[df_eval["label"] == "attack"]
    for atype, grp in attack_df.groupby("attack_type"):
        recall = (grp[score_col] >= alert_threshold).mean()
        print(f"    {atype:<20s} n={len(grp):<4d} recall={recall:.3f}")
 
    overall_attack_recall = (attack_df[score_col] >= alert_threshold).mean()
    print(f"[{score_col}] Overall attack recall: {overall_attack_recall:.3f}")
    return auc, fp_rate, overall_attack_recall
 
 
def main():
    print("Loading features.csv...")
    df = pd.read_csv(f"{DATA_DIR}/features.csv", parse_dates=["timestamp"])
    print(f"  {len(df)} rows  |  {(df['label'] == 'attack').sum()} attack rows")
 
    normal_df = df[df["label"] == "normal"].copy()
    attack_df = df[df["label"] == "attack"].copy()
 
    train_df, holdout_normal_df = train_test_split_df(normal_df, HOLDOUT_FRACTION, RANDOM_STATE)
    print(f"  Normal split -> train: {len(train_df)}  |  held-out: {len(holdout_normal_df)}")
 
    # ---------------- 1. Fit feature matrix + fill values on TRAIN only ----------------
    X_train, fill_values = prepare_matrix(train_df, fill_values=None)
 
    # ---------------- 2. Train both models on normal-only train split ----------------
    print("\nTraining Isolation Forest (normal-only)...")
    iso_forest = train_isolation_forest(X_train, random_state=RANDOM_STATE)
 
    print("Training Autoencoder (normal-only)...")
    ae, ae_scaler = train_autoencoder(X_train, random_state=RANDOM_STATE)
 
    save_artifacts(iso_forest, ae, ae_scaler, fill_values)
    print(f"  Saved model artifacts -> {os.path.join(os.path.dirname(DATA_DIR), 'models')}")
 
    # ---------------- 3. Reference distributions for percentile normalization ----------------
    if_train_scores = score_isolation_forest(iso_forest, X_train)
    ae_train_scores = score_autoencoder(ae, ae_scaler, X_train)
 
    # ---------------- 4. Score EVERYTHING (train + holdout + attacks) ----------------
    full_df = pd.concat([
        train_df.assign(split="train_normal"),
        holdout_normal_df.assign(split="holdout_normal"),
        attack_df.assign(split="attack"),
    ], ignore_index=True)
 
    X_full, _ = prepare_matrix(full_df, fill_values=fill_values)
    if_raw = score_isolation_forest(iso_forest, X_full)
    ae_raw = score_autoencoder(ae, ae_scaler, X_full)
 
    from models import percentile_normalize
    if_pct = percentile_normalize(if_raw, if_train_scores)
    ae_pct = percentile_normalize(ae_raw, ae_train_scores)
 
    full_df["iso_forest_score"] = if_pct.round(2)
    full_df["autoencoder_score"] = ae_pct.round(2)
    full_df["ensemble_score_raw"] = ((if_pct + ae_pct) / 2).round(2)
 
    # ---------------- 5. Cold-start blending ----------------
    established_normal = full_df[(full_df["label"] == "normal") & (full_df["is_cold_start_entity"] == 0)]
    population_reference = compute_population_reference(established_normal)
    full_df["final_risk_score"] = blend_cold_start_scores(full_df, population_reference)
 
    # ---------------- 6. Evaluate on holdout_normal + attack ONLY ----------------
    eval_df = full_df[full_df["split"] != "train_normal"].copy()
    alert_threshold = np.percentile(full_df.loc[full_df["split"] == "holdout_normal", "ensemble_score_raw"], 95)
 
    print("\n" + "=" * 70)
    print("EVALUATION (held-out normal + attacks — models never trained on these)")
    print("=" * 70)
    evaluate(eval_df, "ensemble_score_raw", alert_threshold)
 
    alert_threshold_final = np.percentile(full_df.loc[full_df["split"] == "holdout_normal", "final_risk_score"], 95)
    print("\n--- After cold-start blending (final_risk_score) ---")
    evaluate(eval_df, "final_risk_score", alert_threshold_final)
 
    # cold-start entities specifically: did blending reduce their scores as intended?
    cold_rows = full_df[(full_df["is_cold_start_entity"] == 1) & (full_df["label"] == "normal")]
    if len(cold_rows):
        print(f"\nCold-start normal rows (n={len(cold_rows)}): "
              f"mean raw={cold_rows['ensemble_score_raw'].mean():.2f} -> "
              f"mean blended={cold_rows['final_risk_score'].mean():.2f}")
 
    # ---------------- 7. Save scored_events.csv for Day 4 / Day 6 ----------------
    out_cols = ["event_id", "timestamp", "user_id", "dept", "host", "device_id", "status",
                "label", "attack_type", "attack_id", "split",
                "is_cold_start_entity", "cold_start_blend_weight",
                "iso_forest_score", "autoencoder_score",
                "ensemble_score_raw", "final_risk_score"] + FEATURE_COLS
    out_path = f"{DATA_DIR}/scored_events.csv"
    full_df[out_cols].to_csv(out_path, index=False)
    print(f"\nSaved {len(full_df)} scored rows -> {out_path}")
 
 
def train_test_split_df(df: pd.DataFrame, holdout_fraction: float, random_state: int):
    from sklearn.model_selection import train_test_split
    train_df, holdout_df = train_test_split(df, test_size=holdout_fraction, random_state=random_state)
    return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True)
 
 
if __name__ == "__main__":
    main()
 
