import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import build_feature_matrix, load_dataset  # noqa: E402
from src.modeling import (  # noqa: E402
    MODEL_NAMES, RANDOM_SEED, evaluate_at_threshold, naive_baseline_costs,
    score, train_candidates, tune_threshold,
)

REPORT_DIR = "reports"

FALSE_NEGATIVE_FIXED_COST = 250
FALSE_POSITIVE_COST = 180


def _split(df, target_col):
    train_df, temp_df = train_test_split(df, test_size=0.40, stratify=df[target_col], random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df[target_col], random_state=RANDOM_SEED)
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True))



def load_indian_bfsi_upi():
    path = "data/transactions.csv"
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run `python data/generate_data.py` first.")
    df = load_dataset(path)
    target_col = "is_fraud"
    amount_col = "amount_inr"
    numeric_cols = [
        "hour", "is_night", "is_weekend", "amount_inr", "account_age_days",
        "is_new_account", "avg_amount_last_30d", "amount_to_avg_ratio",
        "txn_count_last_24h", "txn_count_last_1h", "new_device_flag",
        "new_beneficiary_flag", "sim_recently_changed_flag", "upi_pin_reset_last_7d",
        "failed_otp_attempts_last_1h", "login_location_mismatch_flag",
    ]
    categorical_cols = ["transaction_type", "channel"]
    return df, target_col, amount_col, numeric_cols, categorical_cols


def load_eu_style_extreme():
    path = "data/benchmark_eu_style.csv"
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run `python data/generate_benchmark_data.py` first.")
    df = pd.read_csv(path)
    target_col = "Class"
    amount_col = "Amount"
    numeric_cols = [c for c in df.columns if c not in (target_col,)]
    categorical_cols = []
    return df, target_col, amount_col, numeric_cols, categorical_cols


DATASET_REGISTRY = {
    "indian_bfsi_upi": load_indian_bfsi_upi,
    "eu_style_extreme": load_eu_style_extreme,
}


def _build_features_generic(df, numeric_cols, categorical_cols, fit_columns=None):
    """A copy of features.build_feature_matrix that works for any dataset's
    column layout, not just the Indian-merchant schema."""
    if categorical_cols:
        encoded = pd.get_dummies(df[categorical_cols], prefix=categorical_cols)
        X = pd.concat([df[numeric_cols].reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)
    else:
        X = df[numeric_cols].reset_index(drop=True)
    if fit_columns is not None:
        X = X.reindex(columns=fit_columns, fill_value=0)
    return X, list(X.columns)


def run_dataset(name: str, loader) -> dict:
    print(f"\n{'='*70}\nBENCHMARK DATASET: {name}\n{'='*70}")
    df, target_col, amount_col, numeric_cols, categorical_cols = loader()
    train_df, val_df, test_df = _split(df, target_col)

    X_train, feature_cols = _build_features_generic(train_df, numeric_cols, categorical_cols)
    y_train = train_df[target_col].values
    X_val, _ = _build_features_generic(val_df, numeric_cols, categorical_cols, fit_columns=feature_cols)
    y_val = val_df[target_col].values
    amounts_val = val_df[amount_col].values
    X_test, _ = _build_features_generic(test_df, numeric_cols, categorical_cols, fit_columns=feature_cols)
    y_test = test_df[target_col].values
    amounts_test = test_df[amount_col].values

    print(f"Rows: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")
    print(f"Fraud rate: {df[target_col].mean():.4%}  (test set fraud count: {int(y_test.sum())})")

    candidates = train_candidates(X_train, y_train)

    baselines = naive_baseline_costs(
        y_test, amounts_test, FALSE_NEGATIVE_FIXED_COST, FALSE_POSITIVE_COST,
        high_amount_cutoff=float(np.percentile(amounts_test, 90)),
    )

    per_model = {}
    for model_name, entry in candidates.items():
        p_val = score(entry, X_val)
        p_test = score(entry, X_test)
        threshold, _, _, _ = tune_threshold(y_val, amounts_val, p_val, FALSE_NEGATIVE_FIXED_COST, FALSE_POSITIVE_COST)
        result = evaluate_at_threshold(y_test, amounts_test, p_test, threshold,
                                        FALSE_NEGATIVE_FIXED_COST, FALSE_POSITIVE_COST)
        per_model[model_name] = result
        print(f"  {model_name:22s} precision={result['precision']:.3f}  recall={result['recall']:.3f}  "
              f"PR-AUC={result['pr_auc']:.3f}  ROC-AUC={result['roc_auc']:.3f}  "
              f"cost=Rs.{result['estimated_cost']:,.0f}")

    return {
        "dataset": name,
        "n_rows": int(len(df)),
        "fraud_rate": float(df[target_col].mean()),
        "test_rows": int(len(test_df)),
        "test_fraud_count": int(y_test.sum()),
        "baselines": baselines,
        "per_model_results": per_model,
    }


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    all_results = {}
    for name, loader in DATASET_REGISTRY.items():
        all_results[name] = run_dataset(name, loader)

    print(f"\n{'='*70}\nCROSS-DATASET SUMMARY (best model per dataset by PR-AUC)\n{'='*70}")
    header = f"{'Dataset':20s} {'Fraud rate':>12s} {'Best model':>20s} {'Precision':>10s} {'Recall':>8s} {'PR-AUC':>8s}"
    print(header)
    summary_rows = []
    for name, res in all_results.items():
        best_model = max(res["per_model_results"], key=lambda m: res["per_model_results"][m]["pr_auc"])
        r = res["per_model_results"][best_model]
        print(f"{name:20s} {res['fraud_rate']:>11.4%} {best_model:>20s} {r['precision']:>10.3f} "
              f"{r['recall']:>8.3f} {r['pr_auc']:>8.3f}")
        summary_rows.append({
            "dataset": name, "fraud_rate": res["fraud_rate"], "best_model": best_model,
            "precision": r["precision"], "recall": r["recall"], "pr_auc": r["pr_auc"],
            "roc_auc": r["roc_auc"], "estimated_cost": r["estimated_cost"],
        })

    im = summary_rows[0]  
    eu = summary_rows[1]  
    precision_delta = "lower" if eu["precision"] < im["precision"] else "higher"
    prauc_delta = "lower" if eu["pr_auc"] < im["pr_auc"] else "higher"
    print(
        f"\nTakeaway: the SAME three-model pipeline and cost-tuning logic (src/modeling.py) "
        f"was applied unchanged to a {im['fraud_rate']:.2%}-fraud named-feature problem and a "
        f"{eu['fraud_rate']:.2%}-fraud anonymized-component problem. On this run, precision at "
        f"the extreme-imbalance dataset came out {precision_delta} ({eu['precision']:.3f} vs. "
        f"{im['precision']:.3f}) — expected, since ~13x fewer positive examples per transaction "
        f"makes precision fundamentally harder to hold at any ranking quality — while PR-AUC "
        f"came out {prauc_delta} ({eu['pr_auc']:.3f} vs. {im['pr_auc']:.3f}), because this "
        f"synthetic generator's latent fraud/legitimate separation happens to be cleaner than "
        f"the noisier Indian-merchant process. Reporting whichever direction the numbers "
        f"actually move — not the direction that was expected going in — is the point of "
        f"running a second, independently-generated dataset at all."
    )

    with open(os.path.join(REPORT_DIR, "benchmark_report.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    pd.DataFrame(summary_rows).to_csv(os.path.join(REPORT_DIR, "benchmark_summary.csv"), index=False)
    print(f"\nSaved -> {REPORT_DIR}/benchmark_report.json")
    print(f"Saved -> {REPORT_DIR}/benchmark_summary.csv")


if __name__ == "__main__":
    main()
