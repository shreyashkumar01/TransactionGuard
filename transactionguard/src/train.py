import json
import os
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import build_feature_matrix, load_dataset, TARGET_COL  
from src.modeling import (  
    RANDOM_SEED, MODEL_NAMES, cross_validate_candidates, train_candidates,
)

DATA_PATH = "data/transactions.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"


def make_splits(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df, test_size=0.40, stratify=df[TARGET_COL], random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df[TARGET_COL], random_state=RANDOM_SEED
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = load_dataset(DATA_PATH)
    train_df, val_df, test_df = make_splits(df)

    X_train, feature_cols = build_feature_matrix(train_df)
    y_train = train_df[TARGET_COL].values

    print(f"Train: {len(train_df):,} rows ({y_train.mean():.3%} fraud)")
    print(f"Val:   {len(val_df):,} rows ({val_df[TARGET_COL].mean():.3%} fraud)")
    print(f"Test:  {len(test_df):,} rows ({test_df[TARGET_COL].mean():.3%} fraud)")

    print(f"\nRunning 5-fold stratified cross-validation on the training split "
          f"(model comparison, recall-first)...")
    cv_results = cross_validate_candidates(X_train, y_train)

    print(f"\n{'Model':22s} {'Recall (mean±std)':>20s} {'Precision (mean±std)':>22s} "
          f"{'F1 (mean±std)':>18s} {'PR-AUC (mean±std)':>20s}")
    for name in MODEL_NAMES:
        r = cv_results[name]
        print(f"{name:22s} "
              f"{r['recall']['mean']:.3f}±{r['recall']['std']:.3f}".rjust(20) + "  "
              f"{r['precision']['mean']:.3f}±{r['precision']['std']:.3f}".rjust(22) + "  "
              f"{r['f1']['mean']:.3f}±{r['f1']['std']:.3f}".rjust(18) + "  "
              f"{r['pr_auc']['mean']:.3f}±{r['pr_auc']['std']:.3f}".rjust(20))

    
    best_by_recall = max(MODEL_NAMES, key=lambda n: cv_results[n]["recall"]["mean"])
    print(f"\nHighest mean CV recall: {best_by_recall}")

    print("\nFitting final models on the full training split...")
    candidates = train_candidates(X_train, y_train)

    for name, entry in candidates.items():
        joblib.dump(entry, os.path.join(MODEL_DIR, f"fraud_model_{name}.joblib"))
    joblib.dump(candidates[best_by_recall], os.path.join(MODEL_DIR, "fraud_model.joblib"))

    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

    background_sample = X_train.sample(n=min(200, len(X_train)), random_state=RANDOM_SEED)
    joblib.dump(background_sample, os.path.join(MODEL_DIR, "shap_background.joblib"))

    train_df.to_csv(os.path.join(REPORT_DIR, "split_train.csv"), index=False)
    val_df.to_csv(os.path.join(REPORT_DIR, "split_val.csv"), index=False)
    test_df.to_csv(os.path.join(REPORT_DIR, "split_test.csv"), index=False)

    with open(os.path.join(MODEL_DIR, "training_summary.json"), "w") as f:
        json.dump({
            "default_demo_model": best_by_recall,
            "selection_rule": "highest mean 5-fold CV recall on the training split",
            "cross_validation_results": cv_results,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "fraud_rate_overall": float(df[TARGET_COL].mean()),
            "models_saved": [f"fraud_model_{n}.joblib" for n in MODEL_NAMES],
        }, f, indent=2)

    print(f"\nSaved all 3 models -> {MODEL_DIR}/fraud_model_{{name}}.joblib")
    print(f"Saved demo model ({best_by_recall}) -> {MODEL_DIR}/fraud_model.joblib")
    print(f"Saved feature columns -> {MODEL_DIR}/feature_columns.json")
    print("Run `python src/evaluate.py` next for the full cost-based, "
          "multi-model held-out test report.")


if __name__ == "__main__":
    main()
