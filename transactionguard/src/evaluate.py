import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import build_feature_matrix, TARGET_COL  # noqa: E402
from src.modeling import (  # noqa: E402
    MODEL_NAMES, evaluate_at_threshold, naive_baseline_costs, score, tune_threshold,
)

MODEL_DIR = "models"
REPORT_DIR = "reports"

FALSE_NEGATIVE_FIXED_COST_INR = 500        
FALSE_NEGATIVE_RECOVERY_FAILURE_RATE = 0.9  
FALSE_POSITIVE_COST_INR = 120             
HIGH_AMOUNT_BASELINE_CUTOFF = 20000


def load_all_models(feature_cols_path=os.path.join(MODEL_DIR, "feature_columns.json")):
    with open(feature_cols_path) as f:
        feature_cols = json.load(f)
    entries = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODEL_DIR, f"fraud_model_{name}.joblib")
        if os.path.exists(path):
            entries[name] = joblib.load(path)
    return entries, feature_cols


def main():
    entries, feature_cols = load_all_models()
    if not entries:
        raise SystemExit("No trained models found in models/. Run `python src/train.py` first.")

    val_df = pd.read_csv(os.path.join(REPORT_DIR, "split_val.csv"))
    test_df = pd.read_csv(os.path.join(REPORT_DIR, "split_test.csv"))

    X_val, _ = build_feature_matrix(val_df, fit_columns=feature_cols)
    y_val = val_df[TARGET_COL].values
    amounts_val = val_df["amount_inr"].values * FALSE_NEGATIVE_RECOVERY_FAILURE_RATE

    X_test, _ = build_feature_matrix(test_df, fit_columns=feature_cols)
    y_test = test_df[TARGET_COL].values
    amounts_test = test_df["amount_inr"].values * FALSE_NEGATIVE_RECOVERY_FAILURE_RATE

    baselines = naive_baseline_costs(
        y_test, amounts_test, FALSE_NEGATIVE_FIXED_COST_INR, FALSE_POSITIVE_COST_INR,
        HIGH_AMOUNT_BASELINE_CUTOFF,
    )

    print(f"Test rows: {len(test_df):,}  |  Fraud in test: {y_test.sum():,} ({y_test.mean():.3%})")
    print(f"Naive baseline costs (test set): {baselines}\n")

    per_model_results = {}
    test_scores_by_model = {}

    for name, entry in entries.items():
        p_val = score(entry, X_val)
        p_test = score(entry, X_test)
        test_scores_by_model[name] = p_test

        threshold, val_cost, _, _ = tune_threshold(
            y_val, amounts_val, p_val, FALSE_NEGATIVE_FIXED_COST_INR, FALSE_POSITIVE_COST_INR
        )
        result = evaluate_at_threshold(
            y_test, amounts_test, p_test, threshold,
            FALSE_NEGATIVE_FIXED_COST_INR, FALSE_POSITIVE_COST_INR,
        )
        result["validation_cost_at_threshold"] = val_cost
        per_model_results[name] = result

        best_baseline_cost = min(baselines.values())
        saved = best_baseline_cost - result["estimated_cost"]
        pct = (saved / best_baseline_cost * 100) if best_baseline_cost else 0

        print(f"=== {name} ===")
        print(f"  Threshold (cost-tuned on validation): {threshold:.2f}")
        print(f"  Precision: {result['precision']:.4f}   Recall: {result['recall']:.4f}   "
              f"F1: {result['f1']:.4f}")
        print(f"  PR-AUC: {result['pr_auc']:.4f}   ROC-AUC: {result['roc_auc']:.4f}")
        cm = result["confusion_matrix"]
        print(f"  Confusion matrix -> TN:{cm['tn']:,} FP:{cm['fp']:,} FN:{cm['fn']:,} TP:{cm['tp']:,}")
        print(f"  Missed frauds (FN): {result['false_negative_count']:,}   "
              f"Wrongly blocked genuine orders (FP): {result['false_positive_count']:,}")
        print(f"  Estimated cost: Rs.{result['estimated_cost']:,.0f}  "
              f"(vs. best naive baseline Rs.{best_baseline_cost:,.0f} -> "
              f"saves Rs.{saved:,.0f}, {pct:.1f}%)\n")

    # ---- cross-model comparison table (the gouldju1-style summary) ----
    print("===== CROSS-MODEL COST/ERROR-PROFILE COMPARISON (held-out test set) =====")
    header = f"{'Model':22s} {'Precision':>10s} {'Recall':>8s} {'FN (missed fraud)':>20s} " \
             f"{'FP (blocked genuine)':>22s} {'Est. cost (Rs.)':>17s}"
    print(header)
    for name in MODEL_NAMES:
        if name not in per_model_results:
            continue
        r = per_model_results[name]
        print(f"{name:22s} {r['precision']:>10.3f} {r['recall']:>8.3f} "
              f"{r['false_negative_count']:>20,d} {r['false_positive_count']:>22,d} "
              f"{r['estimated_cost']:>17,.0f}")

    cheapest_model = min(per_model_results, key=lambda n: per_model_results[n]["estimated_cost"])
    highest_recall_model = max(per_model_results, key=lambda n: per_model_results[n]["recall"])
    print(f"\nLowest total cost on this cost model: {cheapest_model}")
    print(f"Highest fraud-catch rate (recall): {highest_recall_model}")
    if cheapest_model != highest_recall_model:
        print("These are different models — a merchant who cares more about catching "
              "fraud than about customer friction (e.g. higher-risk categories, thin "
              f"margins) may reasonably prefer {highest_recall_model} over the "
              f"lowest-total-cost pick, even though it costs more under these specific "
              "assumptions. This is exactly why the cost assumptions are exposed as "
              "config, not baked into a single 'winner'.")

    # ---- save everything ----
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "model_comparison.json"), "w") as f:
        json.dump({
            "test_rows": int(len(test_df)),
            "test_fraud_count": int(y_test.sum()),
            "test_fraud_rate": float(y_test.mean()),
            "cost_assumptions": {
                "false_negative_fixed_cost_inr": FALSE_NEGATIVE_FIXED_COST_INR,
                "false_positive_cost_inr": FALSE_POSITIVE_COST_INR,
                "high_amount_baseline_cutoff_inr": HIGH_AMOUNT_BASELINE_CUTOFF,
            },
            "baseline_costs_inr": baselines,
            "per_model_results": per_model_results,
            "cheapest_model": cheapest_model,
            "highest_recall_model": highest_recall_model,
        }, f, indent=2)

    pd.DataFrame([
        {"model": name, **{k: v for k, v in r.items() if k != "confusion_matrix"}}
        for name, r in per_model_results.items()
    ]).to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"), index=False)

    demo_model_name = json.load(open(os.path.join(MODEL_DIR, "training_summary.json")))["default_demo_model"]
    with open(os.path.join(REPORT_DIR, "test_results.json"), "w") as f:
        json.dump({"selected_threshold": per_model_results[demo_model_name]["threshold"],
                   **per_model_results[demo_model_name]}, f, indent=2)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = {"logistic_regression": "#2980b9", "random_forest": "#27ae60", "xgboost": "#c0392b"}

    for name, p_test in test_scores_by_model.items():
        prec, rec, _ = precision_recall_curve(y_test, p_test)
        axes[0].plot(rec, prec, label=name, color=colors.get(name))
    axes[0].axhline(y_test.mean(), color="gray", linestyle="--", label="random baseline")
    axes[0].set_title("Precision-Recall curves (test) — all 3 models")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend(fontsize=8)

    for name, p_test in test_scores_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, p_test)
        axes[1].plot(fpr, tpr, label=name, color=colors.get(name))
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[1].set_title("ROC curves (test) — all 3 models")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(fontsize=8)

    model_labels = list(per_model_results.keys())
    costs_bar = [per_model_results[m]["estimated_cost"] for m in model_labels]
    bar_colors = [colors.get(m, "#7f8c8d") for m in model_labels]
    axes[2].bar(model_labels, costs_bar, color=bar_colors)
    for base_name, base_cost in baselines.items():
        axes[2].axhline(base_cost, linestyle="--", linewidth=1,
                         label=f"baseline: {base_name} (Rs.{base_cost:,.0f})")
    axes[2].set_title("Estimated test-set cost by model vs. naive baselines")
    axes[2].set_ylabel("Estimated cost (Rs.)")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    plot_path = os.path.join(REPORT_DIR, "evaluation_plots.png")
    plt.savefig(plot_path, dpi=140)
    print(f"\nSaved plots -> {plot_path}")
    print(f"Saved per-model metrics -> {REPORT_DIR}/model_comparison.json / .csv")
    print(f"Saved demo-model metrics -> {REPORT_DIR}/test_results.json")


if __name__ == "__main__":
    main()
