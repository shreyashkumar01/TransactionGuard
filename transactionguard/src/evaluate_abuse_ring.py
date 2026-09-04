import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_score,
    recall_score, roc_auc_score,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.abuse_ring_sentinel import build_account_graph, score_clusters  # noqa: E402

DATA_PATH = "data/abuse_ring_accounts.csv"
REPORT_DIR = "reports"

FALSE_NEGATIVE_COST = 5000
FALSE_POSITIVE_COST = 150


def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit("Run `python data/generate_abuse_ring_data.py` first.")

    df = pd.read_csv(DATA_PATH)
    graph = build_account_graph(df)
    cluster_scores = score_clusters(df, graph)

    account_rows = []
    for _, row in cluster_scores.iterrows():
        for member in row["members"]:
            account_rows.append({
                "account_id": member,
                "graph_cluster_id": row["graph_cluster_id"],
                "cluster_size": row["size"],
                "suspicion_score": row["suspicion_score"],
                "predicted_flag": int(row["recommended_action"] == "manual_review"),
            })
    account_scores = pd.DataFrame(account_rows).merge(
        df[["account_id", "is_fraud_ring_member", "cluster_type"]], on="account_id"
    )

    y_true = account_scores["is_fraud_ring_member"].values
    y_score = account_scores["suspicion_score"].values

    grid = np.arange(0.01, 0.99, 0.01)
    costs = []
    for t in grid:
        preds = (y_score >= t).astype(int)
        fn_count = int(((preds == 0) & (y_true == 1)).sum())
        fp_count = int(((preds == 1) & (y_true == 0)).sum())
        costs.append(fn_count * FALSE_NEGATIVE_COST + fp_count * FALSE_POSITIVE_COST)
    best_idx = int(np.argmin(costs))
    cost_optimal_threshold = float(grid[best_idx])

    y_pred_default = account_scores["predicted_flag"].values
    y_pred_cost_optimal = (y_score >= cost_optimal_threshold).astype(int)

    def _metrics_at(y_pred):
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fn * FALSE_NEGATIVE_COST + fp * FALSE_POSITIVE_COST
        return {"precision": float(p), "recall": float(r), "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp), "estimated_cost": int(cost)}

    default_metrics = _metrics_at(y_pred_default)
    cost_optimal_metrics = _metrics_at(y_pred_cost_optimal)

    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    precision, recall = default_metrics["precision"], default_metrics["recall"]
    tn, fp, fn, tp = default_metrics["tn"], default_metrics["fp"], default_metrics["fn"], default_metrics["tp"]
    y_pred = y_pred_default

    fp_mask = (y_pred == 1) & (y_true == 0)
    fp_breakdown = account_scores.loc[fp_mask, "cluster_type"].value_counts().to_dict()

    print(f"Evaluated {len(account_scores):,} accounts "
          f"({int(y_true.sum())} true fraud-ring members, "
          f"{int((1 - y_true).sum())} legitimate)")
    print(f"PR-AUC: {pr_auc:.3f}  (random baseline = {y_true.mean():.3f})")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print()
    print(f"--- Default threshold (0.5, as shipped in abuse_ring_sentinel.py) ---")
    print(f"Precision: {precision:.3f}   Recall: {recall:.3f}")
    print(f"Confusion matrix -> TN:{tn} FP:{fp} FN:{fn} TP:{tp}")
    print(f"False-positive breakdown by cluster type: {fp_breakdown}")
    print(f"Estimated cost @ default threshold: Rs.{default_metrics['estimated_cost']:,}")
    print()
    print(f"--- Cost-optimal threshold ({cost_optimal_threshold:.2f}, "
          f"FN=Rs.{FALSE_NEGATIVE_COST}, FP=Rs.{FALSE_POSITIVE_COST}) ---")
    print(f"Precision: {cost_optimal_metrics['precision']:.3f}   "
          f"Recall: {cost_optimal_metrics['recall']:.3f}")
    print(f"Confusion matrix -> TN:{cost_optimal_metrics['tn']} FP:{cost_optimal_metrics['fp']} "
          f"FN:{cost_optimal_metrics['fn']} TP:{cost_optimal_metrics['tp']}")
    print(f"Estimated cost @ cost-optimal threshold: Rs.{cost_optimal_metrics['estimated_cost']:,}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    account_scores.to_csv(os.path.join(REPORT_DIR, "abuse_ring_scored_accounts.csv"), index=False)
    with open(os.path.join(REPORT_DIR, "abuse_ring_eval_results.json"), "w") as f:
        json.dump({
            "n_accounts": len(account_scores), "n_fraud_ring_members": int(y_true.sum()),
            "n_legitimate": int((1 - y_true).sum()),
            "pr_auc": float(pr_auc), "roc_auc": float(roc_auc),
            "cost_assumptions": {"false_negative_cost": FALSE_NEGATIVE_COST,
                                  "false_positive_cost": FALSE_POSITIVE_COST},
            "default_threshold_0_5": {**default_metrics,
                                       "false_positive_breakdown_by_cluster_type": fp_breakdown},
            "cost_optimal_threshold": cost_optimal_threshold,
            "cost_optimal_metrics": cost_optimal_metrics,
        }, f, indent=2)
    print(f"\nSaved -> {REPORT_DIR}/abuse_ring_eval_results.json")
    print(f"Saved -> {REPORT_DIR}/abuse_ring_scored_accounts.csv")


if __name__ == "__main__":
    main()
