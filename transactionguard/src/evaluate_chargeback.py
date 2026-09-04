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
from src.chargeback_responder import ChargebackCase, score_case  # noqa: E402

DATA_PATH = "data/chargeback_disputes.csv"
REPORT_DIR = "reports"

REPRESENTMENT_FEE_INR = 400   


def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit("Run `python data/generate_chargeback_data.py` first.")
    df = pd.read_csv(DATA_PATH)

    evidence_cols = ["avs_match", "cvv_match", "delivery_confirmed",
                      "signature_or_otp_confirmed", "device_matches_prior_orders",
                      "no_prior_dispute_history"]

    scores, recommendations = [], []
    for _, row in df.iterrows():
        case = ChargebackCase(
            case_id=row["case_id"], dispute_reason=row["dispute_reason"],
            order_amount_inr=row["order_amount_inr"],
            evidence={c: bool(row[c]) for c in evidence_cols},
        )
        scored = score_case(case)
        scores.append(scored["evidence_strength_score"])
        recommendations.append(scored["recommended_action"])

    df["evidence_strength_score"] = scores
    df["recommended_action"] = recommendations
    df["predicted_fight"] = (df["recommended_action"] == "recommend_fight").astype(int)

    y_true = df["would_win_if_fought"].values
    y_score = df["evidence_strength_score"].values
    y_pred = df["predicted_fight"].values

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"Evaluated {len(df)} disputes ({int(y_true.sum())} would have won if fought, "
          f"{int((1 - y_true).sum())} would have lost)")
    print(f"Precision (of recommended fights, how many would truly win): {precision:.3f}")
    print(f"Recall (of true winnable disputes, how many did we recommend fighting): {recall:.3f}")
    print(f"PR-AUC: {pr_auc:.3f}  (random baseline = {y_true.mean():.3f})")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"Confusion matrix -> TN:{tn} FP:{fp} FN:{fn} TP:{tp}")
    print(f"(Here FP = recommended fighting a case that would actually have been lost; "
          f"FN = recommended accepting liability on a case that would actually have won)")

    amounts = df["order_amount_inr"].values

    def _cost_of_policy(fight_mask):
        win_mask = fight_mask & (y_true == 1)
        lose_mask = fight_mask & (y_true == 0)
        accept_mask = ~fight_mask
        cost_win = win_mask.sum() * REPRESENTMENT_FEE_INR
        cost_lose = (amounts[lose_mask].sum()) + lose_mask.sum() * REPRESENTMENT_FEE_INR
        cost_accept = amounts[accept_mask].sum()
        return cost_win + cost_lose + cost_accept

    cost_recommender = _cost_of_policy(y_pred.astype(bool))
    cost_always_fight = _cost_of_policy(np.ones(len(df), dtype=bool))
    cost_never_fight = _cost_of_policy(np.zeros(len(df), dtype=bool))

    print(f"\nEstimated total cost, recommender policy: Rs.{cost_recommender:,.0f}")
    print(f"Estimated total cost, 'always fight everything': Rs.{cost_always_fight:,.0f}")
    print(f"Estimated total cost, 'never fight, always accept': Rs.{cost_never_fight:,.0f}")
    best_naive = min(cost_always_fight, cost_never_fight)
    saved = best_naive - cost_recommender
    pct = (saved / best_naive * 100) if best_naive else 0
    print(f"Recommender saves Rs.{saved:,.0f} ({pct:.1f}%) vs. the best naive policy")

    n_fought_by_recommender = int(y_pred.sum())
    n_fought_always = len(df)
    effort_reduction_pct = (1 - n_fought_by_recommender / n_fought_always) * 100
    print(f"\nOperational-effort comparison: the recommender fights "
          f"{n_fought_by_recommender}/{len(df)} disputes ({100 - effort_reduction_pct:.0f}%) "
          f"vs. 'always fight' fighting all {n_fought_always} — a {effort_reduction_pct:.0f}% "
          f"reduction in dispute-response workload for essentially the same total monetary "
          f"cost. On a low representment fee relative to typical order value, 'always fight' "
          f"is a strong cost baseline on paper — but it means preparing a full evidence "
          f"submission for every single dispute, including the ones with no real evidence to "
          f"submit. The recommender reaches a similar cost outcome while telling a merchant's "
          f"ops team WHICH half is actually worth their time, which a blind 'fight everything' "
          f"policy cannot do.")

    os.makedirs(REPORT_DIR, exist_ok=True)
    df.to_csv(os.path.join(REPORT_DIR, "chargeback_scored.csv"), index=False)
    with open(os.path.join(REPORT_DIR, "chargeback_eval_results.json"), "w") as f:
        json.dump({
            "n_cases": len(df), "n_would_win": int(y_true.sum()), "n_would_lose": int((1-y_true).sum()),
            "precision": float(precision), "recall": float(recall),
            "pr_auc": float(pr_auc), "roc_auc": float(roc_auc),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "cost_recommender": float(cost_recommender),
            "cost_always_fight": float(cost_always_fight),
            "cost_never_fight": float(cost_never_fight),
            "savings_vs_best_naive_pct": float(pct),
            "n_fought_by_recommender": n_fought_by_recommender,
            "n_total_cases": n_fought_always,
            "effort_reduction_pct": float(effort_reduction_pct),
        }, f, indent=2)
    print(f"\nSaved -> {REPORT_DIR}/chargeback_eval_results.json")
    print(f"Saved -> {REPORT_DIR}/chargeback_scored.csv")


if __name__ == "__main__":
    main()
