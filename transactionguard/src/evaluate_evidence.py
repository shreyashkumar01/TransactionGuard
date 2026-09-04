import json
import os
import sys

import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_score,
    recall_score, roc_auc_score,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evidence_integrity import MANUAL_REVIEW_THRESHOLD, analyze_image  # noqa: E402

SAMPLES_DIR = "data/evidence_samples"
REPORT_DIR = "reports"


def main():
    labels_path = os.path.join(SAMPLES_DIR, "labels.json")
    if not os.path.exists(labels_path):
        raise SystemExit("Run `python data/generate_evidence_samples.py` first.")
    with open(labels_path) as f:
        labels = json.load(f)

    rows = []
    for fname, label in labels.items():
        path = os.path.join(SAMPLES_DIR, fname)
        result = analyze_image(path)
        rows.append({
            "file": fname,
            "true_label": label,   # 0 = authentic, 1 = tampered
            "composite_suspicion_score": result["composite_suspicion_score"],
            "predicted_flag": int(result["recommended_action"] == "manual_review"),
        })

    df = pd.DataFrame(rows)
    y_true = df["true_label"].values
    y_score = df["composite_suspicion_score"].values
    y_pred = df["predicted_flag"].values

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"Evaluated {len(df)} images ({int(y_true.sum())} tampered, "
          f"{int((1 - y_true).sum())} authentic)")
    print(f"Manual-review threshold: {MANUAL_REVIEW_THRESHOLD}")
    print(f"Precision: {precision:.3f}   Recall: {recall:.3f}")
    print(f"PR-AUC: {pr_auc:.3f}  (random baseline = {y_true.mean():.3f})")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"Confusion matrix -> TN:{tn} FP:{fp} FN:{fn} TP:{tp}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    df.to_csv(os.path.join(REPORT_DIR, "evidence_scored_labelled.csv"), index=False)
    with open(os.path.join(REPORT_DIR, "evidence_eval_results.json"), "w") as f:
        json.dump({
            "n_images": len(df), "n_tampered": int(y_true.sum()), "n_authentic": int((1 - y_true).sum()),
            "threshold": MANUAL_REVIEW_THRESHOLD, "precision": float(precision), "recall": float(recall),
            "pr_auc": float(pr_auc), "roc_auc": float(roc_auc),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }, f, indent=2)
    print(f"\nSaved -> {REPORT_DIR}/evidence_eval_results.json")
    print(f"Saved -> {REPORT_DIR}/evidence_scored_labelled.csv")


if __name__ == "__main__":
    main()
