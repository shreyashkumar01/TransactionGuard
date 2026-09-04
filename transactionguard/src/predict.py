import argparse
import json
import os
import sys
import warnings

import joblib
import pandas as pd

warnings.filterwarnings("ignore", message="X does not have valid feature names")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import build_feature_matrix  # noqa: E402
from src.modeling import explain_prediction  # noqa: E402

MODEL_DIR = "models"

REVIEW_BAND = 0.03 

def load_model():
    entry = joblib.load(os.path.join(MODEL_DIR, "fraud_model.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)
    threshold = 0.5
    results_path = os.path.join("reports", "test_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            threshold = json.load(f)["selected_threshold"]
    background_path = os.path.join(MODEL_DIR, "shap_background.joblib")
    background = joblib.load(background_path) if os.path.exists(background_path) else None
    return entry, feature_cols, threshold, background


def score_dataframe(df: pd.DataFrame, entry, feature_cols, threshold: float, background=None, explain: bool = False) -> pd.DataFrame:
    X, _ = build_feature_matrix(df, fit_columns=feature_cols)
    model, scaler = entry["model"], entry["scaler"]
    X_use = scaler.transform(X) if scaler is not None else X
    scores = model.predict_proba(X_use)[:, 1]

    def action(p):
        if p >= threshold:
            return "manual_review"
        if p >= REVIEW_BAND:
            return "step_up_auth"
        return "approve"

    out = df.copy()
    out["fraud_probability"] = scores.round(4)
    out["recommended_action"] = [action(p) for p in scores]

    if explain:
        reasons = []
        for i in range(len(df)):
            if out["recommended_action"].iloc[i] == "approve":
                reasons.append("")  
                continue
            top = explain_prediction(entry, X.iloc[[i]], feature_cols, background=background, top_k=3)
            reasons.append("; ".join(f"{t['feature']}={t['value']} ({t['shap_contribution']:+.3f})" for t in top))
        out["top_risk_factors"] = reasons

    return out


def main():
    parser = argparse.ArgumentParser(description="Score transactions for fraud risk (defense-only).")
    parser.add_argument("--input", required=True, help="CSV of raw transactions to score")
    parser.add_argument("--output", default="scored_transactions.csv", help="Where to write scored output")
    parser.add_argument("--explain", action="store_true",
                         help="Add a top_risk_factors column (SHAP-based) for every flagged row, "
                              "so a reviewer sees WHY a transaction was flagged, not just a score.")
    args = parser.parse_args()

    entry, feature_cols, threshold, background = load_model()
    df = pd.read_csv(args.input)
    scored = score_dataframe(df, entry, feature_cols, threshold, background=background, explain=args.explain)
    scored.to_csv(args.output, index=False)

    counts = scored["recommended_action"].value_counts()
    print(f"Scored {len(scored):,} transactions using threshold {threshold:.2f}")
    print(counts.to_string())
    print(f"\nWrote -> {args.output}")


if __name__ == "__main__":
    main()
