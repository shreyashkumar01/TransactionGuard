"""
test_pipeline.py
-----------------
Lightweight sanity tests across all four modules — not a substitute for
the full evaluate*.py reports, but enough to catch a broken pipeline
before submission.

Run:
    python -m pytest tests/ -v
"""

import json
import os
import sys

import joblib
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import build_feature_matrix, load_dataset, TARGET_COL  # noqa: E402

DATA_PATH = "data/transactions.csv"
MODEL_PATH = "models/fraud_model.joblib"
FEATURE_COLS_PATH = "models/feature_columns.json"
RESULTS_PATH = "reports/test_results.json"


def test_data_exists_and_is_shaped_correctly():
    assert os.path.exists(DATA_PATH), "Run data/generate_data.py first"
    df = load_dataset(DATA_PATH)
    assert len(df) > 1000
    assert TARGET_COL in df.columns
    assert df[TARGET_COL].isin([0, 1]).all()


def test_fraud_rate_is_realistic():
    df = load_dataset(DATA_PATH)
    rate = df[TARGET_COL].mean()
    # real-world CNP fraud rates are typically well under 10%
    assert 0.001 < rate < 0.10, f"Fraud rate {rate:.4f} looks unrealistic"


def test_no_missing_values_in_features():
    df = load_dataset(DATA_PATH)
    X, _ = build_feature_matrix(df)
    assert not X.isnull().any().any()


def test_model_and_feature_columns_exist():
    assert os.path.exists(MODEL_PATH), "Run src/train.py first"
    assert os.path.exists(FEATURE_COLS_PATH), "Run src/train.py first"


def test_model_predicts_valid_probabilities():
    entry = joblib.load(MODEL_PATH)
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)
    df = load_dataset(DATA_PATH).head(50)
    X, _ = build_feature_matrix(df, fit_columns=feature_cols)
    model, scaler = entry["model"], entry["scaler"]
    X_use = scaler.transform(X) if scaler is not None else X
    probs = model.predict_proba(X_use)[:, 1]
    assert (probs >= 0).all() and (probs <= 1).all()


def test_higher_risk_signals_increase_predicted_probability():
    """A transaction with strong fraud signals should score higher than
    an otherwise-identical low-risk transaction. This is a monotonicity
    sanity check, not a full evaluation."""
    entry = joblib.load(MODEL_PATH)
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)

    base = {
        "hour": 13, "is_night": 0, "is_weekend": 0, "amount_inr": 1000,
        "transaction_type": "Bill Payment", "channel": "UPI",
        "account_age_days": 400, "is_new_account": 0,
        "avg_amount_last_30d": 950, "amount_to_avg_ratio": 1.05,
        "txn_count_last_24h": 1, "txn_count_last_1h": 0,
        "new_device_flag": 0, "new_beneficiary_flag": 0,
        "sim_recently_changed_flag": 0, "upi_pin_reset_last_7d": 0,
        "failed_otp_attempts_last_1h": 0, "login_location_mismatch_flag": 0,
    }
    risky = dict(base)
    risky.update({
        "hour": 2, "is_night": 1, "amount_inr": 45000,
        "transaction_type": "P2P Transfer", "account_age_days": 1,
        "is_new_account": 1, "amount_to_avg_ratio": 47.0,
        "new_device_flag": 1, "new_beneficiary_flag": 1,
        "sim_recently_changed_flag": 1, "upi_pin_reset_last_7d": 2,
        "failed_otp_attempts_last_1h": 3, "login_location_mismatch_flag": 1,
    })

    df = pd.DataFrame([base, risky])
    X, _ = build_feature_matrix(df, fit_columns=feature_cols)
    model, scaler = entry["model"], entry["scaler"]
    X_use = scaler.transform(X) if scaler is not None else X
    probs = model.predict_proba(X_use)[:, 1]
    assert probs[1] > probs[0], "Riskier transaction should score higher"


def test_reported_metrics_meet_a_minimum_bar():
    assert os.path.exists(RESULTS_PATH), "Run src/evaluate.py first"
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    assert results["pr_auc"] > 0.02  
    assert results["roc_auc"] > 0.6


def test_all_three_models_were_trained_and_saved():
    from src.modeling import MODEL_NAMES
    for name in MODEL_NAMES:
        path = f"models/fraud_model_{name}.joblib"
        assert os.path.exists(path), f"Missing {path} — run src/train.py"


def test_cross_validation_results_are_present_and_sane():
    summary_path = "models/training_summary.json"
    assert os.path.exists(summary_path), "Run src/train.py first"
    with open(summary_path) as f:
        summary = json.load(f)
    assert "cross_validation_results" in summary
    from src.modeling import MODEL_NAMES
    for name in MODEL_NAMES:
        cv = summary["cross_validation_results"][name]
        assert 0 <= cv["recall"]["mean"] <= 1
        assert len(cv["recall"]["folds"]) == 5


def test_model_comparison_report_covers_all_models():
    path = "reports/model_comparison.json"
    assert os.path.exists(path), "Run src/evaluate.py first"
    with open(path) as f:
        comp = json.load(f)
    from src.modeling import MODEL_NAMES
    for name in MODEL_NAMES:
        assert name in comp["per_model_results"]
        r = comp["per_model_results"][name]
        assert 0 <= r["precision"] <= 1
        assert 0 <= r["recall"] <= 1
        assert r["estimated_cost"] >= 0


def test_benchmark_harness_covers_both_datasets():
    path = "reports/benchmark_report.json"
    if not os.path.exists(path):
        pytest.skip("Run src/benchmark.py first (optional benchmark harness)")
    with open(path) as f:
        report = json.load(f)
    assert "indian_bfsi_upi" in report
    assert "eu_style_extreme" in report
    assert report["eu_style_extreme"]["fraud_rate"] < report["indian_bfsi_upi"]["fraud_rate"]


def test_shap_explanation_returns_real_contributions():
    """Regression test for the bug where SHAP was given the row itself as
    its own background and every contribution came out as exactly zero."""
    from src.modeling import explain_prediction
    entry = joblib.load(MODEL_PATH)
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = json.load(f)
    background_path = "models/shap_background.joblib"
    if not os.path.exists(background_path):
        pytest.skip("Run src/train.py first to generate the SHAP background sample")
    background = joblib.load(background_path)

    df = load_dataset(DATA_PATH).head(1)
    X, _ = build_feature_matrix(df, fit_columns=feature_cols)
    top = explain_prediction(entry, X, feature_cols, background=background, top_k=3)
    assert len(top) == 3
    assert any(abs(t["shap_contribution"]) > 1e-6 for t in top), \
        "All SHAP contributions were ~zero — background sample likely missing/degenerate"


def test_evidence_integrity_checker_runs_and_scores_in_range():
    from src.evidence_integrity import analyze_image
    sample_dir = "data/evidence_samples"
    if not os.path.exists(sample_dir):
        pytest.skip("Run data/generate_evidence_samples.py first")
    sample_file = sorted(os.listdir(sample_dir))[0]
    if not sample_file.lower().endswith((".jpg", ".jpeg", ".png")):
        pytest.skip("No image files found in data/evidence_samples")
    result = analyze_image(os.path.join(sample_dir, sample_file))
    assert 0.0 <= result["composite_suspicion_score"] <= 1.0
    assert result["recommended_action"] in ("accept", "manual_review")


def test_evidence_integrity_beats_random_guessing():
    path = "reports/evidence_eval_results.json"
    if not os.path.exists(path):
        pytest.skip("Run src/evaluate_evidence.py first")
    with open(path) as f:
        results = json.load(f)
    assert results["roc_auc"] > 0.5


def test_risk_engine_escalates_rather_than_averages():
    from src.risk_engine import RiskCase, combine_signals
    case = RiskCase(
        case_id="TEST-CASE",
        transaction_fraud_probability=0.05, transaction_action="approve",
        evidence_suspicion_score=0.8, evidence_action="manual_review",
    )
    result = combine_signals(case)
    assert result["overall_action"] == "manual_review", \
        "A strong evidence signal should escalate the case even if the transaction signal is clean"


def test_risk_engine_escalates_on_abuse_ring_signal_too():
    from src.risk_engine import RiskCase, combine_signals
    case = RiskCase(
        case_id="TEST-CASE-2",
        transaction_fraud_probability=0.03, transaction_action="approve",
        abuse_ring_suspicion_score=0.7, abuse_ring_action="manual_review",
    )
    result = combine_signals(case)
    assert result["overall_action"] == "manual_review", \
        "A strong account-network signal should escalate even with a clean transaction score"


def test_abuse_ring_sentinel_beats_random_guessing():
    path = "reports/abuse_ring_eval_results.json"
    if not os.path.exists(path):
        pytest.skip("Run src/evaluate_abuse_ring.py first")
    with open(path) as f:
        results = json.load(f)
    assert results["roc_auc"] > 0.5
    assert results["default_threshold_0_5"]["precision"] >= 0.5


def test_abuse_ring_graph_builds_correct_clusters():
    from src.abuse_ring_sentinel import build_account_graph, score_clusters
    import pandas as pd
    df = pd.DataFrame([
        {"account_id": "A1", "device_id": "DEV-1", "linked_bank_account": "ACC-1",
         "signup_day": 10, "txn_count_first_7d": 1},
        {"account_id": "A2", "device_id": "DEV-1", "linked_bank_account": "ACC-2",
         "signup_day": 10, "txn_count_first_7d": 1},
        {"account_id": "A3", "device_id": "DEV-2", "linked_bank_account": "ACC-3",
         "signup_day": 200, "txn_count_first_7d": 0},
    ])
    graph = build_account_graph(df)
    assert graph.has_edge("A1", "A2")
    assert not graph.has_edge("A1", "A3")
    scored = score_clusters(df, graph)
    assert len(scored) == 2  


def test_chargeback_responder_recommends_fight_on_strong_evidence():
    from src.chargeback_responder import ChargebackCase, score_case
    case = ChargebackCase(
        case_id="TEST-CB", dispute_reason="item_not_received", order_amount_inr=1000,
        evidence={"avs_match": True, "cvv_match": True, "delivery_confirmed": True,
                  "signature_or_otp_confirmed": True, "device_matches_prior_orders": True,
                  "no_prior_dispute_history": True},
    )
    result = score_case(case)
    assert result["recommended_action"] == "recommend_fight"
    assert result["evidence_strength_score"] == pytest.approx(1.0)


def test_chargeback_responder_recommends_accept_on_weak_evidence():
    from src.chargeback_responder import ChargebackCase, score_case
    case = ChargebackCase(
        case_id="TEST-CB-2", dispute_reason="unauthorized", order_amount_inr=1000,
        evidence={},
    )
    result = score_case(case)
    assert result["recommended_action"] == "recommend_accept_liability"


def test_chargeback_responder_missing_evidence_list_is_correct():
    """Regression test for the bug where signals present-but-False were
    incorrectly excluded from the 'missing evidence' list."""
    from src.chargeback_responder import ChargebackCase, score_case, SIGNAL_WEIGHTS
    case = ChargebackCase(
        case_id="TEST-CB-3", dispute_reason="unauthorized", order_amount_inr=1000,
        evidence={k: False for k in SIGNAL_WEIGHTS},
    )
    result = score_case(case)
    assert set(result["missing_evidence"]) == set(SIGNAL_WEIGHTS.keys()), \
        "All-False evidence should mean every signal is reported missing"


def test_chargeback_responder_beats_random_guessing():
    path = "reports/chargeback_eval_results.json"
    if not os.path.exists(path):
        pytest.skip("Run src/evaluate_chargeback.py first")
    with open(path) as f:
        results = json.load(f)
    assert results["roc_auc"] > 0.5


def test_all_four_track_directions_have_a_working_module():
    """Sanity check that every example direction from the track brief is
    represented by real, importable code — not just mentioned in docs."""
    from src.modeling import train_candidates  
    from src.evidence_integrity import analyze_image  
    from src.abuse_ring_sentinel import score_clusters  
    from src.chargeback_responder import score_case  
    assert callable(train_candidates)
    assert callable(analyze_image)
    assert callable(score_clusters)
    assert callable(score_case)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
