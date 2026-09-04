import json
import os
import warnings

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.features import build_feature_matrix
from src.modeling import explain_prediction

warnings.filterwarnings("ignore", message="X does not have valid feature names")

app = Flask(__name__)

MODEL_DIR = "models"
_entry = joblib.load(os.path.join(MODEL_DIR, "fraud_model.joblib"))
with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    _feature_cols = json.load(f)

_background_path = os.path.join(MODEL_DIR, "shap_background.joblib")
_background = joblib.load(_background_path) if os.path.exists(_background_path) else None

_threshold = 0.5
_results_path = os.path.join("reports", "test_results.json")
if os.path.exists(_results_path):
    with open(_results_path) as f:
        _threshold = json.load(f)["selected_threshold"]

REVIEW_BAND = 0.03


@app.get("/")
def index():
    return jsonify({
        "service": "TransactionGuard AI Risk Manager",
        "status": "ok",
        "endpoints": {
            "health": "GET /health",
            "score_transaction": "POST /score?explain=true",
            "verify_evidence": "POST /verify_evidence",
            "score_chargeback": "POST /score_chargeback",
        },
        "note": "Defense-only decision support API. No irreversible actions are taken here.",
    })


def _recommend(p: float) -> str:
    if p >= _threshold:
        return "manual_review"
    if p >= REVIEW_BAND:
        return "step_up_auth"
    return "approve"


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_threshold": _threshold,
        "modules": {
            "upi_account_takeover_detector": "loaded (primary — POST /score)",
            "evidence_integrity_checker": "loaded (POST /verify_evidence)",
            "abuse_ring_sentinel": "available (run as CLI: src/abuse_ring_sentinel.py)",
            "chargeback_responder": "available (POST /score_chargeback)",
        },
        "note": "Defense-only fraud risk scoring API. No irreversible actions are taken here.",
    })


@app.post("/verify_evidence")
def verify_evidence():
    """Accepts a JSON body {"path": "..."} pointing to an already-uploaded
    return-evidence photo on disk, and returns a tampering-suspicion score
    from src/evidence_integrity.py — the Return Fraud with digital evidence
    direction. Read-only / defense-only: it never deletes or modifies the
    photo, and never auto-rejects a claim, only ever recommending 'accept'
    or 'manual_review'."""
    from src.evidence_integrity import analyze_image

    payload = request.get_json(force=True)
    path = payload.get("path")
    if not path or not os.path.exists(path):
        return jsonify({"error": "provide a valid 'path' to an existing image file"}), 400

    try:
        result = analyze_image(path)
    except Exception as e:
        return jsonify({"error": f"could not analyze image: {e}"}), 400

    return jsonify({
        "composite_suspicion_score": result["composite_suspicion_score"],
        "recommended_action": result["recommended_action"],
        "reasons": result["reasons"],
        "disclaimer": "Decision support only. A human must review before any claim is accepted or denied.",
    })


@app.post("/score_chargeback")
def score_chargeback_case():
    """Drafts a chargeback evidence memo + fight/accept recommendation
    (src/chargeback_responder.py) — the Chargeback Evidence Responder
    direction. Never auto-submits a dispute response or auto-issues a
    refund; 'recommend_accept_liability' is presented as an equally valid
    outcome, not a fallback."""
    from src.chargeback_responder import ChargebackCase, draft_evidence_memo, score_case

    payload = request.get_json(force=True)
    try:
        case = ChargebackCase(
            case_id=payload["case_id"], dispute_reason=payload.get("dispute_reason", "unspecified"),
            order_amount_inr=float(payload.get("order_amount_inr", 0)),
            evidence=payload.get("evidence", {}),
        )
    except KeyError as e:
        return jsonify({"error": f"missing required field: {e}"}), 400

    scored = score_case(case)
    memo = draft_evidence_memo(case, scored)
    return jsonify({**scored, "evidence_memo": memo})


@app.post("/score")
def score_transaction():
    """The primary, BFSI-headline endpoint: scores a UPI/net-banking
    transaction for account-takeover risk using models/ (src/train.py /
    src/evaluate.py). Expects a JSON body with the same raw fields as
    data/transactions.csv (minus is_fraud). Optional query param
    ?explain=true adds a SHAP-based 'why' for flagged transactions — the
    kind of compliance-facing explanation a human reviewer needs before
    acting on a risk score, not just the bare number."""
    payload = request.get_json(force=True)
    df = pd.DataFrame([payload])

    try:
        X, _ = build_feature_matrix(df, fit_columns=_feature_cols)
    except KeyError as e:
        return jsonify({"error": f"missing required field: {e}"}), 400

    model, scaler = _entry["model"], _entry["scaler"]
    X_use = scaler.transform(X) if scaler is not None else X
    p = float(model.predict_proba(X_use)[:, 1][0])
    action = _recommend(p)

    response = {
        "fraud_probability": round(p, 4),
        "recommended_action": action,
        "threshold_used": _threshold,
        "disclaimer": "Decision support only. A human/authorized system must confirm any block.",
    }

    if request.args.get("explain", "").lower() == "true" and action != "approve":
        top = explain_prediction(_entry, X, _feature_cols, background=_background, top_k=3)
        response["top_risk_factors"] = top

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
