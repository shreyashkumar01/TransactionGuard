from __future__ import annotations

import argparse
from dataclasses import dataclass, field

SIGNAL_WEIGHTS = {
    "avs_match": 0.15,                
    "cvv_match": 0.10,                 
    "delivery_confirmed": 0.30,        
    "signature_or_otp_confirmed": 0.20,  
    "device_matches_prior_orders": 0.15,  
    "no_prior_dispute_history": 0.10,  
}

STRONG_EVIDENCE_THRESHOLD = 0.65
WEAK_EVIDENCE_THRESHOLD = 0.30


@dataclass
class ChargebackCase:
    case_id: str
    dispute_reason: str   
    order_amount_inr: float
    evidence: dict = field(default_factory=dict)   
    notes: list = field(default_factory=list)


def score_case(case: ChargebackCase) -> dict:
    present_signals = {k: v for k, v in case.evidence.items() if k in SIGNAL_WEIGHTS}
    evidence_strength = sum(SIGNAL_WEIGHTS[k] for k, v in present_signals.items() if v)
    missing_signals = [k for k in SIGNAL_WEIGHTS if not present_signals.get(k, False)]

    if evidence_strength >= STRONG_EVIDENCE_THRESHOLD:
        recommendation = "recommend_fight"
        strength_label = "strong"
    elif evidence_strength <= WEAK_EVIDENCE_THRESHOLD:
        recommendation = "recommend_accept_liability"
        strength_label = "weak"
    else:
        recommendation = "recommend_manual_review"
        strength_label = "moderate"

    return {
        "case_id": case.case_id,
        "evidence_strength_score": round(float(evidence_strength), 3),
        "evidence_strength_label": strength_label,
        "recommended_action": recommendation,
        "present_evidence": [k for k, v in present_signals.items() if v],
        "missing_evidence": missing_signals,
        "disclaimer": "Draft recommendation only. A human reviews and submits any dispute "
                       "response — this module never auto-submits to a card network or "
                       "auto-issues a refund.",
    }


def draft_evidence_memo(case: ChargebackCase, scored: dict) -> str:
    """Drafts a plain-language evidence summary a merchant's ops team can
    review, edit, and attach to their own dispute submission. Deliberately
    does not reference specific card-network reason codes."""
    lines = [
        f"Chargeback evidence summary — Case {case.case_id}",
        f"Dispute reason (merchant-recorded): {case.dispute_reason}",
        f"Disputed amount: Rs.{case.order_amount_inr:,.2f}",
        f"Evidence strength: {scored['evidence_strength_label']} "
        f"({scored['evidence_strength_score']:.2f} / 1.00)",
        "",
        "Evidence available:",
    ]
    if scored["present_evidence"]:
        for sig in scored["present_evidence"]:
            lines.append(f"  - {sig.replace('_', ' ')}")
    else:
        lines.append("  - None of the standard evidence signals were available for this order.")

    lines.append("")
    lines.append("Evidence NOT available (gaps a reviewer should be aware of):")
    for sig in scored["missing_evidence"]:
        lines.append(f"  - {sig.replace('_', ' ')}")

    lines.append("")
    lines.append(f"Recommendation: {scored['recommended_action']}")
    if scored["recommended_action"] == "recommend_fight":
        lines.append("Rationale: multiple strong evidence signals are present; a dispute "
                      "response citing this evidence has a reasonable chance of success.")
    elif scored["recommended_action"] == "recommend_accept_liability":
        lines.append("Rationale: little to no corroborating evidence is available; fighting "
                      "this dispute is unlikely to succeed and risks the network-fee cost of "
                      "a failed representment on top of the original chargeback.")
    else:
        lines.append("Rationale: evidence is mixed — a human reviewer should weigh the "
                      "specific gaps above against the disputed amount before deciding "
                      "whether fighting this case is worth the effort.")

    lines.append("")
    lines.append(scored["disclaimer"])
    return "\n".join(lines)


def _demo():
    cases = [
        ChargebackCase(
            case_id="CB-001", dispute_reason="item_not_received", order_amount_inr=4200,
            evidence={"avs_match": True, "cvv_match": True, "delivery_confirmed": True,
                      "signature_or_otp_confirmed": True, "device_matches_prior_orders": True,
                      "no_prior_dispute_history": True},
        ),
        ChargebackCase(
            case_id="CB-002", dispute_reason="unauthorized", order_amount_inr=18500,
            evidence={"avs_match": False, "cvv_match": False, "delivery_confirmed": False,
                      "signature_or_otp_confirmed": False, "device_matches_prior_orders": False,
                      "no_prior_dispute_history": False},
        ),
        ChargebackCase(
            case_id="CB-003", dispute_reason="not_as_described", order_amount_inr=2100,
            evidence={"avs_match": True, "delivery_confirmed": True,
                      "signature_or_otp_confirmed": False, "device_matches_prior_orders": False},
        ),
    ]
    for case in cases:
        scored = score_case(case)
        print(draft_evidence_memo(case, scored))
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chargeback evidence responder (defense-only, draft-only).")
    parser.add_argument("--demo", action="store_true", help="Run three illustrative example cases.")
    args = parser.parse_args()
    if args.demo:
        _demo()
    else:
        print("This module is meant to be imported (ChargebackCase, score_case, "
              "draft_evidence_memo) or run with --demo.")
