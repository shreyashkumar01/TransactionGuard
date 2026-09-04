from __future__ import annotations

import argparse
from dataclasses import dataclass, field


ACTIONS_BY_SEVERITY = ["approve", "step_up_auth", "manual_review"] 


@dataclass
class RiskCase:
    case_id: str
    transaction_fraud_probability: float | None = None       
    transaction_action: str | None = None                     
    evidence_suspicion_score: float | None = None              
    evidence_action: str | None = None
    abuse_ring_suspicion_score: float | None = None             
    abuse_ring_action: str | None = None
    notes: list[str] = field(default_factory=list)


def combine_signals(case: RiskCase) -> dict:
    """
    Combines whichever signals are present on the case into one verdict.

    Rule (deliberately simple and auditable, not a black box): the case's
    overall action is the MOST SEVERE of the individual signal actions
    that are present. This is a conservative "escalate, don't average"
    policy — appropriate for a defense-only risk tool, where averaging
    away a single strong signal (e.g. tampered evidence on an otherwise
    unremarkable transaction) would be the wrong failure mode.
    """
    present_actions = [a for a in (case.transaction_action, case.evidence_action,
                                    case.abuse_ring_action) if a is not None]
    if not present_actions:
        return {
            "case_id": case.case_id,
            "overall_action": "insufficient_signal",
            "reason": "No transaction, evidence, or account-network signal was provided for this case.",
        }

    overall_action = max(present_actions, key=lambda a: ACTIONS_BY_SEVERITY.index(a))

    reasons = []
    if case.transaction_action is not None:
        reasons.append(
            f"transaction signal: {case.transaction_action} "
            f"(p_fraud={case.transaction_fraud_probability:.3f})"
            if case.transaction_fraud_probability is not None else
            f"transaction signal: {case.transaction_action}"
        )
    if case.evidence_action is not None:
        reasons.append(
            f"evidence signal: {case.evidence_action} "
            f"(suspicion={case.evidence_suspicion_score:.3f})"
            if case.evidence_suspicion_score is not None else
            f"evidence signal: {case.evidence_action}"
        )
    if case.abuse_ring_action is not None:
        reasons.append(
            f"account-network signal: {case.abuse_ring_action} "
            f"(suspicion={case.abuse_ring_suspicion_score:.3f})"
            if case.abuse_ring_suspicion_score is not None else
            f"account-network signal: {case.abuse_ring_action}"
        )

    return {
        "case_id": case.case_id,
        "overall_action": overall_action,
        "contributing_signals": reasons,
        "notes": case.notes,
        "disclaimer": "Decision support only. A human/authorized system confirms any final action.",
    }


def _demo():
    print("=== Case 1: ordinary transaction, no return evidence or network signal involved ===")
    case1 = RiskCase(case_id="CASE-001",
                      transaction_fraud_probability=0.12,
                      transaction_action="approve")
    print(combine_signals(case1))

    print("\n=== Case 2: transaction looked fine, but the return-evidence photo is suspicious ===")
    case2 = RiskCase(case_id="CASE-002",
                      transaction_fraud_probability=0.08,
                      transaction_action="approve",
                      evidence_suspicion_score=0.77,
                      evidence_action="manual_review",
                      notes=["Customer requested refund citing item damage; photo submitted."])
    print(combine_signals(case2))

    print("\n=== Case 3: transaction and evidence both look fine, but the account belongs to a "
          "flagged shared-infrastructure cluster ===")
    case3 = RiskCase(case_id="CASE-003",
                      transaction_fraud_probability=0.05,
                      transaction_action="approve",
                      abuse_ring_suspicion_score=0.68,
                      abuse_ring_action="manual_review",
                      notes=["Account is part of a 9-member cluster sharing a device, "
                             "signed up within a 2-day window."])
    print(combine_signals(case3))

    print("\n=== Case 4: multiple signals elevated at once — escalates, doesn't average down ===")
    case4 = RiskCase(case_id="CASE-004",
                      transaction_fraud_probability=0.42,
                      transaction_action="step_up_auth",
                      evidence_suspicion_score=0.81,
                      evidence_action="manual_review",
                      abuse_ring_suspicion_score=0.55,
                      abuse_ring_action="manual_review")
    print(combine_signals(case4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merchant risk scoring engine — combines multiple signals into one case verdict.")
    parser.add_argument("--demo", action="store_true", help="Run four illustrative example cases.")
    args = parser.parse_args()
    if args.demo:
        _demo()
    else:
        print("This module is meant to be imported (RiskCase, combine_signals) or run with --demo.")
