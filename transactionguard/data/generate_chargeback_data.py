import json

import numpy as np
import pandas as pd

RANDOM_SEED = 33
N_CASES = 500
GENUINE_ORDER_RATE = 0.55   
NOISE_STD = 0.4


def generate(n: int = N_CASES, seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    is_genuinely_legit = rng.binomial(1, GENUINE_ORDER_RATE, n)

    def _noisy_signal(base_rate_if_legit, base_rate_if_fraud):
        p = np.where(is_genuinely_legit == 1, base_rate_if_legit, base_rate_if_fraud)
        return rng.binomial(1, p)

    avs_match = _noisy_signal(0.85, 0.55)           
    cvv_match = _noisy_signal(0.90, 0.70)
    delivery_confirmed = _noisy_signal(0.75, 0.35)
    signature_or_otp_confirmed = _noisy_signal(0.55, 0.15)
    device_matches_prior_orders = _noisy_signal(0.70, 0.20)
    no_prior_dispute_history = _noisy_signal(0.80, 0.45)

    dispute_reasons = rng.choice(
        ["item_not_received", "unauthorized", "not_as_described", "duplicate_charge"],
        size=n, p=[0.35, 0.40, 0.20, 0.05],
    )
    order_amount = np.round(np.clip(rng.lognormal(mean=7.2, sigma=1.0, size=n), 100, 100_000), 2)

    win_probability = np.where(is_genuinely_legit == 1, 0.80, 0.10)
    win_probability = np.clip(win_probability + rng.normal(0, NOISE_STD, n) * 0.15, 0.02, 0.98)
    would_win_if_fought = rng.binomial(1, win_probability)

    df = pd.DataFrame({
        "case_id": [f"CB-{1000+i}" for i in range(n)],
        "dispute_reason": dispute_reasons,
        "order_amount_inr": order_amount,
        "avs_match": avs_match.astype(bool),
        "cvv_match": cvv_match.astype(bool),
        "delivery_confirmed": delivery_confirmed.astype(bool),
        "signature_or_otp_confirmed": signature_or_otp_confirmed.astype(bool),
        "device_matches_prior_orders": device_matches_prior_orders.astype(bool),
        "no_prior_dispute_history": no_prior_dispute_history.astype(bool),
        "is_genuinely_legit": is_genuinely_legit,    
        "would_win_if_fought": would_win_if_fought,         
    })
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "data/chargeback_disputes.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} disputes to {out_path}")
    print(f"Genuinely legit orders: {df['is_genuinely_legit'].mean():.1%}")
    print(f"Would win if fought: {df['would_win_if_fought'].mean():.1%}")
