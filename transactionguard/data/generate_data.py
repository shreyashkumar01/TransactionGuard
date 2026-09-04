import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_TRANSACTIONS = 60_000
BASE_FRAUD_RATE = 0.021       
RANDOM_FRAUD_SHARE = 0.15        
NOISE_STD = 0.35                 

TRANSACTION_TYPES = [
    "P2P Transfer", "Merchant Payment", "Bill Payment", "Loan EMI Payment",
    "Wallet Top-up", "Cash Withdrawal (UPI ATM)",
]
TXN_TYPE_RISK = {
    "P2P Transfer": 1.0, "Merchant Payment": 0.1, "Bill Payment": -0.6,
    "Loan EMI Payment": -0.5, "Wallet Top-up": 0.2, "Cash Withdrawal (UPI ATM)": 0.9,
}
CHANNELS = ["UPI", "IMPS", "NetBanking", "Card"]
CHANNEL_RISK = {"UPI": 0.2, "IMPS": 0.3, "NetBanking": -0.15, "Card": -0.1}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n: int = N_TRANSACTIONS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    hour = rng.integers(0, 24, n)
    is_night = ((hour >= 0) & (hour <= 5)).astype(int)
    is_weekend = rng.integers(0, 7, n) >= 5
    is_weekend = is_weekend.astype(int)

    account_age_days = np.round(rng.exponential(scale=220, size=n)).clip(0, 3000)
    is_new_account = (account_age_days < 3).astype(int)   # freshly opened / mule accounts

    transaction_type = rng.choice(TRANSACTION_TYPES, size=n,
                                   p=[0.30, 0.28, 0.16, 0.10, 0.10, 0.06])
    channel = rng.choice(CHANNELS, size=n, p=[0.52, 0.20, 0.18, 0.10])

    type_amount_mult = np.array([
        1.3 if t == "P2P Transfer" else 0.9 if t == "Cash Withdrawal (UPI ATM)" else
        0.6 if t in ("Bill Payment", "Loan EMI Payment") else 1.0
        for t in transaction_type
    ])
    amount = np.round(rng.lognormal(mean=6.6, sigma=0.9, size=n) * type_amount_mult, 2)
    amount = np.clip(amount, 49, 500_000)

    avg_amount_last_30d = np.round(amount * rng.uniform(0.5, 1.3, n) + rng.normal(0, 200, n), 2)
    avg_amount_last_30d = np.clip(avg_amount_last_30d, 49, None)
    amount_to_avg_ratio = amount / avg_amount_last_30d

    txn_count_last_24h = rng.poisson(lam=1.3, size=n)
    txn_count_last_1h = np.minimum(txn_count_last_24h, rng.poisson(lam=0.4, size=n))

    new_device_flag = rng.binomial(1, 0.12, n)
    new_beneficiary_flag = rng.binomial(1, 0.10, n)          
    sim_recently_changed_flag = rng.binomial(1, 0.03, n)      
    upi_pin_reset_last_7d = rng.poisson(lam=0.08, size=n)     
    failed_otp_attempts_last_1h = rng.poisson(lam=0.15, size=n)
    login_location_mismatch_flag = rng.binomial(1, 0.08, n)   

    type_risk = np.array([TXN_TYPE_RISK[t] for t in transaction_type])
    channel_risk = np.array([CHANNEL_RISK[c] for c in channel])

    z = (
        1.7 * is_night
        + 1.9 * new_device_flag
        + 2.1 * new_beneficiary_flag
        + 2.6 * sim_recently_changed_flag
        + 1.0 * np.log1p(upi_pin_reset_last_7d)
        + 1.3 * np.log1p(amount_to_avg_ratio.clip(0, 20))
        + 0.9 * np.log1p(txn_count_last_1h)
        + 1.4 * np.log1p(failed_otp_attempts_last_1h)
        + 1.0 * login_location_mismatch_flag
        + 0.6 * type_risk
        + 0.5 * channel_risk
        + 1.2 * is_new_account
        - 0.9 * np.log1p(account_age_days / 30)
        + rng.normal(0, NOISE_STD, n)        
    )

    p_fraud_signal = _sigmoid(z - 6.4)       
    signal_driven = rng.binomial(1, p_fraud_signal)

    random_driven = rng.binomial(1, BASE_FRAUD_RATE * RANDOM_FRAUD_SHARE, n)

    is_fraud = np.maximum(signal_driven, random_driven)

    current_rate = is_fraud.mean()
    if current_rate > BASE_FRAUD_RATE:
        keep_prob = BASE_FRAUD_RATE / current_rate
        thinning = rng.binomial(1, keep_prob, n)
        is_fraud = is_fraud * thinning

    df = pd.DataFrame({
        "txn_id": [f"TXN{100000+i}" for i in range(n)],
        "hour": hour,
        "is_night": is_night,
        "is_weekend": is_weekend,
        "amount_inr": amount,
        "transaction_type": transaction_type,
        "channel": channel,
        "account_age_days": account_age_days,
        "is_new_account": is_new_account,
        "avg_amount_last_30d": avg_amount_last_30d,
        "amount_to_avg_ratio": np.round(amount_to_avg_ratio, 3),
        "txn_count_last_24h": txn_count_last_24h,
        "txn_count_last_1h": txn_count_last_1h,
        "new_device_flag": new_device_flag,
        "new_beneficiary_flag": new_beneficiary_flag,
        "sim_recently_changed_flag": sim_recently_changed_flag,
        "upi_pin_reset_last_7d": upi_pin_reset_last_7d,
        "failed_otp_attempts_last_1h": failed_otp_attempts_last_1h,
        "login_location_mismatch_flag": login_location_mismatch_flag,
        "is_fraud": is_fraud.astype(int),
    })
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "data/transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(f"Fraud rate: {df['is_fraud'].mean():.4%}")
    print(df.head())
