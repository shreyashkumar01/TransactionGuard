import json
import random

import numpy as np
import pandas as pd

RANDOM_SEED = 5
N_INDEPENDENT_ACCOUNTS = 3000
N_LEGIT_CLUSTERS = 60          # e.g. families / small shops sharing infra
LEGIT_CLUSTER_SIZE_RANGE = (2, 4)
N_FRAUD_RINGS = 18
FRAUD_RING_SIZE_RANGE = (4, 12)

OUT_PATH = "data/abuse_ring_accounts.csv"


def _random_device_id(rng: random.Random) -> str:
    return f"DEV-{rng.randint(100000, 999999)}"


def _random_bank_account(rng: random.Random) -> str:
    return f"ACC-{rng.randint(10**8, 10**9 - 1)}"


def _random_ip_subnet(rng: random.Random) -> str:
    return f"{rng.randint(1,223)}.{rng.randint(0,255)}.{rng.randint(0,255)}.0/24"


def generate():
    rng = random.Random(RANDOM_SEED)
    np_rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    account_counter = 0

    for _ in range(N_INDEPENDENT_ACCOUNTS):
        account_counter += 1
        rows.append({
            "account_id": f"ACCT-{account_counter:06d}",
            "device_id": _random_device_id(rng),
            "linked_bank_account": _random_bank_account(rng),
            "ip_subnet": _random_ip_subnet(rng),
            "signup_day": int(np_rng.integers(0, 365)),
            "txn_count_first_7d": int(np_rng.poisson(2.0)),
            "cluster_type": "none",
            "cluster_id": None,
            "is_fraud_ring_member": 0,
        })


    for c in range(N_LEGIT_CLUSTERS):
        size = rng.randint(*LEGIT_CLUSTER_SIZE_RANGE)
        shared_device = _random_device_id(rng)
        is_fast_legit_cluster = rng.random() < 0.20
        base_day = int(np_rng.integers(0, 340))
        signup_spread = int(np_rng.integers(0, 3)) if is_fast_legit_cluster else int(np_rng.integers(4, 30))
        velocity_mean = 4.5 if is_fast_legit_cluster else 2.2   # a bit more active, still not ring-level
        for _ in range(size):
            account_counter += 1
            rows.append({
                "account_id": f"ACCT-{account_counter:06d}",
                "device_id": shared_device,
                "linked_bank_account": _random_bank_account(rng),   # NOT shared
                "ip_subnet": _random_ip_subnet(rng),
                "signup_day": base_day + int(np_rng.integers(0, max(signup_spread, 1))),
                "txn_count_first_7d": int(np_rng.poisson(velocity_mean)),
                "cluster_type": "legit_shared_device",
                "cluster_id": f"LEGIT-{c:03d}",
                "is_fraud_ring_member": 0,
            })

    for r in range(N_FRAUD_RINGS):
        size = rng.randint(*FRAUD_RING_SIZE_RANGE)
        shared_device = _random_device_id(rng)
        shared_bank_account = _random_bank_account(rng)
        base_day = int(np_rng.integers(0, 350))
        is_slow_burn_ring = rng.random() < 0.20
        signup_spread = int(np_rng.integers(6, 14)) if is_slow_burn_ring else 3
        velocity_mean = 4.0 if is_slow_burn_ring else 9.0
        for _ in range(size):
            account_counter += 1
            shares_device = rng.random() < 0.85
            shares_bank = rng.random() < 0.70
            rows.append({
                "account_id": f"ACCT-{account_counter:06d}",
                "device_id": shared_device if shares_device else _random_device_id(rng),
                "linked_bank_account": shared_bank_account if shares_bank else _random_bank_account(rng),
                "ip_subnet": _random_ip_subnet(rng),
                "signup_day": base_day + int(np_rng.integers(0, signup_spread)),
                "txn_count_first_7d": int(np_rng.poisson(velocity_mean)),
                "cluster_type": "fraud_ring",
                "cluster_id": f"RING-{r:03d}",
                "is_fraud_ring_member": 1,
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True) 
    df.to_csv(OUT_PATH, index=False)
    return df


if __name__ == "__main__":
    df = generate()
    print(f"Wrote {len(df):,} accounts to {OUT_PATH}")
    print(f"Fraud-ring members: {df['is_fraud_ring_member'].sum():,} "
          f"({df['is_fraud_ring_member'].mean():.2%})")
    print(df["cluster_type"].value_counts().to_string())
