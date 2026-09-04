import numpy as np
import pandas as pd

RANDOM_SEED = 7
N_TRANSACTIONS = 80_000
FRAUD_RATE = 0.0017         
N_COMPONENTS = 28


def generate(n: int = N_TRANSACTIONS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    time_seconds = np.sort(rng.integers(0, 172_800, n))  # two days, like the original
    amount = np.round(np.clip(rng.lognormal(mean=3.2, sigma=1.4, size=n), 0.5, 25_000), 2)

    is_fraud = rng.binomial(1, FRAUD_RATE, n)
    legit_factor = rng.normal(0, 1, n)
    fraud_factor = np.where(is_fraud == 1, rng.normal(2.2, 0.9, n), rng.normal(0, 0.6, n))

    loadings_legit = rng.normal(0, 1, N_COMPONENTS)
    loadings_fraud = rng.normal(0, 1, N_COMPONENTS)

    V = np.zeros((n, N_COMPONENTS))
    for j in range(N_COMPONENTS):
        V[:, j] = (
            loadings_legit[j] * legit_factor
            + loadings_fraud[j] * fraud_factor
            + rng.normal(0, 1.6, n)  
        )

    df = pd.DataFrame(V, columns=[f"V{i+1}" for i in range(N_COMPONENTS)])
    df.insert(0, "Time", time_seconds)
    df["Amount"] = amount
    df["Class"] = is_fraud
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "data/benchmark_eu_style.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(f"Fraud rate: {df['Class'].mean():.4%}  ({df['Class'].sum()} fraud rows)")
    print(df.head())
