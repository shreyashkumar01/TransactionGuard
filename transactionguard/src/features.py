import pandas as pd

CATEGORICAL_COLS = ["transaction_type", "channel"]

NUMERIC_COLS = [
    "hour", "is_night", "is_weekend", "amount_inr", "account_age_days",
    "is_new_account", "avg_amount_last_30d", "amount_to_avg_ratio",
    "txn_count_last_24h", "txn_count_last_1h", "new_device_flag",
    "new_beneficiary_flag", "sim_recently_changed_flag", "upi_pin_reset_last_7d",
    "failed_otp_attempts_last_1h", "login_location_mismatch_flag",
]

TARGET_COL = "is_fraud"
ID_COL = "txn_id"


def build_feature_matrix(df: pd.DataFrame, fit_columns: list[str] | None = None):
    """
    One-hot encodes categoricals and returns (X, feature_names).

    If `fit_columns` is provided (the column layout learned at train
    time), the output is reindexed to exactly that layout so inference
    on new data always matches what the model was trained on, even if a
    category is missing/unseen in the new batch.
    """
    encoded = pd.get_dummies(df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)
    X = pd.concat([df[NUMERIC_COLS].reset_index(drop=True),
                   encoded.reset_index(drop=True)], axis=1)

    if fit_columns is not None:
        X = X.reindex(columns=fit_columns, fill_value=0)

    return X, list(X.columns)


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
