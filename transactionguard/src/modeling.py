from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_SEED = 42
CV_FOLDS = 5


def build_logreg():
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)


def build_random_forest():
    return RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED,
    )


def build_xgb(pos: int, neg: int):
    return XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg / max(pos, 1),
        eval_metric="aucpr", random_state=RANDOM_SEED, n_jobs=-1,
    )


MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]


def _fit_one(name: str, X_train, y_train):
    """Fits one named model, returns {'model':..., 'scaler':...}."""
    if name == "logistic_regression":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        model = build_logreg()
        model.fit(X_scaled, y_train)
        return {"model": model, "scaler": scaler}

    if name == "random_forest":
        model = build_random_forest()
        model.fit(X_train, y_train)
        return {"model": model, "scaler": None}

    if name == "xgboost":
        pos = int(np.sum(y_train))
        neg = int(len(y_train) - pos)
        model = build_xgb(pos, neg)
        model.fit(X_train, y_train)
        return {"model": model, "scaler": None}

    raise ValueError(f"Unknown model name: {name}")


def score(entry: dict, X) -> np.ndarray:
    """Returns fraud-probability scores for a fitted model entry."""
    model, scaler = entry["model"], entry["scaler"]
    X_use = scaler.transform(X) if scaler is not None else X
    return model.predict_proba(X_use)[:, 1]


def train_candidates(X_train, y_train) -> dict:
    """Fits all three candidate models on the full training split."""
    return {name: _fit_one(name, X_train, y_train) for name in MODEL_NAMES}



def cross_validate_candidates(X_train, y_train, k: int = CV_FOLDS) -> dict:
    """
    Returns, for each candidate model:
        {"recall": {"mean":..., "std":..., "folds": [...]},
         "precision": {...}, "f1": {...}, "pr_auc": {...}}

    Uses a default 0.5 probability cut for precision/recall/F1 during CV
    (the real, cost-tuned threshold is chosen later on the validation
    split in evaluate.py) — CV here is for MODEL SELECTION / sanity
    checking, not for picking the final operating threshold.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    X_arr = X_train.values if hasattr(X_train, "values") else X_train
    y_arr = np.asarray(y_train)

    results = {name: {"recall": [], "precision": [], "f1": [], "pr_auc": []} for name in MODEL_NAMES}

    for fold_idx, (tr_idx, ho_idx) in enumerate(skf.split(X_arr, y_arr)):
        X_tr, X_ho = X_arr[tr_idx], X_arr[ho_idx]
        y_tr, y_ho = y_arr[tr_idx], y_arr[ho_idx]

        for name in MODEL_NAMES:
            entry = _fit_one(name, X_tr, y_tr)
            p_ho = score(entry, X_ho)
            preds = (p_ho >= 0.5).astype(int)

            results[name]["recall"].append(recall_score(y_ho, preds, zero_division=0))
            results[name]["precision"].append(precision_score(y_ho, preds, zero_division=0))
            results[name]["f1"].append(f1_score(y_ho, preds, zero_division=0))
            results[name]["pr_auc"].append(average_precision_score(y_ho, p_ho))

    summary = {}
    for name in MODEL_NAMES:
        summary[name] = {
            metric: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "folds": [round(float(v), 4) for v in vals],
            }
            for metric, vals in results[name].items()
        }
    return summary



def expected_cost_at_threshold(y_true, amounts, p_scores, threshold, fn_fixed_cost, fp_cost):
    preds = (p_scores >= threshold).astype(int)
    fn_mask = (preds == 0) & (y_true == 1)
    fp_mask = (preds == 1) & (y_true == 0)
    fn_cost_total = (np.asarray(amounts)[fn_mask] + fn_fixed_cost).sum()
    fp_cost_total = fp_mask.sum() * fp_cost
    return fn_cost_total + fp_cost_total, int(fn_mask.sum()), int(fp_mask.sum())


def tune_threshold(y_val, amounts_val, p_val, fn_fixed_cost, fp_cost, grid=None):
    """Sweeps thresholds and returns the cost-minimizing one, plus the full curve."""
    if grid is None:
        grid = np.arange(0.01, 0.99, 0.01)
    costs = [expected_cost_at_threshold(y_val, amounts_val, p_val, t, fn_fixed_cost, fp_cost)[0]
             for t in grid]
    best_idx = int(np.argmin(costs))
    return float(grid[best_idx]), float(costs[best_idx]), grid, costs


def evaluate_at_threshold(y_true, amounts, p_scores, threshold, fn_fixed_cost, fp_cost) -> dict:
    """Full metric bundle at a fixed threshold — used for final test-set reporting."""
    preds = (p_scores >= threshold).astype(int)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    pr_auc = average_precision_score(y_true, p_scores)
    roc_auc = roc_auc_score(y_true, p_scores) if len(np.unique(y_true)) > 1 else float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    cost, fn_count, fp_count = expected_cost_at_threshold(
        y_true, amounts, p_scores, threshold, fn_fixed_cost, fp_cost
    )
    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_negative_count": fn_count,
        "false_positive_count": fp_count,
        "estimated_cost": float(cost),
    }


def naive_baseline_costs(y_true, amounts, fn_fixed_cost, fp_cost, high_amount_cutoff) -> dict:
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts)

    cost_block_nothing = (amounts[y_true == 1] + fn_fixed_cost).sum()

    blocked = amounts >= high_amount_cutoff
    fn_mask_b = (~blocked) & (y_true == 1)
    fp_mask_b = blocked & (y_true == 0)
    cost_block_high = (amounts[fn_mask_b] + fn_fixed_cost).sum() + fp_mask_b.sum() * fp_cost

    return {
        "block_nothing": float(cost_block_nothing),
        f"block_above_{high_amount_cutoff}": float(cost_block_high),
    }


def explain_prediction(entry: dict, X_row, feature_names: list[str], background=None, top_k: int = 5) -> list[dict]:
    """
    Returns the top_k features pushing this single row's prediction toward
    'fraud', as [{"feature": ..., "value": ..., "shap_contribution": ...}, ...],
    sorted by |contribution| descending.

    `background` is a small reference sample (unscaled, same columns as
    X_row) used to define "what a typical transaction looks like" for the
    linear explainer. Without a real background sample, SHAP has nothing to
    contrast the row against and every contribution comes out as zero —
    this is a common, easy-to-miss mistake, so it's a required concept here
    even though the parameter itself is optional (falls back to a small
    synthetic background of zeros, which still works but is less meaningful
    than real transaction data).

    Uses shap.LinearExplainer for logistic regression (exact, fast) and
    shap.TreeExplainer for random forest / xgboost (exact for trees, no
    background sample needed).
    """
    import shap  

    model, scaler = entry["model"], entry["scaler"]
    X_arr = X_row.values if hasattr(X_row, "values") else np.asarray(X_row)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    if scaler is not None:
        X_use = scaler.transform(X_arr)
        if background is not None:
            bg_arr = background.values if hasattr(background, "values") else np.asarray(background)
            bg_use = scaler.transform(bg_arr)
        else:
            bg_use = np.zeros((1, X_use.shape[1]))
        masker = shap.maskers.Independent(bg_use, max_samples=bg_use.shape[0])
        explainer = shap.LinearExplainer(model, masker)
        shap_values = explainer.shap_values(X_use)
    else:
        X_use = X_arr
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(X_use)
        if isinstance(raw, list):
            shap_values = raw[1]
        else:
            shap_values = raw[..., 1] if np.ndim(raw) == 3 else raw

    values = np.asarray(shap_values).reshape(-1)
    order = np.argsort(-np.abs(values))[:top_k]

    return [
        {
            "feature": feature_names[i],
            "value": float(X_arr[0, i]) if np.isscalar(X_arr[0, i]) or isinstance(X_arr[0, i], (int, float, np.floating)) else str(X_arr[0, i]),
            "shap_contribution": round(float(values[i]), 4),
        }
        for i in order
    ]
