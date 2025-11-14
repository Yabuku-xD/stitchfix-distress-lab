"""Model training sandbox: logistic baseline + tree-ish experiments."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import REPORTS_DIR


@dataclass
class FoldResult:
    metrics: Dict[str, float]
    feature_importances: Dict[str, float]
    classification_report: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray


@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, float]
    feature_importances: Dict[str, float]
    classification_report: str
    fold_metrics: List[Dict[str, float]]
    fold_reports: Dict[str, str]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray


def _evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics = {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(y_true.unique()) > 1 else float("nan"),
    }
    if len(y_true.unique()) > 1:
        metrics["avg_precision"] = average_precision_score(y_true, y_prob)
    else:
        metrics["avg_precision"] = float("nan")
    metrics["positive_rate"] = float(y_true.mean())
    metrics["predicted_positive_rate"] = float(y_pred.mean())
    return metrics


def train_logistic_regression(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
) -> ModelResult:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    # TODO: add Platt scaling once we actually have both classes in the holdouts.
    pipeline.fit(train_X, train_y)
    y_prob = pipeline.predict_proba(test_X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = _evaluate_predictions(test_y, y_pred, y_prob)
    report = classification_report(test_y, y_pred, zero_division=0)
    clf = pipeline.named_steps["clf"]
    feature_importances = {
        feature: float(coef)
        for feature, coef in zip(train_X.columns, clf.coef_[0])
    }
    return FoldResult(
        metrics=metrics,
        feature_importances=feature_importances,
        classification_report=report,
        y_true=test_y.to_numpy(),
        y_pred=y_pred,
        y_prob=y_prob,
    )


def train_random_forest(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
) -> ModelResult:
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(train_X, train_y)
    y_prob = clf.predict_proba(test_X)[:, 1]
    y_pred = clf.predict(test_X)
    metrics = _evaluate_predictions(test_y, y_pred, y_prob)
    report = classification_report(test_y, y_pred, zero_division=0)
    feature_importances = {
        feature: float(importance)
        for feature, importance in zip(train_X.columns, clf.feature_importances_)
    }
    return FoldResult(
        metrics=metrics,
        feature_importances=feature_importances,
        classification_report=report,
        y_true=test_y.to_numpy(),
        y_pred=y_pred,
        y_prob=y_prob,
    )


def train_hist_gradient_boosting(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
) -> FoldResult:
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=400,
        min_samples_leaf=2,
        random_state=42,
    )
    clf.fit(train_X, train_y)
    y_prob = clf.predict_proba(test_X)[:, 1]
    y_pred = clf.predict(test_X)
    metrics = _evaluate_predictions(test_y, y_pred, y_prob)
    report = classification_report(test_y, y_pred, zero_division=0)
    perm = permutation_importance(
        clf,
        test_X,
        test_y,
        n_repeats=15,
        random_state=42,
        scoring="f1",
    )
    feature_importances = {
        feature: float(importance)
        for feature, importance in zip(train_X.columns, perm.importances_mean)
    }
    return FoldResult(
        metrics=metrics,
        feature_importances=feature_importances,
        classification_report=report,
        y_true=test_y.to_numpy(),
        y_pred=y_pred,
        y_prob=y_prob,
    )


def _aggregate_results(name: str, fold_results: List[FoldResult]) -> ModelResult:
    metrics_frame = pd.DataFrame([fold.metrics for fold in fold_results])
    avg_metrics = metrics_frame.mean().to_dict()
    importances_frame = pd.DataFrame([fold.feature_importances for fold in fold_results])
    avg_importances = importances_frame.mean().to_dict()
    fold_reports = {
        f"fold_{idx + 1}": fold.classification_report for idx, fold in enumerate(fold_results)
    }
    combined_report = "\n\n".join(
        f"Fold {idx + 1}:\n{fold.classification_report}"
        for idx, fold in enumerate(fold_results)
    )
    y_true = np.concatenate([fold.y_true for fold in fold_results])
    y_pred = np.concatenate([fold.y_pred for fold in fold_results])
    y_prob = np.concatenate([fold.y_prob for fold in fold_results])

    return ModelResult(
        name=name,
        metrics=avg_metrics,
        feature_importances=avg_importances,
        classification_report=combined_report,
        fold_metrics=metrics_frame.to_dict("records"),
        fold_reports=fold_reports,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def _walk_forward_slices(
    length: int,
    min_train: int,
    test_window: int,
    step: int,
) -> List[Tuple[slice, slice]]:
    slices: List[Tuple[slice, slice]] = []
    start = max(min_train, 1)
    while start + test_window <= length:
        slices.append((slice(0, start), slice(start, start + test_window)))
        start += max(step, 1)

    if not slices and length > 1:
        split = max(1, length - test_window)
        slices.append((slice(0, split), slice(split, length)))
    elif not slices:
        slices.append((slice(0, 1), slice(1, 1)))

    return slices


def run_model_suite(
    dataset: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "distress_flag",
    *,
    min_train: int = 8,
    test_window: int = 2,
    step: int = 1,
) -> Dict[str, ModelResult]:
    """Train both baseline and non-linear models with walk-forward evaluation."""

    clean = (
        dataset.dropna(subset=feature_cols + [target_col])
        .sort_values("period_end")
        .reset_index(drop=True)
    )
    if clean.empty:
        raise ValueError("Need at least two classes to train the model.")

    slices = _walk_forward_slices(len(clean), min_train, test_window, step)

    results = {}
    models = {
        "logistic_regression": train_logistic_regression,
        "random_forest": train_random_forest,
        "hist_gradient_boosting": train_hist_gradient_boosting,
    }

    for model_name, trainer in models.items():
        fold_results: List[FoldResult] = []
        for train_slice, test_slice in slices:
            train_df = clean.iloc[train_slice]
            test_df = clean.iloc[test_slice]
            if train_df.empty or test_df.empty:
                continue
            train_y = train_df[target_col]
            test_y = test_df[target_col]
            if train_y.nunique() < 2:
                continue
            train_X = train_df[feature_cols]
            test_X = test_df[feature_cols]
            fold_results.append(trainer(train_X, train_y, test_X, test_y))

        if not fold_results:
            raise ValueError(
                f"Unable to create valid training folds for model {model_name}. "
                "Try adjusting walk-forward parameters or lookahead length."
            )

        results[model_name] = _aggregate_results(model_name, fold_results)

    metrics_payload = {
        name: {
            "aggregate_metrics": res.metrics,
            "fold_metrics": res.fold_metrics,
            "feature_importances": res.feature_importances,
            "fold_reports": res.fold_reports,
            "classification_report": res.classification_report,
        }
        for name, res in results.items()
    }

    output_path = REPORTS_DIR / "model_metrics.json"
    output_path.write_text(
        json.dumps(_sanitize_for_json(metrics_payload), indent=2),
        encoding="utf-8",
    )
    return results


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {key: _sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


