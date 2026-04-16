"""
Isolation Forest anomaly detector with standardized outputs (AD-08 / AD-09).

Convention: anomaly_score is higher for more anomalous points (sklearn IF
decision_function is inverted).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support


def anomaly_scores_isolation_forest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Higher score = more anomalous."""
    return -model.decision_function(X)


@dataclass
class DetectorResult:
    anomaly_score: np.ndarray
    pred_label: np.ndarray
    threshold: float
    method: str = "isolation_forest"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_score": self.anomaly_score,
            "pred_label": self.pred_label,
            "threshold": float(self.threshold),
            "method": self.method,
        }


def fit_isolation_forest(
    X_train: np.ndarray,
    *,
    n_estimators: int = 200,
    max_samples: Union[float, str] = "auto",
    max_features: float = 1.0,
    contamination: float = 0.001,
    random_state: int = 42,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def sweep_thresholds_val(
    model: IsolationForest,
    X_val: np.ndarray,
    y_val: np.ndarray,
    quantile_grid: Optional[np.ndarray] = None,
) -> Tuple[float, pd.DataFrame]:
    """Pick threshold on validation by maximizing F1 over score quantiles."""
    if quantile_grid is None:
        quantile_grid = np.linspace(0.90, 0.999, 40)
    s = anomaly_scores_isolation_forest(model, X_val)
    rows = []
    best_f1 = -1.0
    best_thr = float(np.quantile(s, 0.99))
    for q in quantile_grid:
        thr = float(np.quantile(s, q))
        pred = (s >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_val, pred, average="binary", zero_division=0
        )
        tn = int(np.sum((pred == 0) & (y_val == 0)))
        fp = int(np.sum((pred == 1) & (y_val == 0)))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        rows.append(
            {
                "quantile": q,
                "threshold": thr,
                "precision": p,
                "recall": r,
                "f1": f1,
                "fpr": fpr,
            }
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, pd.DataFrame(rows)


def predict_with_threshold(
    model: IsolationForest, X: np.ndarray, threshold: float
) -> DetectorResult:
    scores = anomaly_scores_isolation_forest(model, X)
    pred = (scores >= threshold).astype(int)
    return DetectorResult(
        anomaly_score=scores, pred_label=pred, threshold=float(threshold)
    )


def metrics_at_threshold(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "fpr": float(fpr),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }
