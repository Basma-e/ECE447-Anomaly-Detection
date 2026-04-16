"""
data_pipeline.py
Shared preprocessing pipeline for ECE447 Anomaly Detection Project.
All members import from this file to ensure consistent data splits.

Member A saves ``artifacts/*.npy`` from ``notebooks/memberA/memberA_eda_statistical.ipynb``.
Teammates load with ``load_splits`` or ``load_credit_card_splits``.

Usage:
    from pathlib import Path
    from src.data_pipeline import load_credit_card_splits

    root = Path(__file__).resolve().parent.parent
    X_train, X_val, X_test, y_train, y_val, y_test = load_credit_card_splits(root)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DATA_PATH = "../data/creditcard.csv"


def _split_from_dataframe(df: pd.DataFrame):
    """Shared train/val/test split + scaling (same logic as Member A notebook)."""
    df = df.drop_duplicates()

    normal = df[df["Class"] == 0].copy()
    fraud = df[df["Class"] == 1].copy()

    X_norm = normal.drop(columns=["Class"])
    y_norm = normal["Class"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_norm, y_norm, test_size=0.40, random_state=RANDOM_STATE
    )

    X_val_norm, X_test_norm, y_val_norm, y_test_norm = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    X_fraud = fraud.drop(columns=["Class"])
    y_fraud = fraud["Class"]

    X_val_fraud, X_test_fraud, y_val_fraud, y_test_fraud = train_test_split(
        X_fraud, y_fraud, test_size=0.50, random_state=RANDOM_STATE
    )

    X_val = pd.concat([X_val_norm, X_val_fraud])
    y_val = pd.concat([y_val_norm, y_val_fraud])
    X_test = pd.concat([X_test_norm, X_test_fraud])
    y_test = pd.concat([y_test_norm, y_test_fraud])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train.values,
        y_val.values,
        y_test.values,
        scaler,
    )


def resolve_creditcard_csv(project_root: Union[str, Path]) -> Path:
    """
    Locate creditcard.csv. Search order:

    1. Environment variable CREDITCARD_CSV
    2. <project_root>/data/creditcard.csv
    3. <project_root>/creditcard.csv
    4. kagglehub.dataset_download(\"mlg-ulb/creditcardfraud\") and return path to creditcard.csv

    Raises FileNotFoundError if nothing works.
    """
    root = Path(project_root).resolve()

    env = os.environ.get("CREDITCARD_CSV", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p

    for rel in ("data/creditcard.csv", "creditcard.csv"):
        p = (root / rel).resolve()
        if p.is_file():
            return p

    try:
        import kagglehub

        hub_dir = Path(kagglehub.dataset_download("mlg-ulb/creditcardfraud"))
        for f in hub_dir.rglob("creditcard.csv"):
            if f.is_file():
                return f.resolve()
    except Exception:
        pass

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    raise FileNotFoundError(
        "Could not find creditcard.csv.\n\n"
        "  • Place the Kaggle file at: "
        f"{data_dir / 'creditcard.csv'}\n"
        "  • Or set CREDITCARD_CSV to its full path\n"
        "  • Or install kagglehub + Kaggle API credentials for auto-download\n"
    )


def load_and_split(data_path=DATA_PATH):
    """
    Load CSV (same as Member A: ``pd.read_csv``), split, scale.
    Scaler is fit on training normal-only rows only in the returned arrays
    (train is normal-only after split construction in Member A pipeline).

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    df = pd.read_csv(data_path)
    return _split_from_dataframe(df)


def load_splits(artifact_path: Union[str, Path] = "../artifacts/") -> Tuple[np.ndarray, ...]:
    """
    Load pre-saved numpy splits from Member A (``np.save`` under project ``artifacts/``).
    """
    base = Path(artifact_path)
    if not base.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {base}")
    X_train = np.load(base / "X_train.npy")
    X_val = np.load(base / "X_val.npy")
    X_test = np.load(base / "X_test.npy")
    y_train = np.load(base / "y_train.npy")
    y_val = np.load(base / "y_val.npy")
    y_test = np.load(base / "y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_credit_card_splits(
    project_root: Union[str, Path],
    *,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preferred team entry point — matches Member A workflow:

    1. If ``<project_root>/artifacts/X_train.npy`` exists, load with :func:`load_splits`
       (arrays produced by Member A notebook).
    2. Else if ``<project_root>/data/creditcard.csv`` exists, run :func:`load_and_split`
       (same as reading ``DATA_PATH + 'creditcard.csv'`` in Member A).
    3. Else try :func:`resolve_creditcard_csv` then :func:`load_and_split`.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    root = Path(project_root).resolve()
    art = root / "artifacts"
    if (art / "X_train.npy").is_file():
        if verbose:
            print(f"Loaded splits from {art} (Member A .npy).")
        return load_splits(art)

    csv_local = root / "data" / "creditcard.csv"
    if csv_local.is_file():
        if verbose:
            print(f"Loaded from {csv_local} via load_and_split (shared pipeline).")
        X_train, X_val, X_test, y_train, y_val, y_test, _ = load_and_split(str(csv_local))
        return X_train, X_val, X_test, y_train, y_val, y_test

    csv_path = resolve_creditcard_csv(root)
    if verbose:
        print(f"Using creditcard.csv: {csv_path}")
    X_train, X_val, X_test, y_train, y_val, y_test, _ = load_and_split(str(csv_path))
    return X_train, X_val, X_test, y_train, y_val, y_test
