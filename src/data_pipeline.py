"""
data_pipeline.py
Shared preprocessing pipeline for ECE447 Anomaly Detection Project.
All members import from this file to ensure consistent data splits.

Usage:
    from src.data_pipeline import load_splits

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATA_PATH = '../data/creditcard.csv'


def load_and_split(data_path=DATA_PATH):
    """
    Loads the dataset, splits into train/val/test, scales features.
    Scaler is fit on training data only.

    Returns:
        X_train, X_val, X_test: scaled numpy arrays
        y_train, y_val, y_test: label arrays
        scaler: fitted StandardScaler instance
    """
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()

    normal = df[df['Class'] == 0].copy()
    fraud  = df[df['Class'] == 1].copy()

    X_norm = normal.drop(columns=['Class'])
    y_norm = normal['Class']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_norm, y_norm, test_size=0.40, random_state=RANDOM_STATE)

    X_val_norm, X_test_norm, y_val_norm, y_test_norm = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE)

    X_fraud = fraud.drop(columns=['Class'])
    y_fraud = fraud['Class']

    X_val_fraud, X_test_fraud, y_val_fraud, y_test_fraud = train_test_split(
        X_fraud, y_fraud, test_size=0.50, random_state=RANDOM_STATE)

    X_val  = pd.concat([X_val_norm,  X_val_fraud])
    y_val  = pd.concat([y_val_norm,  y_val_fraud])
    X_test = pd.concat([X_test_norm, X_test_fraud])
    y_test = pd.concat([y_test_norm, y_test_fraud])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train.values, y_val.values, y_test.values,
            scaler)


def load_splits(artifact_path='../artifacts/'):
    """
    Loads pre-saved numpy split files produced by Member A.
    Use this if you do not have the raw CSV.
    """
    X_train = np.load(artifact_path + 'X_train.npy')
    X_val   = np.load(artifact_path + 'X_val.npy')
    X_test  = np.load(artifact_path + 'X_test.npy')
    y_train = np.load(artifact_path + 'y_train.npy')
    y_val   = np.load(artifact_path + 'y_val.npy')
    y_test  = np.load(artifact_path + 'y_test.npy')

    return X_train, X_val, X_test, y_train, y_val, y_test