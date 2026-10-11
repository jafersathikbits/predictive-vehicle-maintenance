# src/preprocess.py
import pandas as pd
from src.config import (
    CATEGORIES, CONDITION_MAP, HISTORY_MAP, REFERENCE_DATE, TARGET,
)

DROP_COLS = [
    'Maintenance_History', 'Tire_Condition', 'Brake_Condition', 'Battery_Status',
    'Last_Service_Date', 'Warranty_Expiry_Date',
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw vehicle records into model features.

    Shared by training and the app so both always encode inputs the same way.
    Rows may carry either the raw date columns or precomputed
    ``days_since_last_service`` / ``days_to_warranty_end``.
    Does not modify ``df``.
    """
    df = df.copy()

    # Time features
    ref_date = pd.Timestamp(REFERENCE_DATE)
    if 'days_since_last_service' not in df:
        df['days_since_last_service'] = (ref_date - pd.to_datetime(df['Last_Service_Date'])).dt.days
    if 'days_to_warranty_end' not in df:
        df['days_to_warranty_end'] = (pd.to_datetime(df['Warranty_Expiry_Date']) - ref_date).dt.days

    # Ordinal encoding
    df['Tire_num'] = df['Tire_Condition'].map(CONDITION_MAP)
    df['Brake_num'] = df['Brake_Condition'].map(CONDITION_MAP)
    df['Battery_num'] = df['Battery_Status'].map(CONDITION_MAP)
    df['History_num'] = df['Maintenance_History'].map(HISTORY_MAP)
    df['condition_score'] = (df['Tire_num'] + df['Brake_num'] + df['Battery_num']) / 3

    # One-hot encoding with fixed levels, so a single row or a partial batch
    # still produces every dummy column the model was trained on
    for col, levels in CATEGORIES.items():
        df[col] = pd.Categorical(df[col], categories=levels)
    df = pd.get_dummies(df, columns=list(CATEGORIES), drop_first=True, dtype=int)

    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])


def validate_raw(df: pd.DataFrame) -> list[str]:
    """Return human-readable problems with raw input rows (empty list = OK)."""
    problems = []
    checks = {
        'Tire_Condition': CONDITION_MAP, 'Brake_Condition': CONDITION_MAP,
        'Battery_Status': CONDITION_MAP, 'Maintenance_History': HISTORY_MAP,
        **CATEGORIES,
    }
    for col, allowed in checks.items():
        if col not in df:
            problems.append(f"Missing column: {col}")
            continue
        bad = sorted(set(df[col].dropna().astype(str)) - set(allowed))
        if bad:
            problems.append(f"{col} has unknown values: {', '.join(bad[:5])}")
    return problems


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Starting preprocessing...")
    df = engineer_features(df)

    # Keep the target as the last column
    df = df[[c for c in df.columns if c != TARGET] + [TARGET]]

    print("Preprocessing complete. Final shape:", df.shape)
    return df
