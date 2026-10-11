# src/explain.py
import numpy as np
import pandas as pd

# Plain-language names for model features
FEATURE_LABELS = {
    'Mileage': 'Mileage',
    'Reported_Issues': 'Reported issues',
    'Vehicle_Age': 'Vehicle age',
    'Engine_Size': 'Engine size',
    'Odometer_Reading': 'Odometer reading',
    'Insurance_Premium': 'Insurance premium',
    'Service_History': 'Number of past services',
    'Accident_History': 'Accident history',
    'Fuel_Efficiency': 'Fuel efficiency',
    'days_since_last_service': 'Days since last service',
    'days_to_warranty_end': 'Days to warranty end',
    'Tire_num': 'Tire condition',
    'Brake_num': 'Brake condition',
    'Battery_num': 'Battery status',
    'History_num': 'Maintenance history',
    'condition_score': 'Overall condition score',
}

# Advice shown when a feature pushes risk up
RECOMMENDATIONS = {
    'Brake_num': "Inspect brake pads and discs; replace worn components.",
    'Battery_num': "Test the battery and charging system; replace if weak.",
    'Tire_num': "Check tread depth and pressure; replace worn tires.",
    'Reported_Issues': "Diagnose and resolve the open reported issues.",
    'History_num': "Bring the maintenance schedule up to date.",
    'Accident_History': "Have accident-related repairs and alignment checked.",
    'days_since_last_service': "Book a routine service.",
    'Service_History': "Review past service records for recurring faults.",
    'Mileage': "Follow the high-mileage service checklist.",
    'Vehicle_Age': "Schedule an age-related inspection (hoses, belts, seals).",
}


def label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    # One-hot columns, e.g. Vehicle_Model_Truck -> "Vehicle model: Truck"
    for prefix in ('Vehicle_Model', 'Fuel_Type', 'Transmission_Type', 'Owner_Type'):
        if feature.startswith(prefix + '_'):
            return f"{prefix.replace('_', ' ').capitalize()}: {feature[len(prefix) + 1:]}"
    return feature


def shap_values_for(explainer, X: pd.DataFrame) -> np.ndarray:
    """SHAP values for the positive class as a (rows, features) array, for any tree model."""
    values = explainer.shap_values(X)
    if isinstance(values, list):  # older SHAP: one array per class
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:  # (rows, features, classes)
        values = values[:, :, -1]
    return values


def top_factors(shap_row: np.ndarray, feature_names: list[str], n: int = 8) -> pd.DataFrame:
    """The n features with the largest absolute impact, strongest first."""
    df = pd.DataFrame({'feature': feature_names, 'shap': shap_row})
    df = df.loc[df['shap'].abs().sort_values(ascending=False).index].head(n)
    df['label'] = df['feature'].map(label)
    return df.reset_index(drop=True)


def recommendations(factors: pd.DataFrame) -> list[str]:
    tips = []
    for feat in factors.loc[factors['shap'] > 0, 'feature']:
        tip = RECOMMENDATIONS.get(feat)
        if tip and tip not in tips:
            tips.append(tip)
    return tips
