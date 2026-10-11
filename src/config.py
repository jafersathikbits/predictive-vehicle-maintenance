# src/config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths (all absolute, so scripts and the app work from any working directory)
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "vehicle_maintenance_data.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "vehicle_maintenance_processed.csv")
FEATURE_NAMES_PATH = os.path.join(DATA_DIR, "feature_names.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
INPUT_DEFAULTS_PATH = os.path.join(MODEL_DIR, "input_defaults.json")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Need_Maintenance"

# Date the time-based features are measured from. The training data's service
# and warranty dates are fixed, so this must stay the same at train and predict
# time, otherwise "days since last service" drifts out of the training range.
REFERENCE_DATE = "2025-07-01"

# Categorical levels, in the order used for one-hot encoding (first = dropped baseline)
CATEGORIES = {
    "Vehicle_Model": ["Bus", "Car", "Motorcycle", "SUV", "Truck", "Van"],
    "Fuel_Type": ["Diesel", "Electric", "Petrol"],
    "Transmission_Type": ["Automatic", "Manual"],
    "Owner_Type": ["First", "Second", "Third"],
}
CONDITION_MAP = {"New": 2, "Good": 1, "Worn Out": 0, "Weak": 0}
HISTORY_MAP = {"Good": 2, "Average": 1, "Poor": 0}

NUMERIC_FEATURES = [
    'Mileage', 'Reported_Issues', 'Vehicle_Age', 'Engine_Size',
    'Odometer_Reading', 'Insurance_Premium', 'Service_History',
    'Accident_History', 'Fuel_Efficiency', 'days_since_last_service',
    'days_to_warranty_end', 'condition_score'
]
