# run_preprocess.py
import pandas as pd
from src.config import PROCESSED_DATA_PATH, FEATURE_NAMES_PATH, TARGET
from src.data_loader import load_raw_data
from src.preprocess import preprocess_data

if __name__ == "__main__":
    print("=== Starting Data Preprocessing Phase ===")
    df_processed = preprocess_data(load_raw_data())

    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed file saved → {PROCESSED_DATA_PATH}")

    # Save exact feature names the model is trained on
    feature_names = df_processed.drop(columns=[TARGET]).columns.tolist()
    pd.Series(feature_names).to_csv(FEATURE_NAMES_PATH, index=False, header=False)
    print(f"Feature names saved → {FEATURE_NAMES_PATH}")

    print("=== Preprocessing Finished Successfully ===")
