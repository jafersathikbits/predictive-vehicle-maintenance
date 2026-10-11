# src/data_loader.py
import pandas as pd
from src.config import RAW_DATA_PATH

def load_raw_data():
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded raw data: {df.shape}")
    return df