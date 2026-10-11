# run_eda.py
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import PROCESSED_DATA_PATH, REPORTS_DIR, TARGET

# Better looking plots
plt.style.use('seaborn-v0_8-whitegrid')

def run_eda():
    print("=== Phase 4: Exploratory Data Analysis ===")
    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    print("Shape:", df.shape, "\n")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    target = TARGET

    # ──────────────── 1. Target distribution ────────────────
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target, data=df, hue=target, palette='viridis', legend=False)
    plt.title("Target Distribution\n(Need_Maintenance: 1 = Yes, 0 = No)")
    plt.xlabel("Need Maintenance")
    plt.ylabel("Count")
    plt.savefig(os.path.join(REPORTS_DIR, "eda_target_distribution.png"), dpi=120, bbox_inches='tight')
    plt.close()

    print("Target distribution (%):")
    print(df[target].value_counts(normalize=True).mul(100).round(2))
    print()

    # ──────────────── 2. Top correlations with target ────────────────
    corr_with_target = df.corr(numeric_only=True)[target].drop(target).abs().sort_values(ascending=False)
    top_corr = corr_with_target.head(15)

    print("Top 15 features correlated with Need_Maintenance (absolute value):")
    print(top_corr.round(3))
    print()

    # Bar plot of top correlations
    plt.figure(figsize=(10, 6))
    top_corr.plot(kind='bar', color='teal')
    plt.title("Top Features by Correlation with Need_Maintenance")
    plt.ylabel("Absolute Correlation")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "eda_top_correlations.png"), dpi=120, bbox_inches='tight')
    plt.close()

    # ──────────────── 3. Boxplots of most promising numerical features ────────────────
    key_features = [
        'days_since_last_service',
        'condition_score',
        'Mileage',
        'Reported_Issues',
        'Vehicle_Age',
        'days_to_warranty_end'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    for i, col in enumerate(key_features):
        if col in df.columns:
            sns.boxplot(x=target, y=col, data=df, hue=target, ax=axes[i], palette='viridis', legend=False)
            axes[i].set_title(f"{col} by Need_Maintenance")
        else:
            axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "eda_boxplots_key_features.png"), dpi=120, bbox_inches='tight')
    plt.close()

    # ──────────────── 4. Condition score histogram by target ────────────────
    plt.figure(figsize=(9, 5))
    sns.histplot(data=df, x='condition_score', hue=target, multiple='stack', bins=25, palette='viridis')
    plt.title("Condition Score Distribution by Need_Maintenance")
    plt.xlabel("Condition Score (higher = better)")
    plt.ylabel("Count")
    plt.savefig(os.path.join(REPORTS_DIR, "eda_condition_score_by_target.png"), dpi=120, bbox_inches='tight')
    plt.close()

    print(f"\nAll EDA plots saved in {REPORTS_DIR}:")
    print("• eda_target_distribution.png")
    print("• eda_top_correlations.png")
    print("• eda_boxplots_key_features.png")
    print("• eda_condition_score_by_target.png")
    print("\nPhase 4 (EDA) completed. Check the images!")

if __name__ == "__main__":
    run_eda()