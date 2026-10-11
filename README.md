# Predictive Vehicle Maintenance

A Streamlit machine-learning application that estimates whether a vehicle may need maintenance based on its usage, condition, and service-history features. It supports single-vehicle predictions, batch CSV analysis, and SHAP-based model explanations.

> **Important:** This project uses synthetic data and is intended for educational demonstration only. Predictions are not a substitute for inspection by a qualified mechanic. All compared models currently show perfect saved test metrics, which should be investigated for target leakage or an overly deterministic synthetic target before drawing conclusions about real-world performance.

## Overview

This project provides an end-to-end machine-learning workflow for vehicle maintenance prediction. It includes data preprocessing, feature engineering, exploratory data analysis, model training and comparison, saved model artifacts, and an interactive Streamlit interface.

The application offers three main tabs: **Single vehicle**, **Batch prediction**, and **Model insights**. Users can explore maintenance predictions, review influential features, and process multiple vehicles from a CSV file.

## Key Features

- Predict maintenance needs for an individual vehicle.
- Display a maintenance risk level and suggested actions.
- Explain predictions using SHAP feature contributions.
- Upload a CSV for batch predictions and download the results.
- Download a sample CSV template for batch inputs.
- View saved evaluation metrics and exploratory data analysis charts.
- Rebuild processed data, charts, and trained model artifacts using the included scripts.

## Tech Stack

- **Language:** Python 3
- **Application:** Streamlit
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn, XGBoost, LightGBM
- **Explainability:** SHAP
- **Visualization:** Altair, Matplotlib, Seaborn
- **Model persistence:** joblib

## System Architecture

1. **Data ingestion:** Load the raw vehicle maintenance dataset.
2. **Preprocessing:** Validate input data and engineer model features.
3. **Exploratory analysis:** Generate charts to examine feature distributions, correlations, and the target variable.
4. **Model training:** Train and compare tree-based machine-learning models.
5. **Model persistence:** Save the selected model, scaler, feature names, input defaults, and evaluation metrics.
6. **Prediction:** Process user-entered vehicle details or uploaded CSV rows.
7. **Explanation:** Use SHAP values to highlight features that influence each prediction.
8. **Presentation:** Display the predicted maintenance probability, risk level, recommendations, and model insights in Streamlit.

## Project Structure

```text
predictive-vehicle-maintenance/
├── app.py                          # Streamlit application
├── run_preprocess.py               # Preprocessing and feature engineering
├── run_eda.py                      # Exploratory data analysis
├── run_training.py                 # Model training and comparison
├── requirements.txt                # Python dependencies
├── src/
│   ├── __init__.py
│   ├── config.py                   # Paths and shared constants
│   ├── data_loader.py              # Raw data loading
│   ├── preprocess.py               # Validation and feature engineering
│   └── explain.py                  # SHAP explanations and recommendations
├── data/
│   ├── vehicle_maintenance_data.csv
│   ├── vehicle_maintenance_processed.csv
│   └── feature_names.csv
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── metrics.json
│   └── input_defaults.json
└── reports/
    └── eda_*.png                   # Exploratory analysis charts
```

The raw and processed datasets are included in the repository. The model artifacts are also included so that the application can normally start without retraining.

## Prerequisites

- Python 3.10 or newer.
- pip.
- Sufficient memory and disk space for the dataset, model artifacts, and dependencies.
- A compatible Python environment for the versions of scikit-learn and other libraries used to create the saved model artifacts.

## Installation and Setup

Clone the repository and enter the project directory:

```bash
git clone <your-repository-url>
cd predictive-vehicle-maintenance
```

Create and activate a virtual environment.

**macOS/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The included scaler was saved with scikit-learn 1.8.0. If loading the saved artifacts fails or produces compatibility warnings, use a compatible scikit-learn version or regenerate the artifacts using the training pipeline.

## Environment Variables

No environment variables are required for the default local workflow. The application does not currently read environment variables.

If deployment-specific secrets are introduced later, keep them outside source control and use the hosting platform's secrets-management facility.

## Usage

### Launch the Application

The trained model artifacts are included, so you can normally start the application without retraining.

```bash
streamlit run app.py
```

Streamlit prints a local URL, typically [http://localhost:8501](http://localhost:8501).

### Single-Vehicle Prediction

Open the **Single vehicle** tab, enter the requested vehicle details, and submit the form. The application displays the prediction, risk level, influential features, and suggested actions.

### Batch Prediction

Open the **Batch prediction** tab and download the CSV template. Upload a CSV containing the expected input columns.

The `Need_Maintenance` target column is not required for prediction. Use the template to check the expected column names and input format.

### Model Insights

Use the **Model insights** tab to explore saved evaluation metrics and exploratory analysis charts.

### Rebuild the Pipeline

Run the following commands from the project root, in order:

```bash
python run_preprocess.py
python run_eda.py
python run_training.py
```

These scripts regenerate processed data, exploratory charts, and saved model artifacts. Existing outputs may be overwritten.

The `REFERENCE_DATE` setting in `src/config.py` is used for date-derived features. Keep the training and prediction feature logic consistent to avoid unintended feature drift.

### Testing

No automated test suite is included in the current project. Before making changes, verify that the application starts, single predictions work, batch uploads are validated, and SHAP explanations render correctly.

## Deployment

This is a Python and Streamlit application, so it cannot run as a static website on GitHub Pages.

A suitable hosting option is [Streamlit Community Cloud](https://streamlit.io/cloud) or another service that supports Python applications.

Typical deployment steps:

1. Push the repository to GitHub.
2. Configure the hosting service to use the repository, branch, and `app.py` entry point.
3. Confirm that the Python and dependency versions are compatible with the saved model artifacts.
4. Ensure the required CSV files, model artifacts, and reports are available at runtime.
5. Test single predictions, batch uploads, and model explanations after deployment.

Review repository and hosting file-size limits before committing datasets and model artifacts. Do not include confidential vehicle records or personal information in a public repository. Only load trusted pickle-based model artifacts because loading untrusted pickle files can execute arbitrary code.

## Future Improvements

- Evaluate models on realistic, independently sourced vehicle maintenance data.
- Investigate target leakage and the perfect saved evaluation metrics.
- Add automated tests for preprocessing, validation, predictions, and explanations.
- Introduce reproducible experiments and documented model evaluation.
- Improve validation and error handling for uploaded CSV files.
- Add model versioning and monitoring for prediction drift.
- Document limitations and the intended use of maintenance recommendations.

## Contributing

1. Create a feature branch for your changes.
2. Keep changes focused and follow the existing project structure.
3. Test the affected workflows before opening a pull request.
4. Document changes to data processing, feature engineering, and model training.
5. Never commit private datasets, personal information, credentials, or untrusted model files.

## License

A standalone `LICENSE` file has not yet been added. Licensing will be finalized across the project repositories together. Until a license is added, do not assume that the project is available for unrestricted reuse.
