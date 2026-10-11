# app.py
import json
import os

import altair as alt
import joblib
import pandas as pd
import shap
import streamlit as st

from src.config import (
    CATEGORIES, FEATURE_NAMES_PATH, HISTORY_MAP, INPUT_DEFAULTS_PATH,
    METRICS_PATH, MODEL_PATH, NUMERIC_FEATURES, RAW_DATA_PATH, REPORTS_DIR, SCALER_PATH,
    TARGET,
)
from src.explain import label, recommendations, shap_values_for, top_factors
from src.preprocess import engineer_features, validate_raw

# Must be the first Streamlit call
st.set_page_config(page_title="Vehicle Maintenance Predictor", page_icon="🚗", layout="wide")

st.markdown("""
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    div[data-testid="stMetricValue"] {font-size: 2rem;}
    .risk-badge {display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px;
                 font-weight: 600; color: white;}
    </style>
""", unsafe_allow_html=True)

RISK_LEVELS = [  # (lower bound, name, colour)
    (0.70, "High", "#dc2626"),
    (0.50, "Elevated", "#ea580c"),
    (0.30, "Moderate", "#ca8a04"),
    (0.00, "Low", "#16a34a"),
]
NUMERIC_RAW = [
    'Mileage', 'Reported_Issues', 'Vehicle_Age', 'Engine_Size', 'Odometer_Reading',
    'Insurance_Premium', 'Service_History', 'Accident_History', 'Fuel_Efficiency',
]
# Fallback form ranges if models/input_defaults.json is missing (training data ranges)
FALLBACK_DEFAULTS = {
    'Mileage': (30000, 80000, 55000), 'Reported_Issues': (0, 5, 2), 'Vehicle_Age': (1, 10, 5),
    'Engine_Size': (800, 2500, 1500), 'Odometer_Reading': (1000, 149999, 75000),
    'Insurance_Premium': (5000, 30000, 17500), 'Service_History': (1, 10, 6),
    'Accident_History': (0, 3, 1), 'Fuel_Efficiency': (10.0, 20.0, 15.0),
    'days_since_last_service': (489, 824, 656), 'days_to_warranty_end': (-430, 271, -80),
}


# ─── Cached resources ───
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = pd.read_csv(FEATURE_NAMES_PATH, header=None)[0].tolist()
    explainer = shap.TreeExplainer(model)
    return model, scaler, feature_names, explainer


@st.cache_data
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def csv_template() -> bytes:
    sample = pd.read_csv(RAW_DATA_PATH, nrows=5).drop(columns=[TARGET])
    return sample.to_csv(index=False).encode()


def risk_level(p: float):
    return next((name, colour) for bound, name, colour in RISK_LEVELS if p >= bound)


def predict(raw: pd.DataFrame):
    """Return (probabilities, scaled feature frame) for raw vehicle rows."""
    model, scaler, feature_names, _ = load_artifacts()
    X = engineer_features(raw).reindex(columns=feature_names, fill_value=0)
    X = X.astype(float)
    X[NUMERIC_FEATURES] = scaler.transform(X[NUMERIC_FEATURES])
    return model.predict_proba(X)[:, 1], X


try:
    model, scaler, feature_names, explainer = load_artifacts()
except FileNotFoundError as e:
    st.error(f"Model files not found ({e.filename}). Run `python run_preprocess.py` "
             "and `python run_training.py` first.")
    st.stop()

metrics = load_json(METRICS_PATH)
defaults = load_json(INPUT_DEFAULTS_PATH) or {}


def rng(col, integer=True):
    """(min, max, default) for a form input, from the training data."""
    if col in defaults:
        d = defaults[col]
        lo, hi, mid = d['min'], d['max'], d['median']
    else:
        lo, hi, mid = FALLBACK_DEFAULTS[col]
    cast = int if integer else float
    return cast(lo), cast(hi), cast(round(mid) if integer else mid)


# ─── Sidebar ───
with st.sidebar:
    st.header("About")
    st.write("Predicts whether a vehicle needs maintenance and explains which "
             "factors drive the prediction.")
    if metrics:
        best = metrics['results'][metrics['best_model']]
        st.subheader("Model")
        st.write(f"**{metrics['best_model']}** on {metrics['test_rows']:,} held-out vehicles")
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC", f"{best['AUC']:.3f}")
        c2.metric("F1", f"{best['F1_1']:.3f}")
        c1.metric("Recall", f"{best['Recall_1']:.3f}")
        c2.metric("Precision", f"{best['Precision_1']:.3f}")
    st.caption("Demo trained on synthetic data – for educational purposes only.")

st.title("🚗 Vehicle Maintenance Predictor")
st.caption("Enter vehicle details to get a maintenance prediction, risk level and explanation.")

tab_single, tab_batch, tab_insights = st.tabs(["Single vehicle", "Batch prediction", "Model insights"])

# ─── Single vehicle ───
with tab_single:
    with st.form("vehicle_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Usage**")
            lo, hi, mid = rng('Mileage')
            mileage = st.number_input("Mileage (km)", lo, hi, mid, step=1000)
            lo, hi, mid = rng('Vehicle_Age')
            vehicle_age = st.slider("Vehicle age (years)", lo, hi, mid)
            lo, hi, mid = rng('Reported_Issues')
            reported_issues = st.slider("Reported issues", lo, hi, mid)
            lo, hi, mid = rng('Fuel_Efficiency', integer=False)
            fuel_eff = st.slider("Fuel efficiency (km/l)", lo, hi, round(mid, 1), step=0.1)

        with col2:
            st.markdown("**Vehicle**")
            vehicle_model = st.selectbox("Vehicle model", CATEGORIES['Vehicle_Model'], index=1)
            fuel_type = st.selectbox("Fuel type", CATEGORIES['Fuel_Type'], index=2)
            transmission = st.selectbox("Transmission", CATEGORIES['Transmission_Type'])
            owner_type = st.selectbox("Owner type", CATEGORIES['Owner_Type'])

        with col3:
            st.markdown("**Condition**")
            tire_cond = st.selectbox("Tire condition", ["New", "Good", "Worn Out"], index=1)
            brake_cond = st.selectbox("Brake condition", ["New", "Good", "Worn Out"], index=1)
            battery_stat = st.selectbox("Battery status", ["New", "Good", "Weak"], index=1)
            maint_history = st.selectbox("Maintenance history", list(HISTORY_MAP), index=1)

        with st.expander("More details (pre-filled with typical values)"):
            a1, a2, a3 = st.columns(3)
            with a1:
                lo, hi, mid = rng('Service_History')
                service_history = st.slider("Number of past services", lo, hi, mid)
                lo, hi, mid = rng('Accident_History')
                accident_history = st.slider("Accidents", lo, hi, mid)
                lo, hi, mid = rng('Engine_Size')
                engine_size = st.number_input("Engine size (cc)", lo, hi, mid, step=100)
            with a2:
                lo, hi, mid = rng('Odometer_Reading')
                odometer = st.number_input("Odometer reading (km)", lo, hi, mid, step=1000)
                lo, hi, mid = rng('Insurance_Premium')
                insurance = st.number_input("Insurance premium", lo, hi, mid, step=500)
            with a3:
                lo, hi, mid = rng('days_since_last_service')
                days_since_service = st.slider("Days since last service", lo, hi, mid,
                                               help="Limited to the range seen in the training data.")
                lo, hi, mid = rng('days_to_warranty_end')
                days_to_warranty = st.slider("Days to warranty end", lo, hi, mid,
                                             help="Negative = warranty already expired.")

        submit = st.form_submit_button("Get prediction", type="primary", width="stretch")

    if submit:
        raw = pd.DataFrame([{
            'Vehicle_Model': vehicle_model, 'Mileage': mileage,
            'Maintenance_History': maint_history, 'Reported_Issues': reported_issues,
            'Vehicle_Age': vehicle_age, 'Fuel_Type': fuel_type,
            'Transmission_Type': transmission, 'Engine_Size': engine_size,
            'Odometer_Reading': odometer, 'Owner_Type': owner_type,
            'Insurance_Premium': insurance, 'Service_History': service_history,
            'Accident_History': accident_history, 'Fuel_Efficiency': fuel_eff,
            'Tire_Condition': tire_cond, 'Brake_Condition': brake_cond,
            'Battery_Status': battery_stat,
            'days_since_last_service': days_since_service,
            'days_to_warranty_end': days_to_warranty,
        }])

        with st.spinner("Analyzing your vehicle..."):
            proba, X = predict(raw)
            proba = float(proba[0])
            factors = top_factors(shap_values_for(explainer, X)[0], feature_names)

        level, colour = risk_level(proba)
        needs = proba >= 0.5

        st.divider()
        r1, r2, r3 = st.columns([1.2, 1, 1])
        with r1:
            st.markdown("### Prediction")
            if needs:
                st.error("**Maintenance needed**", icon="🔧")
            else:
                st.success("**No immediate maintenance needed**", icon="✅")
        r2.metric("Maintenance probability", f"{proba:.1%}")
        with r3:
            st.markdown("Risk level")
            st.markdown(f'<span class="risk-badge" style="background:{colour}">{level}</span>',
                        unsafe_allow_html=True)
        st.progress(proba)

        st.markdown("### Why this prediction")
        left, right = st.columns([1.4, 1])
        with left:
            chart_df = factors.assign(
                effect=factors['shap'].map(lambda v: "Increases risk" if v > 0 else "Decreases risk"))
            chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X('shap:Q', title="Impact on prediction (SHAP value)"),
                y=alt.Y('label:N', sort=None, title=None, axis=alt.Axis(labelLimit=220)),
                color=alt.Color('effect:N', title=None, legend=alt.Legend(orient='bottom'),
                                scale=alt.Scale(domain=["Increases risk", "Decreases risk"],
                                                range=["#ef4444", "#10b981"])),
                tooltip=[alt.Tooltip('label:N', title="Factor"),
                         alt.Tooltip('shap:Q', title="Impact", format=".3f")],
            ).properties(height=320)
            st.altair_chart(chart, width="stretch")
        with right:
            st.markdown("**Top factors**")
            for _, row in factors.head(5).iterrows():
                arrow = "🔺 raises" if row['shap'] > 0 else "🔻 lowers"
                st.markdown(f"- **{row['label']}** {arrow} risk ({row['shap']:+.2f})")
            tips = recommendations(factors)
            if needs and tips:
                st.markdown("**Recommended actions**")
                for tip in tips:
                    st.markdown(f"- {tip}")
            elif not needs:
                st.info("Vehicle looks in good shape. Keep to the regular service schedule.")

# ─── Batch prediction ───
with tab_batch:
    st.markdown("Upload a CSV with one vehicle per row, in the same format as the raw dataset "
                "(the `Need_Maintenance` column is not required).")
    st.download_button("Download CSV template", csv_template(), "vehicle_template.csv", "text/csv")
    upload = st.file_uploader("Vehicle CSV", type="csv")
    if upload is not None:
        try:
            batch = pd.read_csv(upload)
            problems = validate_raw(batch)
        except Exception as e:
            batch, problems = pd.DataFrame(), [f"Could not read the CSV: {e}"]

        has_dates = {'Last_Service_Date', 'Warranty_Expiry_Date'} <= set(batch.columns)
        has_days = {'days_since_last_service', 'days_to_warranty_end'} <= set(batch.columns)
        if not batch.empty and not (has_dates or has_days):
            problems.append("Need Last_Service_Date and Warranty_Expiry_Date "
                            "(or days_since_last_service and days_to_warranty_end).")
        problems += [f"Missing column: {c}" for c in NUMERIC_RAW if not batch.empty and c not in batch]
        if batch.empty and not problems:
            problems.append("The file has no rows.")

        if problems:
            st.error("The file has problems:\n\n" + "\n".join(f"- {p}" for p in problems))
        else:
            with st.spinner(f"Scoring {len(batch):,} vehicles..."):
                probs, _ = predict(batch)
            out = batch.drop(columns=[TARGET], errors='ignore').copy()
            out.insert(0, 'Maintenance_Probability', probs.round(4))
            out.insert(1, 'Risk_Level', [risk_level(p)[0] for p in probs])
            out.insert(2, 'Prediction', ["Maintenance needed" if p >= 0.5 else "OK" for p in probs])

            m1, m2, m3 = st.columns(3)
            m1.metric("Vehicles", f"{len(out):,}")
            m2.metric("Need maintenance", f"{(probs >= 0.5).sum():,}")
            m3.metric("Share flagged", f"{(probs >= 0.5).mean():.1%}")
            st.dataframe(
                out.sort_values('Maintenance_Probability', ascending=False),
                width="stretch", hide_index=True,
                column_config={"Maintenance_Probability": st.column_config.ProgressColumn(
                    "Probability", min_value=0.0, max_value=1.0, format="%.2f")},
            )
            st.download_button("Download predictions", out.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv", type="primary")

# ─── Model insights ───
with tab_insights:
    if metrics:
        st.markdown("#### Model comparison (held-out test set)")
        st.dataframe(pd.DataFrame(metrics['results']).T.round(4), width="stretch")

    if hasattr(model, 'feature_importances_'):
        st.markdown("#### Global feature importance")
        imp = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
        imp['label'] = imp['feature'].map(label)
        imp = imp.sort_values('importance', ascending=False).head(12)
        st.altair_chart(alt.Chart(imp).mark_bar(color="#3b82f6").encode(
            x=alt.X('importance:Q', title="Importance"),
            y=alt.Y('label:N', sort=None, title=None, axis=alt.Axis(labelLimit=220)),
        ).properties(height=340), width="stretch")

    images = sorted(f for f in os.listdir(REPORTS_DIR) if f.endswith('.png')) \
        if os.path.isdir(REPORTS_DIR) else []
    if images:
        st.markdown("#### Exploratory data analysis")
        cols = st.columns(2)
        for i, img in enumerate(images):
            cols[i % 2].image(os.path.join(REPORTS_DIR, img),
                              caption=img.removeprefix('eda_').removesuffix('.png').replace('_', ' '))
    else:
        st.caption("Run `python run_eda.py` to generate EDA charts here.")
