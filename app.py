import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------------------------------
# MUST be the first Streamlit command
# --------------------------------------------------
st.set_page_config(page_title="Crop Yield DSS", layout="wide")

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
MODEL_PATH = "models/best_model.joblib"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Train and save best_model.joblib first.")
        return None
    return joblib.load(MODEL_PATH)


model = load_model()

# --------------------------------------------------
# Page Title + Description
# --------------------------------------------------
st.title("🌾 Crop Yield Prediction – Decision Support System")

st.markdown(
    """
This Decision Support System predicts **crop yield (hg/ha)** based on:
- Region (Country)
- Crop Type
- Year
- Rainfall
- Temperature
- Pesticide Usage

It supports:
✔ Single prediction  
✔ What-if analysis  
✔ Model KPI monitoring  

To keep predictions realistic, numeric inputs are constrained to the
**5th–95th percentile** of historical data. Values near the extremes
may be less reliable.
"""
)

# --------------------------------------------------
# Load dataset for dropdown values
# --------------------------------------------------
DATA_PATH = "data/yield_df.csv"
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Unnamed: 0"])

countries = sorted(df["Area"].unique())
crops = sorted(df["Item"].unique())

# Year range for historical data and future scenarios
MIN_YEAR = int(df["Year"].min())  # e.g., 1990
HIST_MAX_YEAR = int(df["Year"].max())  # 2013 in this dataset
FUTURE_MAX_YEAR = 2030

# Percentile-based ranges for realism
RAIN_LOW, RAIN_HIGH = df["average_rain_fall_mm_per_year"].quantile([0.05, 0.95])
TEMP_LOW, TEMP_HIGH = df["avg_temp"].quantile([0.05, 0.95])
PEST_LOW, PEST_HIGH = df["pesticides_tonnes"].quantile([0.05, 0.95])

RAIN_MED = df["average_rain_fall_mm_per_year"].median()
TEMP_MED = df["avg_temp"].median()
PEST_MED = df["pesticides_tonnes"].median()

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
page = st.sidebar.radio(
    "Navigation", ["Single Prediction", "What-if Analysis", "Model KPIs"]
)

# Small helper to flag extreme values
def flag_extremes(rainfall, temp, pesticides, container):
    messages = []
    if rainfall <= RAIN_LOW + 0.05 * (RAIN_HIGH - RAIN_LOW):
        messages.append("very low rainfall")
    if rainfall >= RAIN_HIGH - 0.05 * (RAIN_HIGH - RAIN_LOW):
        messages.append("very high rainfall")
    if temp <= TEMP_LOW + 0.05 * (TEMP_HIGH - TEMP_LOW):
        messages.append("very low temperature")
    if temp >= TEMP_HIGH - 0.05 * (TEMP_HIGH - TEMP_LOW):
        messages.append("very high temperature")
    if pesticides <= PEST_LOW + 0.05 * (PEST_HIGH - PEST_LOW):
        messages.append("very low pesticide use")
    if pesticides >= PEST_HIGH - 0.05 * (PEST_HIGH - PEST_LOW):
        messages.append("very high pesticide use")

    if messages:
        container.warning(
            "Inputs are at the edge of the historical range ("
            + ", ".join(messages)
            + "). Predictions here may be less reliable."
        )


# --------------------------------------------------
# PAGE 1 — Single Prediction
# --------------------------------------------------
if page == "Single Prediction":

    st.subheader("🔮 Single Yield Prediction")

    col1, col2 = st.columns(2)

    with col1:
        area = st.selectbox("Country", countries)
        item = st.selectbox("Crop Type", crops)
        year = st.slider("Year", MIN_YEAR, FUTURE_MAX_YEAR, 2008)

    with col2:
        rainfall = st.number_input(
            "Average rainfall (mm/year)",
            float(RAIN_LOW),
            float(RAIN_HIGH),
            float(RAIN_MED),
        )

        temp = st.number_input(
            "Average temperature (°C)",
            float(TEMP_LOW),
            float(TEMP_HIGH),
            float(TEMP_MED),
        )

        pesticides = st.number_input(
            "Pesticides (tonnes)",
            float(PEST_LOW),
            float(PEST_HIGH),
            float(PEST_MED),
        )

    st.caption(
        f"Historical data available for {MIN_YEAR}–{HIST_MAX_YEAR}. "
        "Years after 2013 are forward-looking scenario predictions based on past patterns."
    )

    # Alert if inputs are at extremes of the range
    flag_extremes(rainfall, temp, pesticides, st)

    if st.button("Predict Yield") and model is not None:

        input_df = pd.DataFrame(
            [
                {
                    "Area": area,
                    "Item": item,
                    "Year": year,
                    "average_rain_fall_mm_per_year": rainfall,
                    "pesticides_tonnes": pesticides,
                    "avg_temp": temp,
                }
            ]
        )

        pred = model.predict(input_df)[0]

        st.success(f"### 🌱 Predicted Yield: **{pred:,.0f} hg/ha**")
        st.caption("Model: Random Forest Regressor + full preprocessing pipeline.")

# --------------------------------------------------
# PAGE 2 — What-If Analysis
# --------------------------------------------------
elif page == "What-if Analysis":

    st.subheader("🧪 What-If Scenario Analysis")

    base_area = st.selectbox("Country", countries)
    base_item = st.selectbox("Crop Type", crops)
    base_year = st.slider("Year", MIN_YEAR, FUTURE_MAX_YEAR, 2008)

    col1, col2, col3 = st.columns(3)

    with col1:
        rainfall = st.slider(
            "Rainfall (mm/year)",
            float(RAIN_LOW),
            float(RAIN_HIGH),
            float(RAIN_MED),
        )

    with col2:
        temp = st.slider(
            "Temperature (°C)",
            float(TEMP_LOW),
            float(TEMP_HIGH),
            float(TEMP_MED),
        )

    with col3:
        # Slider in log scale within realistic pesticide range
        pesticides_log = st.slider(
            "Pesticides (log scale)",
            float(np.log1p(PEST_LOW)),
            float(np.log1p(PEST_HIGH)),
            float(np.log1p(PEST_MED)),
        )
        pesticides = np.expm1(pesticides_log)

    st.caption(
        f"Model trained on historical years {MIN_YEAR}–{HIST_MAX_YEAR}. "
        "Future years (after 2013) represent hypothetical what-if scenarios."
    )

    flag_extremes(rainfall, temp, pesticides, st)

    if model is not None:

        input_df = pd.DataFrame(
            [
                {
                    "Area": base_area,
                    "Item": base_item,
                    "Year": base_year,
                    "average_rain_fall_mm_per_year": rainfall,
                    "pesticides_tonnes": pesticides,
                    "avg_temp": temp,
                }
            ]
        )

        pred = model.predict(input_df)[0]

        st.metric("Predicted Yield (hg/ha)", f"{pred:,.0f}")
        st.caption(
            "Adjust sliders to explore how realistic changes in climate and pesticide "
            "usage could influence future yields."
        )

# --------------------------------------------------
# PAGE 3 — Model KPIs
# --------------------------------------------------
else:

    st.subheader("📈 Model Performance & KPIs")

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    sample = df.sample(3000, random_state=42)
    X_sample = sample.drop(columns=["hg/ha_yield"])
    y_sample = sample["hg/ha_yield"]

    if model is not None:
        preds = model.predict(X_sample)
        rmse = np.sqrt(mean_squared_error(y_sample, preds))
        mae = mean_absolute_error(y_sample, preds)
        r2 = r2_score(y_sample, preds)

        st.write(f"**RMSE:** {rmse:,.0f}")
        st.write(f"**MAE:**  {mae:,.0f}")
        st.write(f"**R²:**   {r2:.3f}")

        st.markdown(
            """
        **Interpretation**
        - Lower RMSE & MAE → more accurate predictions  
        - Higher R² → model explains a greater share of yield variance  
        - Random Forest achieved the best performance in training  

        **Business KPIs (Extrinsic)**
        - Improved fertilizer allocation  
        - Reduction in planning errors  
        - Higher yield for resource-constrained farmers  
        - Increased adoption of advisory services  
        """
        )
