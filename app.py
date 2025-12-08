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

st.markdown("""
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
""")


# --------------------------------------------------
# Load dataset for dropdown values
# --------------------------------------------------
DATA_PATH = "data/yield_df.csv"
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Unnamed: 0"])

countries = sorted(df["Area"].unique())
crops = sorted(df["Item"].unique())


# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Single Prediction", "What-if Analysis", "Model KPIs"]
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
        year = st.slider("Year", int(df["Year"].min()), int(df["Year"].max()), 2005)

    with col2:
        rainfall = st.number_input(
            "Average rainfall (mm/year)",
            float(df["average_rain_fall_mm_per_year"].min()),
            float(df["average_rain_fall_mm_per_year"].max()),
            float(df["average_rain_fall_mm_per_year"].median())
        )

        temp = st.number_input(
            "Average temperature (°C)",
            float(df["avg_temp"].min()),
            float(df["avg_temp"].max()),
            float(df["avg_temp"].median())
        )

        pesticides = st.number_input(
            "Pesticides (tonnes)",
            float(df["pesticides_tonnes"].min()),
            float(df["pesticides_tonnes"].max()),
            float(df["pesticides_tonnes"].median())
        )

    if st.button("Predict Yield") and model is not None:

        input_df = pd.DataFrame([{
            "Area": area,
            "Item": item,
            "Year": year,
            "average_rain_fall_mm_per_year": rainfall,
            "pesticides_tonnes": pesticides,
            "avg_temp": temp
        }])

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
    base_year = st.slider("Year", int(df["Year"].min()), int(df["Year"].max()), 2005)

    col1, col2, col3 = st.columns(3)

    with col1:
        rainfall = st.slider(
            "Rainfall (mm/year)",
            float(df["average_rain_fall_mm_per_year"].min()),
            float(df["average_rain_fall_mm_per_year"].max()),
            float(df["average_rain_fall_mm_per_year"].median())
        )

    with col2:
        temp = st.slider(
            "Temperature (°C)",
            float(df["avg_temp"].min()),
            float(df["avg_temp"].max()),
            float(df["avg_temp"].median())
        )

    with col3:
        # Slider in log scale → better for extreme pesticide values
        pesticides_log = st.slider(
            "Pesticides (log scale)",
            float(np.log1p(df["pesticides_tonnes"].min())),
            float(np.log1p(df["pesticides_tonnes"].max())),
            float(np.log1p(df["pesticides_tonnes"].median()))
        )
        pesticides = np.expm1(pesticides_log)

    if model is not None:

        input_df = pd.DataFrame([{
            "Area": base_area,
            "Item": base_item,
            "Year": base_year,
            "average_rain_fall_mm_per_year": rainfall,
            "pesticides_tonnes": pesticides,
            "avg_temp": temp
        }])

        pred = model.predict(input_df)[0]

        st.metric("Predicted Yield (hg/ha)", f"{pred:,.0f}")

        st.caption("Adjust sliders to understand how climate & pesticide changes influence yield.")



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

        st.markdown("""
        **Interpretation**
        - Lower RMSE & MAE → more accurate predictions  
        - Higher R² → model explains a greater share of yield variance  
        - Random Forest achieved the best performance in training  

        **Business KPIs (Extrinsic)**
        - Improved fertilizer allocation  
        - Reduction in planning errors  
        - Higher yield for resource-constrained farmers  
        - Increased adoption of advisory services  
        """)

