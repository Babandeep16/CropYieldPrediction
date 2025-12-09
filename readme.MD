#  Crop Yield Prediction – Decision Support System (DSS)

A full machine-learning powered **Decision Support System (DSS)** designed to help
farmers, agricultural officers, and planners estimate **crop yield (hg/ha)** based
on climate variables such as rainfall, temperature, and pesticide use.

This project was developed by  
**Scrum Team 3 – CropYieldPrediction**  
for **AI for Business Decisions & Transformation (Fall 2025)**.

---

##  Project Overview

Agricultural yield is influenced by multiple variables, including climate
conditions, resource usage, crop type, and regional factors. Farmers and
governments often struggle with:

- Uncertain yield outcomes  
- Inefficient resource allocation  
- Lack of accessible predictive tools  
- Climate variability and extreme weather  

To address this, we built a **Decision Support System** that combines:

- Machine learning  
- Data preprocessing & EDA  
- Scenario simulations  
- Explainability & fairness rules  
- A Streamlit-based interactive dashboard  

The DSS allows users to:

-  Predict yield for a given country/crop/year  
-  Perform **What-If Analysis** using climate sliders  
-  View **ML KPIs & performance metrics**  
-  Get warnings when climate values are unrealistic  
-  See **sanity-adjusted yields** based on agronomic rules  

---

##  Key Features

###  Machine Learning Pipeline
- Preprocessing with `ColumnTransformer`
- One-hot encoding for categorical features
- Scaling for numerical features
- Random Forest Regressor (best model)
- Added **Agronomic Sanity Layer** to correct unrealistic ML outputs

###  Data Source
FAO crop yield dataset (1990–2013) merged with:
- Rainfall data  
- Temperature data  
- Pesticide usage  

Stored in:  
`data/yield_df.csv`

###  Streamlit Decision Support System
- Single prediction interface  
- What-if scenario simulation  
- KPI monitoring  
- Warnings for extreme values  
- Raw vs sanity-adjusted predictions  

---

##  Repository Structure
CropYieldPrediction/
│
├── streamlit.py # Main Streamlit DSS application
├── src/
│ ├── data_preprocess.py # Data cleaning & merging scripts
│ ├── train_model.py # ML training pipeline
│ └── utils.py # Sanity rules & helper functions
│
├── models/
│ └── best_model.joblib # Final Random Forest model
│
├── data/ # Raw and processed datasets
│ ├── yield.csv
│ ├── rainfall.csv
│ ├── temp.csv
│ ├── pesticides.csv
│ └── yield_df.csv
│
├── dashboards/
│ └── mockups/ # Screenshots 
│
└── notebooks/
└── DDS.ipynb # EDA & Model Development Notebook




---

##  Installation & Setup

###  Clone the repository
```bash
git clone https://github.com/Babandeep16/CropYieldPrediction.git
cd CropYieldPrediction

### Create a clean environment
conda create -n cropyield python=3.10 "numpy<2" pandas scikit-learn streamlit -y
conda activate cropyield

### Install dependencies
pip install -r requirements.txt

### Run the DSS
streamlit run streamlit.py
