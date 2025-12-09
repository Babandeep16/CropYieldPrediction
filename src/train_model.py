import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_crop_yield_model():
    df = pd.read_csv("data/yield_df.csv")

    # Features & target
    X = df[["Area", "Item", "Year", "rainfall", "pesticides", "temperature"]]
    y = df["yield_hg_ha"]

    # Define preprocessing
    numeric_features = ["Year", "rainfall", "pesticides", "temperature"]
    categorical_features = ["Area", "Item"]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Model pipeline
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "models/best_model.joblib")

    print("Model trained and saved successfully.")

if __name__ == "__main__":
    train_crop_yield_model()
