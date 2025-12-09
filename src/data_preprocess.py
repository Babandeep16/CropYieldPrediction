import pandas as pd

def load_and_merge_data():
    # Load datasets
    yield_df = pd.read_csv("data/yield.csv")
    rainfall_df = pd.read_csv("data/rainfall.csv")
    temp_df = pd.read_csv("data/temp.csv")
    pesticide_df = pd.read_csv("data/pesticides.csv")

    # Merge datasets on common keys
    df = yield_df.merge(rainfall_df, on=["Area", "Year"], how="left")
    df = df.merge(temp_df, on=["Area", "Year"], how="left")
    df = df.merge(pesticide_df, on=["Area", "Year"], how="left")

    # Rename consistent columns
    df.rename(columns={
        "hg/ha_yield": "yield_hg_ha",
        "average_rain_fall_mm_per_year": "rainfall",
        "avg_temp": "temperature",
        "pesticides_tonnes": "pesticides"
    }, inplace=True)

    # Drop missing/duplicate rows
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    # Save final merged dataset
    df.to_csv("data/yield_df.csv", index=False)

    return df

if __name__ == "__main__":
    df = load_and_merge_data()
    print("Merged dataset shape:", df.shape)
