from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "Dataset.csv"

MODEL_PATH = BASE_DIR / "models" / "car_price_model_tuned.joblib"


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

df = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)


# ============================================================
# SHOW PRICE INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("PRICE INFORMATION")
print("=" * 60)

print(df["price"].describe())


# ============================================================
# SHOW EXTREME PRICES
# ============================================================

print("\n" + "=" * 60)
print("HIGHEST PRICES")
print("=" * 60)

print(
    df[
        [
            "brand",
            "model_name",
            "year",
            "price"
        ]
    ]
    .sort_values(
        "price",
        ascending=False
    )
    .head(10)
)


# ============================================================
# SHOW LOWEST PRICES
# ============================================================

print("\n" + "=" * 60)
print("LOWEST PRICES")
print("=" * 60)

print(
    df[
        [
            "brand",
            "model_name",
            "year",
            "price"
        ]
    ]
    .sort_values(
        "price",
        ascending=True
    )
    .head(10)
)