
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "Dataset.csv"
MODEL_DIR = BASE_DIR / "models"

FEATURES = [
    "year",
    "brand",
    "model_name",
    "distance_travelled(kms)",
    "fuel_type",
    "city",
]

NUMERIC = ["year", "distance_travelled(kms)"]
CATEGORICAL = ["brand", "model_name", "fuel_type", "city"]


def main():
    # 1. Load the original dataset
    df = pd.read_csv(DATA_PATH)

    # 2. Select the useful columns
    df = df[FEATURES + ["price"]].copy()

    # 3. Convert numerical values
    for column in NUMERIC + ["price"]:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # 4. Remove invalid records
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    df = df.drop_duplicates()

    # 5. Prepare input and output
    X = df[FEATURES]
    y = df["price"]

    # 6. Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 7. Preprocess numerical and categorical data
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="most_frequent"
        )),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL)
    ])

    # 8. Create the machine learning pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1
        ))
    ])

    # 9. Train the model
    print("Training CarValue model...")
    model.fit(X_train, y_train)

    # 10. Evaluate on unseen test data
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Absolute Error: {mae:,.4f}")
    print(f"R2 Score: {r2:.4f}")

    # 11. Save model and input options
    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(
        model,
        MODEL_DIR / "car_price_model.joblib"
    )

    options = {
        column: sorted(
            df[column].dropna().astype(str).unique().tolist()
        )
        for column in CATEGORICAL
    }

    joblib.dump(
        options,
        MODEL_DIR / "input_options.joblib"
    )

    print("Model saved successfully!")


if __name__ == "__main__":
    main()