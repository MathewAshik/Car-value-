from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "Dataset.csv"


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# ============================================================
# 3. SELECT USEFUL FEATURES
# ============================================================

features = [
    "year",
    "brand",
    "model_name",
    "distance_travelled(kms)",
    "fuel_type",
    "city"
]

target = "price"


df = df[features + [target]].copy()


# ============================================================
# 4. CLEAN NUMERICAL DATA
# ============================================================

numeric_columns = [
    "year",
    "distance_travelled(kms)",
    "price"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Remove rows where price is missing
df = df.dropna(subset=["price"])


# Remove invalid prices
df = df[df["price"] > 0]


# Remove duplicate rows
df = df.drop_duplicates()


print(f"Clean dataset: {len(df)} rows")


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

CURRENT_YEAR = 2026

df["car_age"] = CURRENT_YEAR - df["year"]


# ============================================================
# 6. DEFINE FEATURES
# ============================================================

features = [
    "year",
    "car_age",
    "brand",
    "model_name",
    "distance_travelled(kms)",
    "fuel_type",
    "city"
]


X = df[features]

y = df[target]


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# ============================================================
# 8. DEFINE NUMERICAL & CATEGORICAL FEATURES
# ============================================================

numeric_features = [
    "year",
    "car_age",
    "distance_travelled(kms)"
]

categorical_features = [
    "brand",
    "model_name",
    "fuel_type",
    "city"
]


# ============================================================
# 9. NUMERICAL PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


# ============================================================
# 10. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),

    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])


# ============================================================
# 11. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),

    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ============================================================
# 12. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )
}


# ============================================================
# 13. TRAIN AND EVALUATE MODELS
# ============================================================

results = []


for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ])


    # Train model
    pipeline.fit(
        X_train,
        y_train
    )


    # Make predictions
    predictions = pipeline.predict(
        X_test
    )


    # Calculate MAE
    mae = mean_absolute_error(
        y_test,
        predictions
    )


    # Calculate RMSE
    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5


    # Calculate R²
    r2 = r2_score(
        y_test,
        predictions
    )


    # Store results
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


    print(f"MAE  : {mae:,.2f}")
    print(f"RMSE : {rmse:,.2f}")
    print(f"R2   : {r2:.4f}")


# ============================================================
# 14. CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)


# Sort by R²
results_df = results_df.sort_values(
    by="R2",
    ascending=False
)


# ============================================================
# 15. DISPLAY FINAL COMPARISON
# ============================================================

print("\n\n")

print("=" * 70)
print("                MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 16. BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

print("\n")
print("=" * 70)
print("                    BEST MODEL")
print("=" * 70)

print(f"Model : {best_model['Model']}")
print(f"MAE   : {best_model['MAE']:,.2f}")
print(f"RMSE  : {best_model['RMSE']:,.2f}")
print(f"R2    : {best_model['R2']:.4f}")

print("=" * 70)