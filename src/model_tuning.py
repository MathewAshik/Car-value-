from pathlib import Path
import joblib

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "Dataset.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")


# ============================================================
# 3. FEATURES
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
# 4. CLEAN DATA
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


df = df.dropna(subset=["price"])

df = df[df["price"] > 0]

df = df.drop_duplicates()


# ============================================================
# 5. INPUT AND TARGET
# ============================================================

X = df[features]

y = df[target]


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 7. PREPROCESSING
# ============================================================

numeric_features = [
    "year",
    "distance_travelled(kms)"
]

categorical_features = [
    "brand",
    "model_name",
    "fuel_type",
    "city"
]


numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),

    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


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
# 8. RANDOM FOREST
# ============================================================

random_forest = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)


pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        random_forest
    )
])


# ============================================================
# 9. PARAMETERS TO TEST
# ============================================================

parameters = {

    "model__n_estimators": [
        200,
        300,
        500
    ],

    "model__max_depth": [
        None,
        10,
        20,
        30
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ],

    "model__max_features": [
        "sqrt",
        "log2",
        1.0
    ]
}


# ============================================================
# 10. RANDOMIZED SEARCH
# ============================================================

search = RandomizedSearchCV(

    estimator=pipeline,

    param_distributions=parameters,

    n_iter=15,

    scoring="neg_mean_absolute_error",

    cv=5,

    random_state=42,

    n_jobs=-1,

    verbose=1
)


print("\nStarting Random Forest tuning...")

search.fit(
    X_train,
    y_train
)


# ============================================================
# 11. BEST PARAMETERS
# ============================================================

print("\n")
print("=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

print(search.best_params_)


# ============================================================
# 12. BEST MODEL
# ============================================================

best_model = search.best_estimator_


# ============================================================
# 13. PREDICTIONS
# ============================================================

predictions = best_model.predict(
    X_test
)


# ============================================================
# 14. EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


# ============================================================
# 15. FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("TUNED RANDOM FOREST RESULTS")
print("=" * 70)

print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R2   : {r2:.4f}")

print("=" * 70)
# ============================================================
# 16. TRAIN FINAL MODEL ON FULL DATA
# ============================================================

print("\nTraining final model on the complete dataset...")

best_model.fit(
    X,
    y
)


# ============================================================
# 17. SAVE FINAL MODEL
# ============================================================

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    exist_ok=True
)

MODEL_PATH = MODEL_DIR / "car_price_model_tuned.joblib"

joblib.dump(
    best_model,
    MODEL_PATH
)

print("\nFinal model saved successfully!")
print(f"Model location: {MODEL_PATH}")


# ============================================================
# 18. SAVE INPUT OPTIONS
# ============================================================

categorical_columns = [
    "brand",
    "model_name",
    "fuel_type",
    "city"
]

options = {}

for column in categorical_columns:

    options[column] = sorted(
        df[column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


OPTIONS_PATH = MODEL_DIR / "input_options.joblib"

joblib.dump(
    options,
    OPTIONS_PATH
)

print("Input options saved successfully!")