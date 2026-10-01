from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "car_price_model_tuned.joblib"


# ============================================================
# 2. LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# 3. GET FEATURE IMPORTANCE
# ============================================================

preprocessor = model.named_steps["preprocessor"]

random_forest = model.named_steps["model"]


# ============================================================
# 4. GET FEATURE NAMES
# ============================================================

feature_names = preprocessor.get_feature_names_out()


# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================

importance = random_forest.feature_importances_


importance_df = pd.DataFrame({

    "feature": feature_names,

    "importance": importance

})


# ============================================================
# 6. SORT FEATURES
# ============================================================

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)


# ============================================================
# 7. SHOW TOP 15 FEATURES
# ============================================================

top_features = importance_df.head(15)


print("\n" + "=" * 70)

print("TOP FEATURES USED BY RANDOM FOREST")

print("=" * 70)

print(
    top_features.to_string(
        index=False
    )
)


# ============================================================
# 8. CREATE GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top Features Influencing Car Price Prediction"
)

plt.tight_layout()


# ============================================================
# 9. SAVE GRAPH
# ============================================================

OUTPUT_PATH = (
    BASE_DIR
    / "models"
    / "feature_importance.png"
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print("\nFeature importance graph saved successfully!")

print(
    f"Location: {OUTPUT_PATH}"
)