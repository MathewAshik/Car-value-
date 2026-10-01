from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "Dataset.csv"


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CarValue",
    page_icon="🚗",
    layout="centered"
)


# ============================================================
# 3. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_DIR / "car_price_model_tuned.joblib"
    )

    options = joblib.load(
        MODEL_DIR / "input_options.joblib"
    )

    return model, options


# ============================================================
# 4. LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(DATA_PATH)


try:

    model, options = load_model()

    df = load_dataset()

except FileNotFoundError:

    st.error(
        "Required model or dataset files were not found."
    )

    st.stop()


# ============================================================
# 5. HEADER
# ============================================================

st.title("🚗 CarValue")

st.subheader(
    "Used Car Price Prediction"
)

st.write(
    "Estimate the market value of a used car "
    "using Machine Learning and historical market data."
)

st.divider()


# ============================================================
# 6. CAR DETAILS
# ============================================================

st.header("🚘 Enter Car Details")


# ------------------------------------------------------------
# Manufacturing Year
# ------------------------------------------------------------

year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2026,
    value=2020,
    step=1
)


# ------------------------------------------------------------
# Brand
# ------------------------------------------------------------

brand = st.selectbox(
    "Car Brand",
    options["brand"]
)


# ------------------------------------------------------------
# Model based on selected brand
# ------------------------------------------------------------

brand_models = (
    df[
        df["brand"].astype(str) == str(brand)
    ]["model_name"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

brand_models = sorted(brand_models)


if len(brand_models) > 0:

    model_name = st.selectbox(
        "Car Model",
        brand_models
    )

else:

    model_name = st.selectbox(
        "Car Model",
        options["model_name"]
    )


# ------------------------------------------------------------
# Distance
# ------------------------------------------------------------

distance = st.number_input(
    "Distance Travelled (km)",
    min_value=0,
    max_value=1000000,
    value=30000,
    step=1000
)


# ------------------------------------------------------------
# Fuel
# ------------------------------------------------------------

fuel_type = st.selectbox(
    "Fuel Type",
    options["fuel_type"]
)


# ------------------------------------------------------------
# City
# ------------------------------------------------------------

city = st.selectbox(
    "City",
    options["city"]
)


# ============================================================
# 7. PREDICT BUTTON
# ============================================================

st.divider()


predict_button = st.button(
    "🚗 Predict Car Price",
    use_container_width=True
)


# ============================================================
# 8. EVERYTHING BELOW RUNS ONLY AFTER PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Create input data
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        "year": year,

        "brand": brand,

        "model_name": model_name,

        "distance_travelled(kms)": distance,

        "fuel_type": fuel_type,

        "city": city

    }])


    # --------------------------------------------------------
    # Machine Learning Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.success(
        "Prediction generated successfully!"
    )

    st.metric(
        label="💰 Estimated Car Price",
        value=f"{prediction:,.0f}"
    )


    # ========================================================
    # COMPARABLE CAR ANALYSIS
    # ========================================================

    st.divider()

    st.header(
        "📊 Comparable Cars Analysis"
    )


    # --------------------------------------------------------
    # Find same brand + model
    # --------------------------------------------------------

    comparable_cars = df[
        (df["brand"].astype(str) == str(brand)) &
        (df["model_name"].astype(str) == str(model_name))
    ].copy()


    # --------------------------------------------------------
    # Create car age
    # --------------------------------------------------------

    comparable_cars["car_age"] = (
        2026 - comparable_cars["year"]
    )

    user_car_age = 2026 - year


    # --------------------------------------------------------
    # Try to match fuel type
    # --------------------------------------------------------

    fuel_matches = comparable_cars[
        comparable_cars["fuel_type"].astype(str)
        == str(fuel_type)
    ]


    if len(fuel_matches) >= 3:

        comparable_cars = fuel_matches


    # --------------------------------------------------------
    # Find closest cars
    # --------------------------------------------------------

    if len(comparable_cars) > 0:

        comparable_cars["age_difference"] = abs(
            comparable_cars["car_age"]
            - user_car_age
        )

        comparable_cars["distance_difference"] = abs(
            comparable_cars["distance_travelled(kms)"]
            - distance
        )


        comparable_cars = comparable_cars.sort_values(
            by=[
                "age_difference",
                "distance_difference"
            ]
        )


        # Keep the 10 closest cars

        comparable_cars = comparable_cars.head(10)


    # ========================================================
    # DISPLAY MARKET ANALYSIS
    # ========================================================

    if len(comparable_cars) > 0:

        average_price = (
            comparable_cars["price"].mean()
        )

        minimum_price = (
            comparable_cars["price"].min()
        )

        maximum_price = (
            comparable_cars["price"].max()
        )

        average_distance = (
            comparable_cars[
                "distance_travelled(kms)"
            ].mean()
        )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Comparable Cars",
                len(comparable_cars)
            )

        with col2:

            st.metric(
                "Average Comparable Price",
                f"{average_price:,.0f}"
            )


        col3, col4 = st.columns(2)

        with col3:

            st.metric(
                "Average Distance",
                f"{average_distance:,.0f} km"
            )

        with col4:

            st.metric(
                "Market Price Range",
                f"{minimum_price:,.0f} – {maximum_price:,.0f}"
            )


        # ====================================================
        # PREDICTION VS MARKET
        # ====================================================

        st.subheader(
            "🤖 ML Prediction vs Comparable Market"
        )


        difference = (
            prediction - average_price
        )


        if difference > 0:

            st.info(
                f"CarValue's prediction is "
                f"{difference:,.0f} price units above "
                f"the average of the closest comparable cars."
            )


        elif difference < 0:

            st.info(
                f"CarValue's prediction is "
                f"{abs(difference):,.0f} price units below "
                f"the average of the closest comparable cars."
            )


        else:

            st.info(
                "CarValue's prediction is approximately "
                "equal to the comparable-car average."
            )

        # ====================================================
        # PRICE COMPARISON CHART
        # ====================================================

        st.subheader(
            "📈 Price Comparison"
        )

        chart_data = pd.DataFrame({
            "Category": [
                "ML Prediction",
                "Comparable Average"
            ],
            "Price": [
                prediction,
                average_price
            ]
        })

        st.bar_chart(
            chart_data.set_index("Category")
        )
        # ====================================================
        # COMPARABLE CAR TABLE
        # ====================================================

        st.subheader(
            "🚗 Closest Comparable Cars"
        )


        display_columns = [
            "brand",
            "model_name",
            "year",
            "distance_travelled(kms)",
            "fuel_type",
            "city",
            "price"
        ]


        st.dataframe(
            comparable_cars[display_columns],
            use_container_width=True,
            hide_index=True
        )


    else:

        st.warning(
            "Not enough comparable cars were found "
            "for this selection."
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.caption(
        "CarValue's estimate is based on historical dataset "
        "patterns and comparable vehicles. Actual market "
        "prices may vary depending on condition, ownership "
        "history, maintenance, location and other factors."
    )

    # ========================================================
    # MODEL EXPLAINABILITY
    # ========================================================

    st.divider()

    st.header("🧠 What Influences the Prediction?")

    st.write(
        "The chart below shows which features the Random Forest "
        "model relies on most when learning used-car prices."
    )

    # Get the preprocessing and Random Forest components
    preprocessor = model.named_steps["preprocessor"]
    random_forest = model.named_steps["model"]

    # Get transformed feature names
    feature_names = (
        preprocessor.get_feature_names_out()
    )

    # Get feature importance
    feature_importance = (
        random_forest.feature_importances_
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importance
    })

    # Sort and keep top 10
    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(10)
    )

    # Display chart
    chart_data = (
        importance_df
        .set_index("Feature")
    )

    st.bar_chart(
        chart_data
    )

    st.caption(
        "Feature importance describes how much the trained "
        "Random Forest uses each feature when making predictions. "
        "It does not imply that a feature causes the price to change."
    )
# ============================================================
# 9. FOOTER
# ============================================================

st.divider()

st.caption(
    "CarValue • Machine Learning • Used Car Market Analysis"
)