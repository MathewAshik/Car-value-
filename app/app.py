
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="CarValue",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 CarValue")
st.subheader("Used Car Price Prediction")
st.write(
    "Enter the details of your car to estimate "
    "its price using Machine Learning."
)


@st.cache_resource
def load_model():
    model = joblib.load(
        MODEL_DIR / "car_price_model.joblib"
    )
    options = joblib.load(
        MODEL_DIR / "input_options.joblib"
    )
    return model, options


try:
    model, options = load_model()
except FileNotFoundError:
    st.error(
        "Model not found. Run python src/train_model.py first."
    )
    st.stop()


year = st.number_input(
    "Manufacturing Year",
    min_value=1990,
    max_value=2026,
    value=2020
)

brand = st.selectbox(
    "Car Brand",
    options["brand"]
)

model_name = st.selectbox(
    "Car Model",
    options["model_name"]
)

distance = st.number_input(
    "Kilometres Driven",
    min_value=0,
    max_value=1000000,
    value=30000,
    step=1000
)

fuel_type = st.selectbox(
    "Fuel Type",
    options["fuel_type"]
)

city = st.selectbox(
    "City",
    options["city"]
)


if st.button("Predict Car Price", type="primary"):

    input_data = pd.DataFrame([{
        "year": year,
        "brand": brand,
        "model_name": model_name,
        "distance_travelled(kms)": distance,
        "fuel_type": fuel_type,
        "city": city
    }])

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Price: {prediction:,.2f} "
        "(dataset price units)"
    )

    st.caption(
        "This is a machine-learning estimate, not a "
        "guaranteed resale or transaction price."
    )
