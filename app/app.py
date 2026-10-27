
from pathlib import Path
import re
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_PATH = DATA_DIR / "Dataset.csv"

MODEL_PATH = MODEL_DIR / "car_price_model_tuned.joblib"
OPTIONS_PATH = MODEL_DIR / "input_options.joblib"


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CarValue | Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 3. PREMIUM CARVALUE UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap'
    );

    :root {
        --bg: #080b12;
        --surface: #101722;
        --surface-light: #151f2e;
        --border: rgba(148, 163, 184, 0.20);
        --blue: #38bdf8;
        --blue-light: #7dd3fc;
        --text: #f1f5f9;
        --muted: #a8b6c9;
    }

    /* ========================================================
       GLOBAL BACKGROUND
    ======================================================== */

    .stApp {
        background:
            radial-gradient(
                ellipse at 15% 0%,
                rgba(14, 165, 233, 0.12),
                transparent 35%
            ),
            radial-gradient(
                ellipse at 90% 15%,
                rgba(37, 99, 235, 0.10),
                transparent 30%
            ),
            var(--bg);

        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 11, 18, 0.92);
    }

    [data-testid="stAppViewContainer"] {
        background: transparent;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ========================================================
       TYPOGRAPHY AND READABILITY
    ======================================================== */

    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
    }

    h1 {
        font-weight: 900 !important;
        letter-spacing: -1.2px !important;
    }

    h2 {
        font-weight: 800 !important;
        letter-spacing: -0.6px !important;
    }

    h3 {
        font-weight: 700 !important;
    }

    p, li {
        color: #d5deea;
        line-height: 1.75;
    }

    label {
        color: #e2e8f0 !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #d5deea;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: var(--muted) !important;
    }

    /* Expanders and their contents */

    [data-testid="stExpander"] {
        background: rgba(16, 23, 34, 0.75);
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
    }

    [data-testid="stExpander"] summary {
        color: #f1f5f9 !important;
        font-weight: 600;
    }

    [data-testid="stExpander"] summary:hover {
        color: var(--blue-light) !important;
    }

    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] li {
        color: #d5deea !important;
    }

    /* ========================================================
       SIDEBAR
    ======================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0d1420 0%,
            #090d15 100%
        );

        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] h1 {
        color: var(--blue-light) !important;
        font-size: 1.8rem !important;
        letter-spacing: -1px !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #d5deea !important;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 10px;
        transition:
            background 0.2s ease,
            transform 0.2s ease;
    }

    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(56, 189, 248, 0.09);
        transform: translateX(3px);
    }

    /* ========================================================
       METRIC CARDS
    ======================================================== */

    [data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            rgba(21, 31, 46, 0.98),
            rgba(13, 20, 32, 0.98)
        );

        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px 22px;

        box-shadow: 0 8px 26px rgba(0, 0, 0, 0.16);

        transition:
            transform 0.2s ease,
            border-color 0.2s ease,
            box-shadow 0.2s ease;

        animation: fadeUp 0.45s ease both;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.55);

        box-shadow:
            0 12px 30px rgba(14, 165, 233, 0.10);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #a8b6c9 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800 !important;
        letter-spacing: -0.7px;
    }

    [data-testid="stMetricDelta"] {
        font-weight: 600;
    }

    /* ========================================================
       BUTTONS
    ======================================================== */

    div.stButton > button {
        width: 100%;
        min-height: 46px;

        border-radius: 11px;
        border: 1px solid rgba(56, 189, 248, 0.45);

        background: linear-gradient(
            110deg,
            #0284c7,
            #2563eb
        );

        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease,
            filter 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        filter: brightness(1.10);

        box-shadow:
            0 8px 24px rgba(14, 165, 233, 0.22);

        border-color: var(--blue-light);
    }

    div.stButton > button:focus-visible {
        outline: 2px solid var(--blue-light);
        outline-offset: 3px;
    }

    /* ========================================================
       INPUTS AND SELECTBOXES
    ======================================================== */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background: #111a28 !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;

        transition:
            border-color 0.2s ease,
            box-shadow 0.2s ease;
    }

    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover,
    div[data-baseweb="textarea"] > div:hover {
        border-color: var(--blue) !important;
    }

    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="textarea"] > div:focus-within {
        border-color: var(--blue) !important;

        box-shadow:
            0 0 0 2px rgba(56, 189, 248, 0.12);
    }

    input, textarea {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }

    div[data-baseweb="select"] span {
        color: #f1f5f9;
    }

    /* Dropdown menu options */

    ul[data-baseweb="menu"] {
        background: #111a28 !important;
        border: 1px solid #334155;
    }

    li[role="option"] {
        color: #f1f5f9 !important;
    }

    li[role="option"]:hover {
        background: #1e293b !important;
    }

    /* ========================================================
       TABS
    ======================================================== */

    button[data-baseweb="tab"] {
        color: #a8b6c9;
        font-weight: 700;
    }

    button[data-baseweb="tab"]:hover {
        color: var(--blue-light);
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--blue) !important;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: var(--blue) !important;
    }

    /* ========================================================
       TABLES AND CHARTS
    ======================================================== */

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stPlotlyChart"],
    [data-testid="stPyplot"],
    [data-testid="stVegaLiteChart"],
    [data-testid="stArrowVegaLiteChart"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ========================================================
       ALERTS
    ======================================================== */

    [data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid var(--border);
    }

    /* ========================================================
       DIVIDERS
    ======================================================== */

    hr {
        border-color: rgba(148, 163, 184, 0.17) !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1.5rem !important;
    }

    /* ========================================================
       ANIMATIONS
    ======================================================== */

    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(8px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .stMainBlockContainer {
        animation: fadeUp 0.4s ease both;
    }

    /* ========================================================
       FOOTER
    ======================================================== */

    footer {
        visibility: hidden;
    }

    /* ========================================================
       ACCESSIBILITY
    ======================================================== */

    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            animation: none !important;
            transition: none !important;
            scroll-behavior: auto !important;
        }
    }

    /* ========================================================
       RESPONSIVE LAYOUT
    ======================================================== */

    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        h1 {
            font-size: 2rem !important;
        }

        [data-testid="stMetric"] {
            padding: 15px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 4. LOAD MODEL AND OPTIONS
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not OPTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Input options file not found: {OPTIONS_PATH}"
        )
    
    trained_model = joblib.load(MODEL_PATH)
    input_options = joblib.load(OPTIONS_PATH)

    return trained_model, input_options


# ============================================================
# 5. LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = [
        "year",
        "brand",
        "model_name",
        "price",
        "distance_travelled(kms)",
        "fuel_type",
        "city"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    return data


# ============================================================
# 6. INITIALIZE APPLICATION
# ============================================================

try:

    model, options = load_model()
    df = load_dataset()

except Exception as error:

    st.error("CarValue could not load its required files.")

    st.write(
        "Verify that Dataset.csv and both saved model files "
        "are present in the expected folders."
    )

    st.code(str(error))

    st.stop()


# ============================================================
# 7. INPUT OPTIONS
# ============================================================

def get_options(column):

    values = []

    if isinstance(options, dict):
        values = options.get(column, [])

    if isinstance(values, str):
        values = [values]

    if not isinstance(values, (list, tuple, set)):
        values = []

    values = [
        str(value)
        for value in values
        if pd.notna(value)
    ]

    if not values and column in df.columns:

        values = (
            df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    return sorted(set(values))


brands = get_options("brand")
fuel_types = get_options("fuel_type")
cities = get_options("city")
all_models = get_options("model_name")


if not brands or not fuel_types or not cities or not all_models:

    st.error(
        "The application could not find all required car options "
        "in the dataset or input options file."
    )

    st.stop()


# ============================================================
# 8. REUSABLE CAR INPUT FUNCTION
# ============================================================

def get_car_details(prefix):

    col1, col2 = st.columns(2)

    with col1:

        year = st.number_input(
            "Manufacturing Year",
            min_value=1990,
            max_value=2026,
            value=2021,
            step=1,
            key=f"{prefix}_year"
        )

    with col2:

        brand = st.selectbox(
            "Car Brand",
            brands,
            key=f"{prefix}_brand"
        )

    available_models = (
        df.loc[
            df["brand"].astype(str) == str(brand),
            "model_name"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    available_models = sorted(available_models)

    if not available_models:
        available_models = all_models

    model_name = st.selectbox(
        "Car Model",
        available_models,
        key=f"{prefix}_model"
    )

    col3, col4 = st.columns(2)

    with col3:

        fuel_type = st.selectbox(
            "Fuel Type",
            fuel_types,
            key=f"{prefix}_fuel"
        )

    with col4:

        city = st.selectbox(
            "City",
            cities,
            key=f"{prefix}_city"
        )

    distance = st.number_input(
        "Distance Travelled (km)",
        min_value=0,
        max_value=1000000,
        value=30000,
        step=1000,
        key=f"{prefix}_distance"
    )

    return {
        "year": int(year),
        "brand": brand,
        "model_name": model_name,
        "distance_travelled(kms)": int(distance),
        "fuel_type": fuel_type,
        "city": city
    }


# ============================================================
# 9. PREDICTION FUNCTION
# ============================================================

def predict_price(car_details):

    input_data = pd.DataFrame(
        [car_details]
    )

    prediction = float(
        model.predict(input_data)[0]
    )

    if not pd.notna(prediction):
        raise ValueError(
            "The model returned an invalid prediction."
        )

    if prediction < 0:
        raise ValueError(
            "The model returned a negative price."
        )

    return prediction


def format_inr(value):
    """Format a number using Indian digit grouping, e.g. ₹12,50,000."""
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return "₹0"

    sign = "-" if number < 0 else ""
    digits = str(abs(number))
    if len(digits) <= 3:
        grouped = digits
    else:
        last_three = digits[-3:]
        remaining = digits[:-3]
        groups = []
        while remaining:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        grouped = ",".join(groups + [last_three])
    return f"{sign}₹{grouped}"


def inr_axis(value, _position=None):
    """Compact INR labels for chart axes."""
    return format_inr(value)
# ============================================================
# 10. PREMIUM APPLICATION HEADER
# ============================================================

st.markdown("")

st.markdown(
    """
    <div style="padding: 10px 0;">
        <p style="color:#38bdf8; font-size:12px;
                  font-weight:bold; letter-spacing:3px;">
            INTELLIGENT AUTOMOTIVE VALUATION
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Know What Your Car Is Really Worth.")

st.markdown(
    "Discover estimated resale values, compare vehicles, "
    "and explore used-car market trends through machine learning."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.info("✦ ML-Powered Estimates")

with col2:
    st.info("₹ Indian Market")

with col3:
    st.info("◈ Interactive Analytics")

st.divider()
# ============================================================
# 11. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.title("🚗 CarValue")

    st.write("Navigation")

    page = st.radio(
        "Choose a section",
        [
            "Price Prediction",
            "Market Insights",
            "Car Comparison",
            "Feature Importance",
            "About the Project"
        ]
    )

    st.divider()

    st.caption(
        "Used Car Price Prediction using Machine Learning"
    )


# ============================================================
# 12. PRICE PREDICTION PAGE
# ============================================================

if page == "Price Prediction":

    from pathlib import Path
    import requests

    # --------------------------------------------------------
    # HELPER: AUTOMATIC ONLINE CAR IMAGE SEARCH
    # --------------------------------------------------------

    @st.cache_data(ttl=3600, show_spinner=False)
    def search_car_image(brand, model_name, year):
        """
        Search Wikimedia Commons for a reference image.
        Returns image URL, source page URL and image title.
        Returns None if no suitable result is available.
        """

        search_queries = [
            f'"{brand}" "{model_name}" {year} car',
            f'"{brand}" "{model_name}" car',
            f'{brand} {model_name} automobile',
        ]

        api_url = "https://commons.wikimedia.org/w/api.php"

        headers = {
            "User-Agent": "CarValueApp/1.0 (car reference image lookup)"
        }

        for search_query in search_queries:

            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": search_query,
                "gsrnamespace": 6,
                "gsrlimit": 8,
                "prop": "imageinfo",
                "iiprop": "url",
                "iiurlwidth": 800,
                "format": "json",
            }

            try:
                response = requests.get(
                    api_url,
                    params=params,
                    headers=headers,
                    timeout=8,
                )

                response.raise_for_status()
                data = response.json()

                pages = data.get("query", {}).get("pages", {})

                for page_data in pages.values():

                    title = page_data.get("title", "")
                    title_lower = title.lower()

                    image_info = page_data.get(
                        "imageinfo", []
                    )

                    if not image_info:
                        continue

                    # Avoid non-image file formats.
                    if not title_lower.endswith(
                        (".jpg", ".jpeg", ".png", ".webp")
                    ):
                        continue

                    # Prefer results whose title mentions the brand
                    # and model rather than unrelated vehicles.
                    if (
                        brand.lower() not in title_lower
                        or model_name.lower() not in title_lower
                    ):
                        continue

                    image_url = image_info[0].get("thumburl")

                    if not image_url:
                        image_url = image_info[0].get("url")

                    if image_url:
                        return {
                            "image_url": image_url,
                            "source_url": image_info[0].get(
                                "descriptionurl",
                                "https://commons.wikimedia.org/",
                            ),
                            "title": title,
                        }

            except Exception:
                continue

        return None

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------

    st.title("🚘 Car Price Prediction")

    st.caption(
        "Discover the estimated resale value of your vehicle "
        "using machine learning."
    )

    st.divider()

    # --------------------------------------------------------
    # VEHICLE INPUTS
    # --------------------------------------------------------

    st.subheader("🚗 Vehicle Specifications")

    st.write(
        "Enter the vehicle details below to calculate "
        "an estimated resale value."
    )

    car = get_car_details("prediction")

    st.divider()

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    predict_clicked = st.button(
        "💰 Estimate Car Value",
        type="primary",
        key="predict_single",
        use_container_width=True,
    )

    if predict_clicked:

        try:
            prediction = predict_price(car)

            st.session_state["last_prediction"] = {
                **car,
                "prediction": prediction,
            }

            st.session_state["prediction_completed"] = True

        except Exception as error:

            st.session_state["prediction_completed"] = False

            st.error("Price prediction failed.")

            with st.expander("View technical error details"):
                st.code(str(error))

    # --------------------------------------------------------
    # DISPLAY THE MOST RECENT PREDICTION
    # --------------------------------------------------------

    if (
        st.session_state.get("prediction_completed", False)
        and "last_prediction" in st.session_state
    ):

        result = st.session_state["last_prediction"]
        prediction = result["prediction"]

        st.divider()

        # ----------------------------------------------------
        # CAR IMAGE AND ESTIMATED PRICE
        # ----------------------------------------------------

        st.subheader("💎 Your Estimated Car Value")

        image_col, price_col = st.columns([1.15, 1])

        with image_col:

            st.markdown("#### 🚘 Vehicle Reference Photo")

            with st.spinner("Searching for a car reference image..."):

                image_result = search_car_image(
                    str(result["brand"]),
                    str(result["model_name"]),
                    str(result["year"]),
                )

            if image_result:

                st.image(
                    image_result["image_url"],
                    use_container_width=True,
                )

                st.caption(
                    "Reference image only. The pictured vehicle "
                    "may differ in year, variant, or colour."
                )

                st.markdown(
                    f"[View image source on Wikimedia Commons]"
                    f"({image_result['source_url']})"
                )

            else:

                st.info(
                    "🔎 No matching reference photo was found. "
                    "Your price prediction is still available."
                )

                st.caption(
                    "Try searching online for the selected brand "
                    "and model to see reference photos."
                )

        with price_col:

            with st.container(border=True):

                st.caption("CARVALUE ESTIMATED RESALE PRICE")

                st.markdown(
                    f"## {format_inr(prediction)}"
                )

                st.success(
                    "Your vehicle valuation has been generated "
                    "successfully."
                )

                st.caption(
                    "This amount is a model-generated estimate "
                    "based on the vehicle details you provided."
                )

        st.divider()

        # ----------------------------------------------------
        # VEHICLE SUMMARY
        # ----------------------------------------------------

        st.subheader("🚘 Vehicle Summary")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            with st.container(border=True):

                st.markdown("#### 🚗 Vehicle Identity")

                st.write(f"**Brand:** {result['brand']}")

                st.write(f"**Model:** {result['model_name']}")

                st.write(
                    f"**Manufacturing Year:** {result['year']}"
                )

        with summary_col2:

            with st.container(border=True):

                st.markdown("#### ⚙️ Vehicle Details")

                st.write(
                    f"**Distance Travelled:** "
                    f"{result['distance_travelled(kms)']:,} km"
                )

                st.write(f"**Fuel Type:** {result['fuel_type']}")

                st.write(f"**City:** {result['city']}")

        # ----------------------------------------------------
        # VALUATION DISCLAIMER
        # ----------------------------------------------------

        st.subheader("📌 Understanding Your Valuation")

        st.info(
            "This is an estimated resale value, not a guaranteed "
            "selling price. The actual market price may vary depending "
            "on the car's condition, service history, ownership, "
            "location, and current market demand."
        )

        # ----------------------------------------------------
        # PREDICT ANOTHER VEHICLE
        # ----------------------------------------------------

        st.divider()

        st.subheader("🔄 Explore Another Vehicle")

        st.write(
            "Change the vehicle specifications above and run "
            "another prediction to estimate a different car's value."
        )

        if st.button(
            "Clear Previous Result",
            key="clear_prediction",
        ):

            st.session_state.pop("last_prediction", None)
            st.session_state["prediction_completed"] = False

            st.rerun()
# ============================================================
# 13. MARKET INSIGHTS PAGE
# ============================================================

elif page == "Market Insights":

    # --------------------------------------------------------
    # DASHBOARD HEADER
    # --------------------------------------------------------

    st.title("📊 Market Intelligence")

    st.caption(
        "Explore used-car prices, compare brands, analyse "
        "manufacturing-year trends, and understand mileage patterns."
    )

    st.divider()

    # --------------------------------------------------------
    # PREPARE DATA SAFELY
    # --------------------------------------------------------

    required_columns = [
        "price",
        "brand",
        "year",
        "distance_travelled(kms)"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        st.error(
            "Market Insights cannot be displayed because these "
            "dataset columns are missing: "
            + ", ".join(missing_columns)
        )

    else:

        chart_df = df.copy()

        chart_df["price"] = pd.to_numeric(
            chart_df["price"],
            errors="coerce"
        )

        chart_df["year"] = pd.to_numeric(
            chart_df["year"],
            errors="coerce"
        )

        chart_df["distance_travelled(kms)"] = pd.to_numeric(
            chart_df["distance_travelled(kms)"],
            errors="coerce"
        )

        chart_df = chart_df.dropna(
            subset=["price"]
        )

        chart_df = chart_df[
            chart_df["price"] >= 0
        ].copy()

        if chart_df.empty:

            st.warning(
                "There are no valid vehicle prices available "
                "for market analysis."
            )

        else:

            # ------------------------------------------------
            # KEY MARKET METRICS
            # ------------------------------------------------

            average_price = chart_df["price"].mean()
            minimum_price = chart_df["price"].min()
            maximum_price = chart_df["price"].max()

            valid_distances = chart_df[
                "distance_travelled(kms)"
            ].dropna()

            average_distance = (
                valid_distances.mean()
                if not valid_distances.empty
                else None
            )

            unique_brands = chart_df["brand"].nunique(
                dropna=True
            )

            st.subheader("📌 Market Overview")

            st.caption(
                "Key statistics calculated from your available dataset."
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "🚘 Cars Analysed",
                    f"{len(chart_df):,}"
                )

            with col2:
                st.metric(
                    "💰 Average Price",
                    format_inr(average_price)
                )

            with col3:
                st.metric(
                    "🏷️ Unique Brands",
                    f"{unique_brands:,}"
                )

            col4, col5, col6 = st.columns(3)

            with col4:
                st.metric(
                    "🔻 Lowest Price",
                    format_inr(minimum_price)
                )

            with col5:
                st.metric(
                    "🔺 Highest Price",
                    format_inr(maximum_price)
                )

            with col6:
                st.metric(
                    "🛣️ Average Distance",
                    (
                        f"{average_distance:,.0f} km"
                        if average_distance is not None
                        else "N/A"
                    )
                )

            st.divider()

            # ------------------------------------------------
            # PRICE DISTRIBUTION
            # ------------------------------------------------

            st.subheader("💰 Price Distribution")

            st.write(
                "See how many vehicles fall into each price range."
            )

            max_price = float(
                chart_df["price"].max()
            )

            if max_price <= 500000:

                edges = [
                    0, 50000, 100000, 200000,
                    300000, 500000, float("inf")
                ]

                labels = [
                    "₹0–50k",
                    "₹50k–1L",
                    "₹1L–2L",
                    "₹2L–3L",
                    "₹3L–5L",
                    "Above ₹5L"
                ]

            elif max_price <= 2000000:

                edges = [
                    0, 100000, 200000, 500000,
                    750000, 1000000, 1500000,
                    2000000, float("inf")
                ]

                labels = [
                    "₹0–1L",
                    "₹1L–2L",
                    "₹2L–5L",
                    "₹5L–7.5L",
                    "₹7.5L–10L",
                    "₹10L–15L",
                    "₹15L–20L",
                    "Above ₹20L"
                ]

            else:

                edges = [
                    0, 100000, 300000, 500000,
                    1000000, 2000000, 5000000,
                    10000000, float("inf")
                ]

                labels = [
                    "₹0–1L",
                    "₹1L–3L",
                    "₹3L–5L",
                    "₹5L–10L",
                    "₹10L–20L",
                    "₹20L–50L",
                    "₹50L–1Cr",
                    "Above ₹1Cr"
                ]

            chart_df["Price Range"] = pd.cut(
                chart_df["price"],
                bins=edges,
                labels=labels,
                right=False,
                include_lowest=True
            )

            distribution = (
                chart_df["Price Range"]
                .value_counts(sort=False)
                .reindex(labels, fill_value=0)
            )

            fig, ax = plt.subplots(
                figsize=(11, 4.5)
            )

            distribution.plot(
                kind="bar",
                ax=ax,
                width=0.7
            )

            ax.set_title(
                "Vehicle Listings by Price Range",
                fontsize=14,
                fontweight="bold",
                pad=16
            )

            ax.set_xlabel("Price Range (INR)")
            ax.set_ylabel("Number of Vehicles")

            ax.tick_params(
                axis="x",
                labelrotation=25
            )

            ax.grid(
                axis="y",
                alpha=0.25
            )

            ax.set_axisbelow(True)

            fig.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

            st.divider()

            # ------------------------------------------------
            # AVERAGE PRICE BY BRAND
            # ------------------------------------------------

            st.subheader("🏷️ Average Price by Brand")

            st.write(
                "Compare average vehicle prices across brands."
            )

            brand_prices = (
                chart_df.dropna(subset=["brand"])
                .groupby("brand")["price"]
                .mean()
                .sort_values(ascending=False)
            )

            if brand_prices.empty:

                st.info(
                    "Brand-wise pricing information is unavailable."
                )

            else:

                display_brands = (
                    brand_prices.head(15)
                    .sort_values(ascending=True)
                )

                fig, ax = plt.subplots(
                    figsize=(
                        11,
                        max(4.5, len(display_brands) * 0.35)
                    )
                )

                display_brands.plot(
                    kind="barh",
                    ax=ax,
                    width=0.7
                )

                ax.set_title(
                    "Top 15 Brands by Average Price",
                    fontsize=14,
                    fontweight="bold",
                    pad=16
                )

                ax.set_xlabel("Average Price (INR)")
                ax.set_ylabel("Brand")

                ax.xaxis.set_major_formatter(
                    FuncFormatter(inr_axis)
                )

                ax.grid(
                    axis="x",
                    alpha=0.25
                )

                ax.set_axisbelow(True)

                fig.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

            st.divider()

            # ------------------------------------------------
            # PRICE BY MANUFACTURING YEAR
            # ------------------------------------------------

            st.subheader("📅 Manufacturing Year Analysis")

            st.write(
                "Explore how average prices vary by vehicle age."
            )

            yearly_data = chart_df.dropna(
                subset=["year"]
            )

            yearly_prices = (
                yearly_data.groupby("year")["price"]
                .mean()
                .sort_index()
            )

            if yearly_prices.empty:

                st.info(
                    "Year-wise pricing information is unavailable."
                )

            else:

                fig, ax = plt.subplots(
                    figsize=(11, 4.5)
                )

                ax.plot(
                    yearly_prices.index,
                    yearly_prices.values,
                    marker="o",
                    markersize=5,
                    linewidth=2
                )

                ax.set_title(
                    "Average Price by Manufacturing Year",
                    fontsize=14,
                    fontweight="bold",
                    pad=16
                )

                ax.set_xlabel("Manufacturing Year")
                ax.set_ylabel("Average Price (INR)")

                ax.yaxis.set_major_formatter(
                    FuncFormatter(inr_axis)
                )

                ax.grid(alpha=0.25)

                fig.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

            st.divider()

            # ------------------------------------------------
            # DISTANCE VS PRICE
            # ------------------------------------------------

            st.subheader("🛣️ Distance Travelled vs Price")

            st.write(
                "Examine the relationship between vehicle mileage "
                "and resale prices."
            )

            scatter_data = chart_df[
                [
                    "distance_travelled(kms)",
                    "price"
                ]
            ].dropna()

            if scatter_data.empty:

                st.info(
                    "Valid distance and price data are unavailable "
                    "for this chart."
                )

            else:

                if len(scatter_data) > 5000:

                    scatter_data = scatter_data.sample(
                        n=5000,
                        random_state=42
                    )

                fig, ax = plt.subplots(
                    figsize=(11, 4.8)
                )

                ax.scatter(
                    scatter_data["distance_travelled(kms)"],
                    scatter_data["price"],
                    alpha=0.45,
                    s=20
                )

                ax.set_title(
                    "Distance Travelled vs Vehicle Price",
                    fontsize=14,
                    fontweight="bold",
                    pad=16
                )

                ax.set_xlabel("Distance Travelled (km)")
                ax.set_ylabel("Price (INR)")

                ax.yaxis.set_major_formatter(
                    FuncFormatter(inr_axis)
                )

                ax.xaxis.set_major_formatter(
                    FuncFormatter(
                        lambda value, position: f"{value:,.0f}"
                    )
                )

                ax.grid(alpha=0.2)

                fig.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)

            # ------------------------------------------------
            # DISCLAIMER
            # ------------------------------------------------

            st.divider()

            st.caption(
                "ℹ️ These insights describe the available dataset. "
                "Historical prices do not guarantee current market "
                "prices or future resale values."
            )
# ============================================================
# 14. PREMIUM CAR COMPARISON PAGE
# ============================================================

elif page == "Car Comparison":

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------

    st.title("⚖️ Car Comparison Studio")

    st.caption(
        "Compare two vehicles side by side, evaluate their "
        "estimated resale values, and explore their specifications."
    )

    st.divider()

    # --------------------------------------------------------
    # VEHICLE SELECTION
    # --------------------------------------------------------

    st.subheader("🚘 Select Your Vehicles")

    st.write(
        "Enter the details for both vehicles. "
        "You can compare different brands, models, years, and mileage."
    )

    tab_a, tab_b = st.tabs(
        ["🚗 Car A", "🚙 Car B"]
    )

    with tab_a:

        st.subheader("Vehicle A")

        car_a = get_car_details("comparison_a")

    with tab_b:

        st.subheader("Vehicle B")

        car_b = get_car_details("comparison_b")

    st.divider()

    # --------------------------------------------------------
    # COMPARE BUTTON
    # --------------------------------------------------------

    compare_clicked = st.button(
        "⚖️ COMPARE VEHICLES",
        type="primary",
        key="compare_cars",
        use_container_width=True
    )

    if compare_clicked:

        try:

            # Calculate prices using the existing model.
            price_a = float(predict_price(car_a))
            price_b = float(predict_price(car_b))

            st.session_state["last_comparison"] = {
                "car_a": car_a,
                "car_b": car_b,
                "price_a": price_a,
                "price_b": price_b
            }

            # ------------------------------------------------
            # PRICE COMPARISON
            # ------------------------------------------------

            st.divider()

            st.subheader("💰 Estimated Resale Value")

            st.caption(
                "Estimated prices generated by your trained "
                "machine-learning model."
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    label=(
                        f"🚗 {car_a['brand']} "
                        f"{car_a['model_name']}"
                    ),
                    value=format_inr(price_a)
                )

            with col2:

                st.metric(
                    label=(
                        f"🚙 {car_b['brand']} "
                        f"{car_b['model_name']}"
                    ),
                    value=format_inr(price_b)
                )

            # ------------------------------------------------
            # PRICE DIFFERENCE
            # ------------------------------------------------

            difference = abs(price_a - price_b)

            st.divider()

            st.subheader("📊 Price Difference")

            diff_col1, diff_col2 = st.columns(2)

            with diff_col1:

                st.metric(
                    "Absolute Price Difference",
                    format_inr(difference)
                )

            with diff_col2:

                if price_a > price_b:

                    higher_value_car = "Car A"

                    st.metric(
                        "Higher Estimated Value",
                        "Car A",
                        help=(
                            "Car A has the higher model-predicted "
                            "resale price."
                        )
                    )

                elif price_b > price_a:

                    higher_value_car = "Car B"

                    st.metric(
                        "Higher Estimated Value",
                        "Car B",
                        help=(
                            "Car B has the higher model-predicted "
                            "resale price."
                        )
                    )

                else:

                    higher_value_car = None

                    st.metric(
                        "Estimated Value",
                        "Equal"
                    )

            if price_a > price_b:

                st.success(
                    f"Car A has a higher estimated resale value "
                    f"by {format_inr(difference)}."
                )

            elif price_b > price_a:

                st.success(
                    f"Car B has a higher estimated resale value "
                    f"by {format_inr(difference)}."
                )

            else:

                st.info(
                    "Both vehicles have the same estimated resale value."
                )

            # ------------------------------------------------
            # VISUAL PRICE COMPARISON
            # ------------------------------------------------

            st.divider()

            st.subheader("📈 Visual Price Comparison")

            comparison_chart = pd.DataFrame(
                {
                    "Vehicle": [
                        f"{car_a['brand']} {car_a['model_name']}",
                        f"{car_b['brand']} {car_b['model_name']}"
                    ],
                    "Predicted Price": [
                        price_a,
                        price_b
                    ]
                }
            )

            st.bar_chart(
                comparison_chart.set_index("Vehicle"),
                y="Predicted Price",
                use_container_width=True
            )

            # ------------------------------------------------
            # SPECIFICATION COMPARISON
            # ------------------------------------------------

            st.divider()

            st.subheader("📋 Detailed Specification Comparison")

            comparison_table = pd.DataFrame(
                {
                    "Parameter": [
                        "Brand",
                        "Model",
                        "Manufacturing Year",
                        "Distance Travelled",
                        "Fuel Type",
                        "City",
                        "Predicted Resale Price"
                    ],
                    "🚗 Car A": [
                        car_a["brand"],
                        car_a["model_name"],
                        car_a["year"],
                        (
                            f"{car_a['distance_travelled(kms)']:,} km"
                        ),
                        car_a["fuel_type"],
                        car_a["city"],
                        format_inr(price_a)
                    ],
                    "🚙 Car B": [
                        car_b["brand"],
                        car_b["model_name"],
                        car_b["year"],
                        (
                            f"{car_b['distance_travelled(kms)']:,} km"
                        ),
                        car_b["fuel_type"],
                        car_b["city"],
                        format_inr(price_b)
                    ]
                }
            )

            st.dataframe(
                comparison_table,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # VEHICLE HIGHLIGHTS
            # ------------------------------------------------

            st.divider()

            st.subheader("✨ Vehicle Highlights")

            highlight_col1, highlight_col2 = st.columns(2)

            with highlight_col1:

                st.markdown("### 🚗 Car A")

                st.write(
                    f"**Brand:** {car_a['brand']}"
                )

                st.write(
                    f"**Model:** {car_a['model_name']}"
                )

                st.write(
                    f"**Manufacturing year:** {car_a['year']}"
                )

                st.write(
                    "**Distance travelled:** "
                    f"{car_a['distance_travelled(kms)']:,} km"
                )

            with highlight_col2:

                st.markdown("### 🚙 Car B")

                st.write(
                    f"**Brand:** {car_b['brand']}"
                )

                st.write(
                    f"**Model:** {car_b['model_name']}"
                )

                st.write(
                    f"**Manufacturing year:** {car_b['year']}"
                )

                st.write(
                    "**Distance travelled:** "
                    f"{car_b['distance_travelled(kms)']:,} km"
                )

            # ------------------------------------------------
            # SIMPLE FACTUAL COMPARISON
            # ------------------------------------------------

            st.divider()

            st.subheader("🏆 Quick Comparison")

            year_a = int(car_a["year"])
            year_b = int(car_b["year"])

            distance_a = float(
                car_a["distance_travelled(kms)"]
            )

            distance_b = float(
                car_b["distance_travelled(kms)"]
            )

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                if year_a > year_b:

                    st.success("📅 Car A is newer.")

                elif year_b > year_a:

                    st.success("📅 Car B is newer.")

                else:

                    st.info(
                        "📅 Both cars have the same manufacturing year."
                    )

            with result_col2:

                if distance_a < distance_b:

                    st.success(
                        "🛣️ Car A has travelled fewer kilometres."
                    )

                elif distance_b < distance_a:

                    st.success(
                        "🛣️ Car B has travelled fewer kilometres."
                    )

                else:

                    st.info(
                        "🛣️ Both cars have the same recorded distance."
                    )

            # ------------------------------------------------
            # DISCLAIMER
            # ------------------------------------------------

            st.divider()

            st.info(
                "The displayed prices are machine-learning estimates, "
                "not guaranteed selling prices. Actual resale values "
                "depend on condition, service history, ownership, "
                "location, and market demand. A higher estimated price "
                "does not necessarily mean a vehicle is the better "
                "purchase."
            )

        except Exception as error:

            st.error(
                "Unable to compare the selected vehicles. "
                "Please verify the details and try again."
            )

            with st.expander("View technical error details"):

                st.code(str(error))
# ============================================================
# 15. FEATURE IMPORTANCE PAGE
# ============================================================

elif page == "Feature Importance":

    st.header("🧠 Feature Importance")
    st.write(
        "Discover which factors influence your car's predicted price "
        "and understand how your machine learning model evaluates vehicles."
    )

    st.divider()

    try:
        # ----------------------------------------------------
        # 1. VALIDATE THE MODEL PIPELINE
        # ----------------------------------------------------

        if not hasattr(model, "named_steps"):
            raise ValueError(
                "The saved model is not a supported pipeline."
            )

        if (
            "preprocessor" not in model.named_steps
            or "model" not in model.named_steps
        ):
            raise ValueError(
                "The required model pipeline steps were not found."
            )

        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]

        if not hasattr(estimator, "feature_importances_"):
            raise ValueError(
                "The trained model does not provide feature importance."
            )

        # ----------------------------------------------------
        # 2. EXTRACT FEATURE IMPORTANCE
        # ----------------------------------------------------

        feature_names = list(
            preprocessor.get_feature_names_out()
        )

        importances = estimator.feature_importances_

        if len(feature_names) != len(importances):
            raise ValueError(
                "Feature names do not match the importance scores."
            )

        def clean_feature_name(name):
            name = str(name)

            name = re.sub(
                r"^(categorical|numeric)__",
                "",
                name
            )

            replacements = {
                "distance_travelled(kms)": "Distance Travelled (km)",
                "model_name_": "Model: ",
                "brand_": "Brand: ",
                "fuel_type_": "Fuel Type: ",
                "city_": "City: ",
                "year": "Manufacturing Year"
            }

            for old, new in replacements.items():
                name = name.replace(old, new)

            return name.replace("_", " ").strip()

        importance_df = pd.DataFrame({
            "Feature": [
                clean_feature_name(name)
                for name in feature_names
            ],
            "Importance": importances
        })

        importance_df["Importance"] = pd.to_numeric(
            importance_df["Importance"],
            errors="coerce"
        )

        importance_df = (
            importance_df
            .dropna(subset=["Importance"])
            .sort_values(
                by="Importance",
                ascending=False
            )
            .reset_index(drop=True)
        )

        if importance_df.empty:
            raise ValueError(
                "No valid feature importance scores were found."
            )

        # ----------------------------------------------------
        # 3. DASHBOARD METRICS
        # ----------------------------------------------------

        total_features = len(importance_df)

        top_feature = importance_df.iloc[0]["Feature"]

        top_score = float(
            importance_df.iloc[0]["Importance"]
        )

        total_importance = float(
            importance_df["Importance"].sum()
        )

        top_five_share = (
            importance_df.head(5)["Importance"].sum()
            / total_importance * 100
            if total_importance > 0
            else 0
        )

        st.subheader("📊 Model Overview")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                label="Features Analysed",
                value=f"{total_features:,}"
            )

        with metric2:
            st.metric(
                label="Most Important Feature",
                value=top_feature
            )

        with metric3:
            st.metric(
                label="Top 5 Importance Share",
                value=f"{top_five_share:.1f}%"
            )

        st.caption(
            "The Top 5 Importance Share represents the proportion "
            "of total feature-importance scores accounted for by the "
            "five highest-ranked features. It is not a measure of "
            "prediction accuracy."
        )

        st.divider()

        # ----------------------------------------------------
        # 4. TOP FEATURES CHART
        # ----------------------------------------------------

        st.subheader("🏆 What Matters Most?")

        chart_limit = min(15, total_features)

        chart_df = (
            importance_df
            .head(chart_limit)
            .sort_values(
                by="Importance",
                ascending=True
            )
        )

        fig, ax = plt.subplots(
            figsize=(12, max(5, len(chart_df) * 0.48))
        )

        bars = ax.barh(
            chart_df["Feature"],
            chart_df["Importance"]
        )

        ax.set_xlabel("Relative Importance Score")
        ax.set_ylabel("Vehicle Feature")
        ax.set_title(
            f"Top {chart_limit} Factors Influencing Model Predictions"
        )

        ax.grid(
            axis="x",
            alpha=0.25
        )

        ax.set_axisbelow(True)

        ax.bar_label(
            bars,
            fmt="%.3f",
            padding=4,
            fontsize=9
        )

        ax.tick_params(
            axis="y",
            labelsize=9
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        st.caption(
            "Higher scores indicate greater overall importance to this "
            "model. Feature importance does not show whether a feature "
            "increases or decreases a car's predicted price, and it does "
            "not establish causation."
        )

        st.divider()

        # ----------------------------------------------------
        # 5. FEATURE IMPORTANCE DETAILS
        # ----------------------------------------------------

        st.subheader("📋 Explore Feature Rankings")

        search_term = st.text_input(
            "🔎 Search for a feature",
            placeholder="For example: Brand, Year, Distance, Fuel Type"
        )

        filtered_df = importance_df.copy()

        if search_term.strip():
            filtered_df = filtered_df[
                filtered_df["Feature"].str.contains(
                    search_term.strip(),
                    case=False,
                    na=False
                )
            ]

        display_df = filtered_df.copy()

        display_df.insert(
            0,
            "Rank",
            range(1, len(display_df) + 1)
        )

        display_df["Importance"] = display_df[
            "Importance"
        ].map(
            lambda value: f"{value:.4f}"
        )

        display_df = display_df.rename(
            columns={
                "Feature": "Vehicle Feature",
                "Importance": "Importance Score"
            }
        )

        if display_df.empty:
            st.info(
                "No matching features were found. "
                "Try another search term."
            )
        else:
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # ----------------------------------------------------
        # 6. KEY TAKEAWAY
        # ----------------------------------------------------

        st.subheader("💡 Key Takeaway")

        st.info(
            f"**{top_feature}** has the highest feature-importance "
            f"score in this model ({top_score:.4f}). "
            "This means the model relies on it more than any other "
            "individual feature according to its importance measure. "
            "It does not mean this feature alone determines a car's price."
        )

    except Exception as error:

        st.error(
            "Feature importance could not be displayed."
        )

        st.code(str(error))

# ============================================================
# 16. ABOUT THE PROJECT PAGE
# ============================================================

elif page == "About the Project":

    from pathlib import Path
    import joblib

    # --------------------------------------------------------
    # 1. LOAD SAVED MODEL METRICS
    # --------------------------------------------------------

    BASE_DIR = Path(__file__).resolve().parents[1]
    METRICS_PATH = BASE_DIR / "models" / "model_metrics.joblib"

    metrics = {}

    if METRICS_PATH.exists():
        try:
            metrics = joblib.load(METRICS_PATH)
        except Exception as error:
            st.warning(
                f"Could not load model evaluation metrics: {error}"
            )
    else:
        st.info(
            "Model evaluation metrics have not been found. "
            "Run the training script to generate them."
        )

    # --------------------------------------------------------
    # 2. PAGE HEADER
    # --------------------------------------------------------

    st.title("🚗 About CarValue")

    st.write(
        "An intelligent used-car valuation platform designed "
        "to estimate resale prices using machine learning."
    )

    st.divider()

    # --------------------------------------------------------
    # 3. PROJECT OVERVIEW
    # --------------------------------------------------------

    st.subheader("🎯 Project Overview")

    st.write(
        """
        CarValue is a machine learning-based application that
        estimates the resale value of used cars based on their
        specifications.

        Built using Python and Streamlit, the application uses
        a saved regression pipeline to generate price predictions.
        It also provides market insights, vehicle comparisons,
        and feature-importance analysis to help users explore
        factors associated with used-car prices.
        """
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            label="Core Technology",
            value="Machine Learning"
        )

    with metric2:
        st.metric(
            label="Primary Purpose",
            value="Price Estimation"
        )

    with metric3:
        st.metric(
            label="User Interface",
            value="Streamlit"
        )

    st.divider()

    # --------------------------------------------------------
    # 4. KEY FEATURES
    # --------------------------------------------------------

    st.subheader("✨ Key Features")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("💰 Price Prediction")
            st.write(
                "Estimate the resale value of a used car "
                "using its selected specifications."
            )

        with st.container(border=True):
            st.subheader("📊 Market Insights")
            st.write(
                "Explore dataset-based price distributions, "
                "brand comparisons, and vehicle trends."
            )

        with st.container(border=True):
            st.subheader("📈 Feature Importance")
            st.write(
                "Examine the features that contribute most "
                "to the trained model's predictions."
            )

    with col2:
        with st.container(border=True):
            st.subheader("⚖️ Car Comparison")
            st.write(
                "Compare two vehicles using their predicted "
                "prices and selected specifications."
            )

        with st.container(border=True):
            st.subheader("🚘 Dynamic Vehicle Selection")
            st.write(
                "Select vehicle details to generate a "
                "prediction for the chosen configuration."
            )

        with st.container(border=True):
            st.subheader("📉 Visual Analytics")
            st.write(
                "Understand vehicle pricing patterns through "
                "charts and structured data tables."
            )

    st.divider()

    # --------------------------------------------------------
    # 5. TECHNOLOGY STACK
    # --------------------------------------------------------

    st.subheader("🛠️ Technology Stack")

    st.write(
        "CarValue combines data processing, machine learning, "
        "visualization, and application development technologies."
    )

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-learn",
            "Random Forest Regression",
            "Streamlit",
            "Joblib",
            "Git and GitHub"
        ],
        "Purpose": [
            "Core programming language",
            "Dataset manipulation and analysis",
            "Numerical computing",
            "Machine learning and preprocessing",
            "Regression model used by the project",
            "Interactive web application",
            "Saving and loading trained model objects",
            "Version control and source-code management"
        ]
    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # 6. HOW CARVALUE WORKS
    # --------------------------------------------------------

    st.subheader("⚙️ How CarValue Works")

    st.write(
        "The application follows a workflow from vehicle "
        "selection to estimated resale price."
    )

    workflow1, workflow2 = st.columns(2)

    with workflow1:
        with st.container(border=True):
            st.markdown("### 01 · Vehicle Details")
            st.write(
                "The user selects the relevant vehicle "
                "specifications through the application."
            )

        with st.container(border=True):
            st.markdown("### 02 · Data Preprocessing")
            st.write(
                "The saved pipeline prepares the input "
                "features for the trained estimator."
            )

    with workflow2:
        with st.container(border=True):
            st.markdown("### 03 · Price Prediction")
            st.write(
                "The trained regression model processes "
                "the vehicle information and estimates its price."
            )

        with st.container(border=True):
            st.markdown("### 04 · Results and Insights")
            st.write(
                "The application presents the estimate and "
                "provides comparison and market analysis tools."
            )

    st.divider()

    # --------------------------------------------------------
    # 7. MACHINE LEARNING MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader("🤖 Machine Learning Model Performance")

    st.write(
        """
        CarValue uses a Random Forest regression model within
        a preprocessing pipeline. The metrics below are loaded
        from the results saved during model training and describe
        performance on the held-out test dataset.
        """
    )

    if metrics:

        mae = metrics.get("mae")
        rmse = metrics.get("rmse")
        r2 = metrics.get("r2")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Mean Absolute Error (MAE)",
                f"₹{mae:,.2f}" if mae is not None else "N/A",
                help=(
                    "Average absolute difference between "
                    "actual and predicted prices. Lower is better."
                )
            )

        with col2:
            st.metric(
                "Root Mean Squared Error (RMSE)",
                f"₹{rmse:,.2f}" if rmse is not None else "N/A",
                help=(
                    "Penalizes larger prediction errors more strongly. "
                    "Lower is generally better."
                )
            )

        with col3:
            st.metric(
                "R² Score",
                f"{r2:.4f}" if r2 is not None else "N/A",
                help=(
                    "Measures how well the model explains variation "
                    "in test-set prices. Interpret alongside MAE and RMSE."
                )
            )

        st.divider()

        # Dataset and evaluation details
        st.subheader("📊 Training and Testing Details")

        detail1, detail2, detail3 = st.columns(3)

        with detail1:
            st.metric(
                "Total Valid Records",
                f"{metrics.get('total_records', 0):,}"
            )

        with detail2:
            st.metric(
                "Training Records",
                f"{metrics.get('training_records', 0):,}"
            )

        with detail3:
            st.metric(
                "Testing Records",
                f"{metrics.get('testing_records', 0):,}"
            )

        with st.expander("📘 Understanding These Metrics"):

            st.markdown(
                """
                **MAE — Mean Absolute Error**

                Measures the average absolute difference between
                actual and predicted car prices. It is expressed
                in the same currency units as the target price.
                Lower values indicate smaller average absolute errors.

                **RMSE — Root Mean Squared Error**

                Measures prediction error while penalizing large
                errors more heavily. It is expressed in the same
                units as the target price. Lower values are generally
                better.

                **R² — Coefficient of Determination**

                Measures how well the model explains the variation
                in test-set prices. A score closer to 1 generally
                indicates a better fit. A negative score is possible
                and indicates performance worse than predicting
                the test-set mean under this metric.

                **Important:** These metrics summarize performance
                on the particular test split used during training.
                They do not guarantee the accuracy of an individual
                car-price prediction.
                """
            )

        st.caption(
            "Evaluation results are loaded from "
            "models/model_metrics.joblib."
        )

    else:
        st.warning(
            "Model evaluation results are not available. "
            "Run your training script using "
            "'python src/train_model.py' from the project root, "
            "then refresh this page."
        )

    st.divider()

    # --------------------------------------------------------
    # 8. PROJECT OBJECTIVES
    # --------------------------------------------------------

    st.subheader("🎯 Project Objectives")

    st.markdown(
        """
        - Develop a machine learning application for used-car
          price estimation.
        - Provide an accessible interface for entering vehicle details.
        - Explore used-car price patterns through data visualization.
        - Enable side-by-side comparisons of selected vehicles.
        - Examine feature importance to better understand model behavior.
        - Present model predictions and supporting information
          through a single application.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # 9. FOOTER
    # --------------------------------------------------------

    st.success(
        "CarValue — Data-driven insights for used-car valuation."
    )

    st.caption(
        "Note: Predicted prices are estimates based on the trained "
        "model and available data. Actual resale prices may differ."
    )

# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.caption(
    "CarValue | Used Car Intelligence Platform | "
    "Machine Learning Project"
)
# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.caption(
    "CarValue | Used Car Intelligence Platform | "
    "Machine Learning Project"
)
