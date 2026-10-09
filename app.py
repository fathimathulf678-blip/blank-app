
import streamlit as st
import pandas as pd
import zipfile
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Used Car Price Prediction", page_icon="🚗")

st.title("🚗 Used Car Price Prediction")
st.write("Predict the estimated price of a used car using Machine Learning.")

# Upload dataset
uploaded_file = st.file_uploader(
    "Upload your used_cars file",
    type=["csv", "zip"]
)

if uploaded_file is None:
    st.info("Please upload your used_cars ZIP or CSV file.")
    st.stop()

# Read CSV or ZIP file
try:
    file_bytes = uploaded_file.getvalue()

    if file_bytes.startswith(b"PK"):
        with zipfile.ZipFile(BytesIO(file_bytes)) as z:
            csv_files = [
                name for name in z.namelist()
                if name.lower().endswith(".csv")
            ]
            if not csv_files:
                st.error("No CSV file found inside the ZIP.")
                st.stop()

            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f, encoding="latin1")
    else:
        df = pd.read_csv(
            BytesIO(file_bytes), encoding="latin1"
        )

except Exception as e:
    st.error(f"Unable to read dataset: {e}")
    st.stop()

df.columns = df.columns.str.strip()

if "price" not in df.columns:
    st.error("The dataset must contain a price column.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# Clean price, mileage and year
data = df.copy()

data["price"] = pd.to_numeric(
    data["price"].astype(str)
    .str.replace(r"[$,]", "", regex=True),
    errors="coerce"
)

if "milage" in data.columns:
    data["milage"] = pd.to_numeric(
        data["milage"].astype(str)
        .str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce"
    )

if "model_year" in data.columns:
    data["model_year"] = pd.to_numeric(
        data["model_year"], errors="coerce"
    )

data = data.dropna(subset=["price"])
data = data[data["price"] > 0]

# Dataset overview
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Cars", len(data))
col2.metric("Total Columns", len(data.columns))
col3.metric("Average Price", f"${data['price'].mean():,.0f}")

with st.expander("View Dataset"):
    st.dataframe(df, use_container_width=True)

st.subheader("Price Analysis")
st.bar_chart(data["price"].head(50))

if "brand" in data.columns:
    brand_prices = (
        data.groupby("brand")["price"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )
    st.write("Average Price by Brand")
    st.bar_chart(brand_prices)

# Prepare machine learning data
X = data.drop(columns=["price"]).dropna(axis=1, how="all")
y = data["price"]

if len(data) < 10:
    st.error("At least 10 valid records are required.")
    st.stop()

numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = [
    col for col in X.columns
    if col not in numeric_cols
]

transformers = []

if numeric_cols:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    transformers.append(
        ("numeric", numeric_pipeline, numeric_cols)
    )

if categorical_cols:
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers.append(
        ("categorical", categorical_pipeline, categorical_cols)
    )

preprocessor = ColumnTransformer(transformers=transformers)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

st.subheader("Train Machine Learning Model")

if st.button("Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with st.spinner("Training model..."):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        score = r2_score(y_test, predictions)

        st.success("Model trained successfully!")
        st.metric("Mean Absolute Error", f"${mae:,.0f}")
        st.metric("R² Score", f"{score:.3f}")

        # Train final model using all records
        model.fit(X, y)

        st.session_state["model"] = model
        st.session_state["columns"] = X.columns.tolist()
        st.session_state["numeric"] = numeric_cols
        st.session_state["data"] = data.copy()

# Prediction form
if "model" in st.session_state:
    st.subheader("Predict Used Car Price")

    model = st.session_state["model"]
    columns = st.session_state["columns"]
    numeric = st.session_state["numeric"]
    training_data = st.session_state["data"]

    with st.form("car_form"):
        car = {}
        left, right = st.columns(2)

        for i, col in enumerate(columns):
            panel = left if i % 2 == 0 else right

            with panel:
                label = col.replace("_", " ").title()

                if col in numeric:
                    values = pd.to_numeric(
                        training_data[col], errors="coerce"
                    ).dropna()

                    default = (
                        float(values.median())
                        if not values.empty else 0.0
                    )

                    car[col] = st.number_input(
                        label, value=default
                    )
                else:
                    options = sorted(
                        training_data[col]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )

                    if not options:
                        options = ["Unknown"]

                    car[col] = st.selectbox(label, options)

        submit = st.form_submit_button("Predict Price")

    if submit:
        input_data = pd.DataFrame([car], columns=columns)
        price = model.predict(input_data)[0]

        st.success(f"Estimated Car Price: ${price:,.0f}")
        st.caption(
            "This is an estimated price, not a guaranteed selling price."
        )
