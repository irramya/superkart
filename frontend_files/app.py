# ============================================
# SuperKart Streamlit Frontend
# Uses the exact instructor/batch model schema.
# ============================================

import pandas as pd
import requests
import streamlit as st

# Both containers are attached to the same Docker network.
# The backend container is explicitly named "backend".
BACKEND_URL = "http://backend:7860"

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    layout="centered",
)

st.title("SuperKart Sales Prediction")
st.write(
    "Enter product and store information to predict sales revenue."
)

# ---------------------------------------------------------
# Online / Single Prediction
# ---------------------------------------------------------
st.subheader("Online Prediction")

product_weight = st.number_input(
    "Product Weight",
    min_value=0.0,
    value=12.66,
)

product_sugar_content = st.selectbox(
    "Product Sugar Content",
    [
        "Low Sugar",
        "No Sugar",
        "Regular",
    ],
)

product_allocated_area = st.number_input(
    "Product Allocated Area",
    min_value=0.0,
    max_value=1.0,
    value=0.027,
    format="%.3f",
)

product_mrp = st.number_input(
    "Product MRP",
    min_value=0.0,
    value=117.08,
)

store_size = st.selectbox(
    "Store Size",
    [
        "High",
        "Medium",
        "Small",
    ],
)

store_location_city_type = st.selectbox(
    "Store Location City Type",
    [
        "Tier 1",
        "Tier 2",
        "Tier 3",
    ],
)

store_type = st.selectbox(
    "Store Type",
    [
        "Departmental Store",
        "Food Mart",
        "Supermarket Type1",
        "Supermarket Type2",
        "Supermarket Type3",
    ],
)

product_id_char = st.selectbox(
    "Product ID Prefix",
    [
        "DR",
        "FD",
        "NC",
    ],
)

store_age_years = st.number_input(
    "Store Age (Years)",
    min_value=0,
    value=16,
    step=1,
)

product_type_category = st.selectbox(
    "Product Type Category",
    [
        "Perishables",
        "Non Perishables",
    ],
)

payload = {
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar_content,
    "Product_Allocated_Area": product_allocated_area,
    "Product_MRP": product_mrp,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_location_city_type,
    "Store_Type": store_type,
    "Product_Id_char": product_id_char,
    "Store_Age_Years": store_age_years,
    "Product_Type_Category": product_type_category,
}

if st.button("Predict"):
    try:
        response = requests.post(
            f"{BACKEND_URL}/v1/predict",
            json=payload,
            timeout=60,
        )

        if response.ok:
            prediction = response.json()["prediction"]
            st.success(
                f"Predicted Sales: {prediction:,.2f}"
            )
        else:
            st.error(response.text)

    except Exception as exception:
        st.error(
            f"Prediction request failed: {exception}"
        )


# ---------------------------------------------------------
# Batch Prediction
# ---------------------------------------------------------
st.divider()
st.subheader("Batch Prediction")

uploaded_file = st.file_uploader(
    "Upload Batch_Data_SuperKart-compatible CSV",
    type=["csv"],
)

if uploaded_file is not None and st.button("Predict Batch"):
    try:
        response = requests.post(
            f"{BACKEND_URL}/v1/predictbatch",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "text/csv",
                )
            },
            timeout=120,
        )

        if response.ok:
            prediction_dict = response.json()

            result = pd.DataFrame({
                "Row_Index": list(prediction_dict.keys()),
                "Predicted_Sales": list(prediction_dict.values()),
            })

            st.dataframe(result)

            st.download_button(
                "Download Predictions",
                result.to_csv(index=False).encode("utf-8"),
                file_name="superkart_batch_predictions.csv",
                mime="text/csv",
            )

        else:
            st.error(response.text)

    except Exception as exception:
        st.error(
            f"Batch request failed: {exception}"
        )
