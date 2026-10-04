
# ============================================
# SuperKart Flask Prediction API
# ============================================

import joblib
import pandas as pd
from flask import Flask, request, jsonify


# ============================================
# Configuration
# ============================================

MODEL_FILE = "superkart_model.joblib"

# Create Flask application
superkart_api = Flask(__name__)

# Load the complete preprocessing + model pipeline
model = joblib.load(MODEL_FILE)

# The serialized model is the source of truth for
# the exact columns and column order expected.
MODEL_FEATURES = model.feature_names_in_.tolist()

print("Model expects the following features:")
print(MODEL_FEATURES)


# ============================================
# Feature Engineering
# ============================================

def transform_input(data):
    """
    Convert frontend-friendly/raw categorical fields into the
    engineered features used when the model was trained.

    The transformations here MUST match the feature engineering
    performed in the training notebook.
    """

    df = data.copy()

    # ---------------------------------------------------------
    # 1. Product Sugar Content
    # ---------------------------------------------------------

    if "Product_Sugar_Content" in df.columns:

        # Standardize inconsistent category used in training data
        df["Product_Sugar_Content"] = (
            df["Product_Sugar_Content"]
            .replace({
                "reg": "Regular"
            })
        )

        sugar_map = {
            "No Sugar": 0,
            "Low Sugar": 1,
            "Regular": 2
        }

        df["Product_Sugar_Content_Num"] = (
            df["Product_Sugar_Content"]
            .map(sugar_map)
        )


    # ---------------------------------------------------------
    # 2. Store Size
    # ---------------------------------------------------------

    if "Store_Size" in df.columns:

        store_size_map = {
            "Small": 0,
            "Medium": 1,
            "High": 2
        }

        df["Store_Size_Num"] = (
            df["Store_Size"]
            .map(store_size_map)
        )


    # ---------------------------------------------------------
    # 3. Store Location City Type
    # ---------------------------------------------------------

    if "Store_Location_City_Type" in df.columns:

        city_tier_map = {
            "Tier 3": 0,
            "Tier 2": 1,
            "Tier 1": 2
        }

        df["Store_Location_City_Type_Num"] = (
            df["Store_Location_City_Type"]
            .map(city_tier_map)
        )


    # ---------------------------------------------------------
    # 4. Product ID Prefix
    # ---------------------------------------------------------

    # Frontend/instructor payload uses Product_Id_char
    # while our trained model expects Product_Id_Char.
    if "Product_Id_char" in df.columns:

        df = df.rename(
            columns={
                "Product_Id_char": "Product_Id_Char"
            }
        )


    # ---------------------------------------------------------
    # 5. Product Type Category
    # ---------------------------------------------------------

    if "Product_Type_Category" in df.columns:

        product_type_category_map = {
            "Non Perishables": 0,
            "Perishables": 1
        }

        df["Product_Type_Category_Num"] = (
            df["Product_Type_Category"]
            .map(product_type_category_map)
        )


    # ---------------------------------------------------------
    # 6. Remove original fields replaced by engineered fields
    # ---------------------------------------------------------

    df = df.drop(
        columns=[
            "Product_Sugar_Content",
            "Store_Size",
            "Store_Location_City_Type",
            "Product_Type_Category"
        ],
        errors="ignore"
    )


    # ---------------------------------------------------------
    # 7. Check for failed categorical mappings
    # ---------------------------------------------------------

    engineered_numeric_columns = [
        "Product_Sugar_Content_Num",
        "Store_Size_Num",
        "Store_Location_City_Type_Num",
        "Product_Type_Category_Num"
    ]

    for column in engineered_numeric_columns:

        if column in df.columns and df[column].isna().any():

            raise ValueError(
                f"Unable to map one or more values for '{column}'. "
                "Check that the incoming categorical values match "
                "the categories used during model training."
            )


    return df


# ============================================
# Health Check
# ============================================

@superkart_api.get("/")
def home():
    """
    Simple health-check endpoint.
    """

    return jsonify({
        "message": "SuperKart Prediction API",
        "status": "ok",
        "model_features": MODEL_FEATURES
    })


# ============================================
# Online Prediction
# ============================================

@superkart_api.post("/v1/predict")
def predict():
    """
    Online inference endpoint.

    Accepts a single JSON record using human-readable
    categorical values.

    The backend converts those fields into the engineered
    features expected by the trained model.
    """

    try:

        # --------------------------------------
        # Read JSON request
        # --------------------------------------

        payload = request.get_json()

        if not payload:

            return jsonify({
                "error": "JSON payload is required."
            }), 400


        # Convert JSON object into one-row DataFrame
        input_data = pd.DataFrame(
            [payload]
        )


        # --------------------------------------
        # Feature engineering
        # --------------------------------------

        input_data = transform_input(
            input_data
        )


        # --------------------------------------
        # Validate model features
        # --------------------------------------

        missing_columns = [
            column
            for column in MODEL_FEATURES
            if column not in input_data.columns
        ]

        if missing_columns:

            return jsonify({
                "error": f"Missing columns: {missing_columns}",
                "provided_columns": input_data.columns.tolist(),
                "expected_columns": MODEL_FEATURES
            }), 400


        # --------------------------------------
        # Preserve exact training column order
        # --------------------------------------

        input_data = input_data[
            MODEL_FEATURES
        ]


        # --------------------------------------
        # Prediction
        # --------------------------------------

        prediction = float(
            model.predict(input_data)[0]
        )


        return jsonify({
            "prediction": prediction
        })


    except Exception as exception:

        return jsonify({
            "error": str(exception)
        }), 400


# ============================================
# Batch Prediction
# ============================================

@superkart_api.post("/v1/predictbatch")
def predict_batch():
    """
    Batch inference endpoint.

    Accepts a CSV file using the same human-readable schema
    as the frontend.

    Expected multipart/form-data key:
        file
    """

    try:

        # --------------------------------------
        # Validate uploaded file
        # --------------------------------------

        if "file" not in request.files:

            return jsonify({
                "error": "CSV file is required under key 'file'."
            }), 400


        # --------------------------------------
        # Read CSV
        # --------------------------------------

        batch_data = pd.read_csv(
            request.files["file"]
        )


        # --------------------------------------
        # Apply the same feature engineering
        # --------------------------------------

        batch_data = transform_input(
            batch_data
        )


        # --------------------------------------
        # Validate model features
        # --------------------------------------

        missing_columns = [
            column
            for column in MODEL_FEATURES
            if column not in batch_data.columns
        ]

        if missing_columns:

            return jsonify({
                "error": f"Missing columns: {missing_columns}",
                "provided_columns": batch_data.columns.tolist(),
                "expected_columns": MODEL_FEATURES
            }), 400


        # --------------------------------------
        # Preserve exact model feature order
        # --------------------------------------

        batch_data = batch_data[
            MODEL_FEATURES
        ]


        # --------------------------------------
        # Generate predictions
        # --------------------------------------

        predictions = model.predict(
            batch_data
        )


        # Instructor-aligned response format:
        # row index -> predicted sales
        response = {
            str(index): float(prediction)
            for index, prediction
            in enumerate(predictions)
        }


        return jsonify(response)


    except Exception as exception:

        return jsonify({
            "error": str(exception)
        }), 400


# ============================================
# Application Entry Point
# ============================================

if __name__ == "__main__":

    superkart_api.run(
        host="0.0.0.0",
        port=7860
    )
