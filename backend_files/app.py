# ============================================
# SuperKart Flask Prediction API
# Instructor-aligned API contract
# ============================================

import io
import joblib
import pandas as pd
from flask import Flask, request, jsonify

MODEL_FILE = "superkart_model.joblib"

# Exact feature order expected by the serialized model.
MODEL_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

# Name the Flask application consistently with the instructor notebook.
superkart_api = Flask(__name__)

# Load the full serialized preprocessing + model pipeline.
model = joblib.load(MODEL_FILE)


@superkart_api.get("/")
def home():
    """Simple health-check route."""
    return jsonify({
        "message": "SuperKart Prediction API",
        "status": "ok",
    })


@superkart_api.post("/v1/predict")
def predict():
    """
    Online inference endpoint.
    Accepts a single JSON object using the 10 engineered model features.
    """
    try:
        payload = request.get_json()

        if not payload:
            return jsonify({
                "error": "JSON payload is required."
            }), 400

        input_data = pd.DataFrame(
            [payload]
        )

        # Validate required fields.
        missing_columns = [
            column
            for column in MODEL_FEATURES
            if column not in input_data.columns
        ]

        if missing_columns:
            return jsonify({
                "error": f"Missing columns: {missing_columns}"
            }), 400

        input_data = input_data[
            MODEL_FEATURES
        ]

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


@superkart_api.post("/v1/predictbatch")
def predict_batch():
    """
    Batch inference endpoint.
    Accepts a CSV file under multipart/form-data key 'file'.

    Returns JSON where each key is a row index and each value is
    the corresponding predicted sales value, matching the instructor
    notebook's described response format.
    """
    try:
        if "file" not in request.files:
            return jsonify({
                "error": "CSV file is required under key 'file'."
            }), 400

        batch_data = pd.read_csv(
            request.files["file"]
        )

        missing_columns = [
            column
            for column in MODEL_FEATURES
            if column not in batch_data.columns
        ]

        if missing_columns:
            return jsonify({
                "error": f"Missing columns: {missing_columns}"
            }), 400

        batch_data = batch_data[
            MODEL_FEATURES
        ]

        predictions = model.predict(
            batch_data
        )

        response = {
            str(index): float(prediction)
            for index, prediction in enumerate(predictions)
        }

        return jsonify(response)

    except Exception as exception:
        return jsonify({
            "error": str(exception)
        }), 400


if __name__ == "__main__":
    superkart_api.run(
        host="0.0.0.0",
        port=7860,
    )
