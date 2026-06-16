"""RESTful DeepBICCN2 Predictor Utilizing Flask"""

import os
import sys
import json
import numpy as np
from flask import Flask

from config import (
    PREDICTOR_NAME,
    HELP_FILE,
    SUPPORTED_REQUEST_FORMATS,
    SUPPORTED_RESPONSE_FORMATS,
)
from error_checking_functions import *
from schema_validation import *
from crested_utils import predict_crested, resolve_cell_type_index
from predictor_content_handler import decode_request, encode_response

# Get the absolute path of the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# PREDICTOR_NAME, HELP_FILE, and the supported wire formats are defined in config.py.
# PREDICTOR_NAME is versioned with the container build date (see config.py).

# --- Flask App and Central Error Handler ---
app = Flask(__name__)
# Maintain order when using jsonify()
app.config["JSON_SORT_KEYS"] = False
app.json.sort_keys = False


def create_error_response(error_key, messages, status_code):
    """
    Formats error response into a standardized JSON structure.

    Args:
        error_key (str): The category of the error (e.g. 'bad_prediction_request', 'prediction_request_failed').
        messages (list or str): A list of error message strings or a single message.
        status_code (int): Standard HTTP error status code based on the error.

    Returns:
        dict: A dictionary formatted for the standardized JSON error response.
    """
    if not isinstance(messages, list):
        messages = [str(messages)]
    error_payload = {"error": [{error_key: msg} for msg in messages]}
    print(error_payload)
    return error_payload, status_code


@app.errorhandler(APIError)
def handle_api_error(error):
    """This single handler catches all of our custom API errors."""
    # Get raw payload and status code
    payload, status_code = create_error_response(
        error.error_key, error.message, error.status_code
    )

    return encode_response(
        payload, status_code=status_code, isError=True, predictor_name=PREDICTOR_NAME
    )


@app.after_request
def after_request_callback(response):
    """This function runs after each request is processed."""
    print(f"\n--- Sending predictions back to Evaluator. ---")
    print(
        f"--- Request Complete. {PREDICTOR_NAME} Predictor is listening on http://{predictor_ip}:{predictor_port} ---\n"
    )
    return response


# --- API Endpoints ---
@app.route("/formats", methods=["GET"])
def formats_endpoint():
    """Provides the Predictor's supported formats"""
    supported_fmts = {
        "predictor_supported_request_formats": SUPPORTED_REQUEST_FORMATS,
        "predictor_supported_response_formats": SUPPORTED_RESPONSE_FORMATS,
    }
    try:
        return encode_response(
            supported_fmts,
            status_code=200,
            predictor_name=PREDICTOR_NAME,
            supported_response_formats=SUPPORTED_RESPONSE_FORMATS,
        )
    except Exception as e:
        raise ServerError(
            f"Error serializing supported format for /format endpoint: {e}"
        )


@app.route("/help", methods=["GET"])
def help_endpoint():
    """Provides the Predictor's help/metadata information."""
    try:
        with open(HELP_FILE, "r") as f:
            help_data = json.load(f)
        return encode_response(
            help_data,
            status_code=200,
            predictor_name=PREDICTOR_NAME,
            supported_response_formats=SUPPORTED_RESPONSE_FORMATS,
        )
    except Exception as e:
        raise ServerError(f"Error reading help file: {e}")


@app.route("/predict", methods=["POST"])
def predict():
    """The main endpoint for receiving sequences and returning predictions."""

    try:
        # Decode incoming request, using the headers or JSON default
        evaluator_request = decode_request(SUPPORTED_REQUEST_FORMATS)

        # Validate the payload using the imported function
        # These functions will raise an APIError on failure,
        # which will be caught automatically by @app.errorhandler
        validate_request_payload(evaluator_request)

        # Preprocess the data (flanking, ranges, padding, cell type validation)
        sequences = preprocess_data(evaluator_request)

        # --- Run DeepBICCN2 Model Inference ---
        # This returns predictions for ALL cell types for each sequence
        # Format: {seq_id: numpy_array[num_cell_types]}
        all_predictions = predict_crested(sequences)

        # Assemble the response
        json_return = {"prediction_tasks": []}

        for task in evaluator_request["prediction_tasks"]:
            task_name = task["name"]
            requested_cell_type = task["cell_type"]
            requested_type = task["type"]
            requested_species = task["species"]
            requested_scale = task.get("scale", "linear")

            # Get the index for the requested cell type. Accepts canonical names
            # (advertised in /help) or the model's short labels.
            cell_type_idx = resolve_cell_type_index(requested_cell_type)

            if cell_type_idx is None:
                raise PredictionFailedError(
                    f"Cell type '{requested_cell_type}' not found in DeepBICCN2 model outputs."
                )

            # Extract predictions for this specific cell type from all sequences
            # DeepBICCN2 model outputs predictions in log scale (log1p transformed)
            task_predictions = {}
            for seq_id, pred_array in all_predictions.items():
                # pred_array is a numpy array with shape (num_cell_types,)
                # Extract the value for this specific cell type
                raw_value = pred_array[cell_type_idx]

                # Apply scale transformation based on requested scale
                # Model outputs log scale, so we need to convert if linear is requested
                if requested_scale == "linear":
                    # Convert from log scale to linear: expm1(x) = exp(x) - 1
                    transformed_value = float(np.expm1(raw_value))
                    actual_scale = "linear"
                else:
                    # Keep in log scale (default model output)
                    transformed_value = float(raw_value)
                    actual_scale = "log"

                # Point-based models return a scalar value per sequence (GAME API
                # spec). transformed_value is already a Python float.
                task_predictions[seq_id] = transformed_value

            # Build the task response
            json_return["prediction_tasks"].append(
                {
                    "name": task_name,
                    "type_requested": requested_type,
                    # type_actual is the list of assay(s) the model actually predicts
                    # (GAME API spec). DeepBICCN2 predicts ATAC accessibility
                    # (Tn5 cut-site counts).
                    "type_actual": ["ATAC"],
                    "cell_type_requested": requested_cell_type,
                    "cell_type_actual": requested_cell_type,
                    "scale_prediction_requested": requested_scale,
                    "scale_prediction_actual": actual_scale,
                    "species_requested": requested_species,
                    "species_actual": "mus_musculus",
                    "predictions": task_predictions,
                }
            )

        return encode_response(
            json_return,
            status_code=200,
            predictor_name=PREDICTOR_NAME,
            supported_response_formats=SUPPORTED_RESPONSE_FORMATS,
        )

    except Exception as e:
        # If it's already an APIError, re-raise it for the handler
        if isinstance(e, APIError):
            raise e
        # Otherwise, wrap the unknown error in a ServerError
        raise ServerError(f"An unexpected internal error occurred: {e}.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"Invalid arguments! Arguments must have: <container image/python script> <ip_address> <port>"
        )
        sys.exit(1)

    predictor_ip = sys.argv[1]
    predictor_port = int(sys.argv[2])
    app.run(host=predictor_ip, port=predictor_port)
