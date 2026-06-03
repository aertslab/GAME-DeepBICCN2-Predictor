'''
Configuration script for the DeepBICCN2 Predictor.
- Determines whether we are running inside an Apptainer/Singularity container.
- Automatically versions the Predictor name using Apptainer's build-date label.
    - Inside container:             "DeepBICCN2_20251128-180629_PST"  (sortable, human-readable)
    - Outside container (Dev mode): "DeepBICCN2_dev"
- Holds the help-file path and the supported wire formats.
'''

import os
import json
from datetime import datetime

# Get the absolute path of this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Base model name (build date is appended automatically below)
MODEL_NAME = "DeepBICCN2"

if os.path.exists("/.singularity.d"):
    # Running inside the container
    print("Running inside the container...🥡")
    HELP_FILE = os.path.join(SCRIPT_DIR, "predictor_help_message.json")

    # Read the build timestamp from Apptainer's auto-generated labels file
    # to version the Predictor name (Apptainer writes this label at build time).
    try:
        with open("/.singularity.d/labels.json", "r") as f:
            labels = json.load(f)
        raw_build_date = labels.get("org.label-schema.build-date", "")

        # Example format: "Friday_28_November_2025_18:6:29_PST"
        # Strip day-of-week and timezone, keep the core date+time
        parts = raw_build_date.split("_")
        # parts: ['Friday', '28', 'November', '2025', '18:6:29', 'PST']
        date_str = f"{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}"

        dt = datetime.strptime(date_str, "%d_%B_%Y_%H:%M:%S")
        build_timestamp = dt.strftime("%Y%m%d-%H%M%S")
        timezone_label = parts[5] if len(parts) > 5 else "UNK"
        PREDICTOR_NAME = f"{MODEL_NAME}_{build_timestamp}_{timezone_label}"

    except Exception as e:
        print(f"Warning: Could not parse build timestamp from labels.json: {e}")
        PREDICTOR_NAME = f"{MODEL_NAME}_unknown"
else:
    # Running outside the container (development mode)
    print("Running outside the container...📋")
    HELP_FILE = os.path.join(SCRIPT_DIR, "predictor_help_message.json")
    PREDICTOR_NAME = f"{MODEL_NAME}_dev"


# ------ Configuration for Wire-Format ------
SUPPORTED_REQUEST_FORMATS = [
    fmt.lower() for fmt in ["application/json", "application/msgpack"]
]
SUPPORTED_RESPONSE_FORMATS = [
    fmt.lower() for fmt in ["application/json", "application/msgpack"]
]
