'''Validation and Preprocessing of Payload for DeepBICCN2 Predictor'''
import tqdm
from error_checking_functions import *
from crested_utils import pad_sequences, valid_cell_type_names, CANONICAL_TO_MODEL

# DeepBICCN2 Model Constraints
SUPPORTED_SPECIES = ["mus_musculus"]
SUPPORTED_TYPES = ["accessibility"]
SUPPORTED_READOUTS = ["point"]
SUPPORTED_SCALES = ["linear", "log"]

def validate_request_payload(payload):
    """
    Performs all validation checks on the incoming request payload.
    Includes DeepBICCN2-specific constraints.

    Raises:
        BadRequestError: If the request format or content is invalid
    """
    errors = {'bad_prediction_request': []}

    # First confirm all mandatory keys are present
    errors = check_mandatory_keys(payload.keys(), errors)
    if any(errors.values()):
        flagged_errors = [msg for sublist in errors.values() for msg in sublist]
        raise BadRequestError(flagged_errors)

    # Check for mandatory keys inside each task object
    errors = check_prediction_task_mandatory_keys(payload['prediction_tasks'], errors)
    if any(errors.values()):
        flagged_errors = [msg for sublist in errors.values() for msg in sublist]
        raise BadRequestError(flagged_errors)

    # Perform all other validation checks
    errors = check_key_values_readout(payload['readout'], errors)
    errors = check_prediction_task_name(payload['prediction_tasks'], errors)
    errors = check_prediction_task_type(payload['prediction_tasks'], errors)
    errors = check_prediction_task_cell_type(payload['prediction_tasks'], errors)
    errors = check_prediction_task_species(payload['prediction_tasks'], errors)
    errors = check_prediction_task_scale(payload['prediction_tasks'], errors)

    if 'prediction_ranges' in payload:
        errors = check_seq_ids(payload['prediction_ranges'], payload['sequences'], errors)
        errors = check_prediction_ranges(payload['prediction_ranges'], payload['sequences'], errors)

    if 'upstream_seq' in payload:
        errors = check_key_values_upstream_flank(payload['upstream_seq'], errors)
    if 'downstream_seq' in payload:
        errors = check_key_values_downstream_flank(payload['downstream_seq'], errors)

    # --- DeepBICCN2-Specific Validation: Check scale in tasks ---
    # Unsupported species, types and cell types are not request errors: those tasks are
    # returned individually as failed tasks (see task_failure_reason), per the GAME spec.
    for task in payload['prediction_tasks']:
        task_name = task.get('name', 'unknown')

        # Check scale (if provided)
        scale = task.get('scale', 'linear').lower()
        if scale not in SUPPORTED_SCALES:
            errors['bad_prediction_request'].append(
                f"DeepBICCN2 only supports scales {SUPPORTED_SCALES}. "
                f"Task '{task_name}' requested '{scale}'."
            )

    if any(errors.values()):
        flagged_errors = [msg for sublist in errors.values() for msg in sublist]
        raise BadRequestError(flagged_errors)

def preprocess_data(payload):
    """
    Handles data preprocessing for DeepBICCN2:
    - Applies flanking sequences if provided
    - Checks prediction ranges are within the flanked sequences, then applies them
    - Pads/crops sequences to required length (2114bp)
    - Checks sequence specifications and that the readout is supported
    (Cell types, species and types are checked per task in task_failure_reason.)

    Returns:
        Processed sequences ready for model input

    Raises:
        PredictionFailedError: If preprocessing fails or sequences don't meet specs
    """
    sequences = payload.get('sequences', {})

    # Apply upstream and downstream flanking sequences if provided
    if 'upstream_seq' in payload or 'downstream_seq' in payload:
        upstream_seq = payload.get('upstream_seq', "")
        downstream_seq = payload.get('downstream_seq', "")
        if upstream_seq or downstream_seq:
            print(
                f"Applying flanking:\
                    \n+{len(upstream_seq)} bases upstream,\
                    \n+{len(downstream_seq)} bases downstream"
            )
            for seq_id, sequence in tqdm.tqdm(
                sequences.items(),
                desc="Flanking sequences",
                unit="sequence",
                total=len(sequences),
                dynamic_ncols=True
            ):
                flanked = f"{upstream_seq}{sequence}{downstream_seq}"
                sequences[seq_id] = flanked

    # Check that prediction_ranges are within the (flanked) sequences.
    # Since this is model-specific, it utilizes `PredictionFailedError`.
    errors = {'prediction_request_failed': []}
    if 'prediction_ranges' in payload:
        for seq_id, pr in payload['prediction_ranges'].items():
            if pr: # Only process non-empty ranges
                start, end = pr
                seq_len = len(sequences[seq_id])

                if start >= seq_len or end > seq_len:
                    # Handle the empty sequence case for a clean message
                    if seq_len == 0:
                        err_msg = f"Invalid range for '{seq_id}': cannot specify a range for a non-existent or empty sequence."
                    else:
                        err_msg = f"Invalid range for '{seq_id}': index is out of bounds. For a sequence of length {seq_len}, start must be < {seq_len} and end must be <= {seq_len}."
                    errors['prediction_request_failed'].append(err_msg)

    if any(errors.values()):
        flagged_errors = [msg for sublist in errors.values() for msg in sublist]
        raise PredictionFailedError(flagged_errors)

    # Apply prediction_ranges if provided
    if 'prediction_ranges' in payload:
        for seq_id, pr in payload['prediction_ranges'].items():
            if pr:  # Only process non-empty ranges
                start, end = pr
                # Slice the sequence. `prediction_range` is half-open per the GAME spec:
                # start inclusive, end exclusive (Python slicing)
                sequences[seq_id] = sequences[seq_id][start:end]
                print(f"Sequence '{seq_id}' trimmed to prediction range [{start}, {end}).")

    # --- DeepBICCN2-Specific: Pad or crop sequences to required length (2114bp) ---
    print("Padding/cropping sequences to 2114bp for DeepBICCN2 model...")
    sequences = pad_sequences(sequences)

    # Check that the final sequences meet model specifications
    errors = {'prediction_request_failed': []}
    errors = check_seqs_specifications(sequences, errors)

    # Model-specific error checking: Readout type check
    readout_type = payload['readout']
    if not readout_type in SUPPORTED_READOUTS:
        raise PredictionFailedError(f"DeepBICCN2 cannot process '{readout_type}' readout type.")

    if any(errors.values()):
        flagged_errors = [msg for sublist in errors.values() for msg in sublist]
        raise PredictionFailedError(flagged_errors)

    return sequences

def task_failure_reason(task):
    """
    Checks whether DeepBICCN2 can perform a single prediction task.

    A task the model cannot perform (unsupported species or type, or a cell type the
    model does not have) is returned on its own as a failed task, while the other tasks
    are still predicted (GAME spec: "Tasks the Predictor cannot perform").

    Returns:
        str: why the task cannot be performed, or None if it can be performed
    """
    species = task['species'].lower()
    if species not in SUPPORTED_SPECIES:
        return (f"DeepBICCN2 only supports species {SUPPORTED_SPECIES}. "
                f"Task '{task['name']}' requested '{task['species']}'.")

    pred_type = task['type'].lower()
    if pred_type not in SUPPORTED_TYPES:
        return (f"DeepBICCN2 only supports prediction types {SUPPORTED_TYPES}. "
                f"Task '{task['name']}' requested '{task['type']}'.")

    if task['cell_type'] not in valid_cell_type_names():
        return (f"Cell type '{task['cell_type']}' in task '{task['name']}' is not recognized by DeepBICCN2. "
                f"Valid cell types: {sorted(CANONICAL_TO_MODEL.keys())}")

    return None
