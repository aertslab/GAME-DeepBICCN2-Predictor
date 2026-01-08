'''Validation and Preprocessing of Payload for DeepBICCN2 Predictor'''
import tqdm
from error_checking_functions import *
from crested_utils import pad_sequences, get_cell_type_index

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

    # --- DeepBICCN2-Specific Validation: Check readout type ---
    readout_type = payload.get('readout')
    if readout_type not in SUPPORTED_READOUTS:
        errors['bad_prediction_request'].append(
            f"DeepBICCN2 only supports readout types {SUPPORTED_READOUTS}. "
            f"Received '{readout_type}'."
        )

    # --- DeepBICCN2-Specific Validation: Check species, type, and scale in tasks ---
    for task in payload['prediction_tasks']:
        task_name = task.get('name', 'unknown')

        # Check species
        species = task.get('species', '').lower()
        if species not in SUPPORTED_SPECIES:
            errors['bad_prediction_request'].append(
                f"DeepBICCN2 only supports species {SUPPORTED_SPECIES}. "
                f"Task '{task_name}' requested '{species}'."
            )

        # Check type
        pred_type = task.get('type', '').lower()
        if pred_type not in SUPPORTED_TYPES:
            errors['bad_prediction_request'].append(
                f"DeepBICCN2 only supports prediction types {SUPPORTED_TYPES}. "
                f"Task '{task_name}' requested '{pred_type}'."
            )

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
    - Applies prediction ranges if provided
    - Pads/crops sequences to required length (2114bp)
    - Validates cell types against model outputs
    - Checks sequence specifications

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

    # Apply prediction_ranges if provided
    if 'prediction_ranges' in payload:
        for seq_id, pr in payload['prediction_ranges'].items():
            if pr:  # Only process non-empty ranges
                start, end = pr
                # Slice the sequence. `prediction_range` is start, end inclusive
                sequences[seq_id] = sequences[seq_id][start:end+1]
                print(f"Sequence '{seq_id}' trimmed to prediction range [{start}, {end}].")

    # --- DeepBICCN2-Specific: Pad or crop sequences to required length (2114bp) ---
    print("Padding/cropping sequences to 2114bp for DeepBICCN2 model...")
    sequences = pad_sequences(sequences)

    # Check that the final sequences meet model specifications
    errors = {'prediction_request_failed': []}
    errors = check_seqs_specifications(sequences, errors)

    # --- DeepBICCN2-Specific: Validate cell types against model outputs ---
    try:
        cell_type_mapping = get_cell_type_index()
        valid_cell_types = set(cell_type_mapping.keys())
    except Exception as e:
        raise PredictionFailedError(f"Failed to load cell type mapping: {e}")

    for task in payload['prediction_tasks']:
        task_name = task.get('name', 'unknown')
        cell_type = task.get('cell_type')

        if cell_type not in valid_cell_types:
            errors['prediction_request_failed'].append(
                f"Cell type '{cell_type}' in task '{task_name}' is not recognized by DeepBICCN2. "
                f"Valid cell types: {sorted(list(valid_cell_types))}"
            )

    if any(errors.values()):
        flagged_errors = [msg for sublist in errors.values() for msg in sublist]
        raise PredictionFailedError(flagged_errors)

    return sequences
