import os
import crested
import pandas as pd
import keras

from error_checking_functions import PredictionFailedError

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_PATH, "model")
MODEL_NAME = "deepbiccn2"
saved_models_path = os.path.join(MODEL_PATH, f"{MODEL_NAME}.keras")
targets_file = os.path.join(MODEL_PATH, f"{MODEL_NAME}_output_classes.tsv")


def get_cell_type_index():
    """
    Returns a dictionary mapping cell type names to their corresponding indices.
    """
    targets_df = pd.read_csv(targets_file, sep="\t", names=["target"])
    return {target: i for i, target in enumerate(targets_df["target"])}

#padd sequences
def pad_sequences(sequences):
    required_length = 2114
    seqs_updated = {}
    for key, seq in sequences.items():

        seq_len = len(seq)
        if seq_len < required_length:
            total_padding = required_length - seq_len
            right_padding = 'N' * (total_padding // 2)
            left_padding = 'N' * (total_padding - len(right_padding))
            padded_seq = left_padding + seq + right_padding
            #print(len(padded_seq))
            seqs_updated[key] = padded_seq

        # The condition for when sequence length is greater than the target length
        elif seq_len > required_length:
            center_pos = seq_len//2
            region_half = required_length//2
            cropped_seq = seq[center_pos-region_half:center_pos+region_half]
            seqs_updated[key] = cropped_seq
        else:
            # Sequence is already the correct length
            seqs_updated[key] = seq

    return seqs_updated

def predict_crested(sequences: dict) -> dict:
    """
    Runs DeepBICCN2 predictions on sequences.

    Args:
        sequences: Dictionary of sequence_id -> DNA sequence (already padded/cropped to 2114bp)

    Returns:
        Dictionary of sequence_id -> numpy array of predictions (one per cell type)

    Raises:
        PredictionFailedError: If model loading or prediction fails
    """
    try:
        # extract sequences from dict
        seqs = list(sequences.values())
        seqs_ids = list(sequences.keys())

        # Load the DeepBICCN2 model
        model = keras.models.load_model(saved_models_path, compile=False)

        # Run predictions - returns (N, C) array where N=num_sequences, C=num_cell_types
        crested_predictions = crested.tl.predict(
            input=seqs,
            model=model,
            genome=None,
        )

        # Package predictions by sequence ID
        predictions = {}
        for i, seq_id in enumerate(seqs_ids):
            predictions[seq_id] = crested_predictions[i]

        print(f"Successfully generated predictions for {len(predictions)} sequences")
        return predictions

    except Exception as e:
        raise PredictionFailedError(f"DeepBICCN2 model prediction failed: {e}")

# def predict_crested(sequences):
#     #print(sequences)
#     #print(pad_sequences(sequences))
#     predictions = {}
#     try:
#         # extract sequences from dict
#         #seqs = list(sequences.values())

#         for seq_id, seq in sequences.items():
            
#             predictions[seq_id] = 0
#     except Exception as e:
#         predictions = str(e)
#     return predictions