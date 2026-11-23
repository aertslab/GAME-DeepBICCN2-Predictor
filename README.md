# DeepBICCN2 Cell Type-Specific Chromatin Accessibility Predictor Container for the Genomic API for Model Evaluation (GAME)

Contains a predictor container for the Genomic API for Model Evaluation (GAME). The system provides computational
predictions of cell type-specific chromatin accessibility in the mouse motor cortex from DNA sequence alone.

## Model Overview

DeepBICCN2 is a deep learning model trained on single-cell ATAC-seq data from the BRAIN Initiative Cell Census Network (BICCN). The
model predicts chromatin accessibility patterns across 19 distinct mouse motor cortex cell types directly from genomic sequences.

## Supported Cell Types

The model provides predictions for 19 mouse motor cortex cell types. Request them by these names (as listed in `/help`):

- Excitatory neurons: `L2/3 IT`, `L5 ET`, `L5 IT`, `L5/6 NP`, `L6 CT`, `L6 IT`, `L6b`
- Inhibitory neurons: `Lamp5`, `Pvalb`, `Sncg`, `Sst`, `Sst Chodl`, `Vip`
- Glial cells: `Astrocyte`, `Micro-PVM`, `OPC` (oligodendrocyte precursor cells), `Oligodendrocyte`
- Vascular cells: `Endothelial`, `VLMC` (vascular leptomeningeal cells)

The model's own short labels (e.g. `Astro`, `Endo`, `Oligo`, `L2_3IT`, from `model/deepbiccn2_output_classes.tsv`) are also accepted. Either way, `cell_type_actual` reports the name from the list above. Names are case-sensitive. A task with any other cell type is returned as a failed task, and the other tasks are still predicted.

## Model Specifications

- Input: DNA sequences of 2114 base pairs. After any flanks are added and any `prediction_ranges` applied, shorter sequences are N-padded and longer ones are cropped, both around the sequence centre, to 2114 bp.
- Species: Mouse (`mus_musculus`). Tasks for any other species are returned as failed tasks.
- Type: `accessibility`, reported as `type_actual: ["ATAC"]`. Tasks for any other type are returned as failed tasks.
- Output: Tn5 cut-site counts representing chromatin accessibility
- Output Scale: `linear` (default) or `log`. The model output is treated as `log1p` of the counts: `log` returns the model output as is, and `linear` returns `expm1` of it.
- Readout Type: `point` only, one value per sequence for the central 2114 bp window. Other readouts are rejected with HTTP 422.
- Architecture: Convolutional neural network trained on BICCN scATAC-seq data

## API Features

The predictor implements the GAME REST API specification and supports:
- /help endpoint: Model metadata and documentation
- /formats endpoint: Supported request/response formats (JSON, MessagePack)
- /predict endpoint: Sequence-to-accessibility predictions
- Batch predictions for multiple sequences and cell types
- Automatic sequence padding and cropping
- Flexible output scaling (linear or log)

**Failed tasks.** A task the model cannot perform (unsupported species or type, or an unknown cell type) does not fail the request. It is returned with `type_actual`, `cell_type_actual`, `scale_prediction_actual` and `species_actual` set to `null` and `"predictions": {"error": [{"prediction_request_failed": "<reason>"}]}`; the response is still HTTP 200 and the other tasks are predicted. If every task fails, the model is not run.

**Prediction ranges** are 0-based and half-open, `[start, end)`. Each range selects the part of the (flanked) input sequence to predict on, which is then padded or cropped to 2114 bp. An empty list `[]` means the whole sequence. Ranges must be non-negative with `start < end` and `end <= len(sequence)`, and their sequence IDs must match `sequences`.

**Errors** are always returned as JSON, as `{"error": [{"<error_key>": "<message>"}]}`:

| HTTP | `error_key` | When |
|---|---|---|
| 400 | `bad_prediction_request` | Malformed request: missing keys, wrong value types, unsupported `scale`, badly formatted `prediction_ranges` |
| 422 | `prediction_request_failed` | Unsupported readout (anything but `point`), out-of-bounds `prediction_ranges`, empty sequences, invalid bases (outside `A/T/C/G/N`), or the model failing to predict |
| 500 | `server_error` | Unexpected backend failure |

## Model Files

- deepbiccn2.keras: Pre-trained model in Keras format
- deepbiccn2_output_classes.tsv: Cell type index mapping

## Build and run

Building is optional: you only need to build (or rebuild) the container if you don't already have `deepbiccn2_restapi_predictor.sif`, or if you have changed the code, `model/` or `environment.yml`. Build from the repository root (the definition copies the code, `model/` and `environment.yml` from there; the build downloads Miniconda and the CREsted environment, so it needs internet access):

```bash
apptainer build deepbiccn2_restapi_predictor.sif predictor.def
```

Run the Predictor:

```bash
apptainer run --nv deepbiccn2_restapi_predictor.sif <HOST> <PORT>
# e.g.
apptainer run --nv deepbiccn2_restapi_predictor.sif 0.0.0.0 5000
```

`--nv` exposes the GPU to the container; without it the model runs on CPU. The Predictor names itself `DeepBICCN2_<build timestamp>` inside the container (from the Apptainer build date) and `DeepBICCN2_dev` outside it.

## Documentation

Full documentation available at: https://crested.readthedocs.io/en/stable/models/BICCN/deepbiccn2.html

## Citations

This predictor is based on the DeepBICCN2 model described in:

Kempynck, N., De Winter, S., et al. CREsted: modeling genomic and synthetic cell type-specific enhancers across tissues and species. Zenodo.      https://doi.org/10.5281/zenodo.13918932
