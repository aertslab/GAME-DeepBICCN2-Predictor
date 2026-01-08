# DeepBICCN2 Cell Type-Specific Chromatin Accessibility Predictor Container for the Genomic API for Model Evaluation (GAME)

Contains a predictor container for the Genomic API for Model Evaluation (GAME). The system provides computational
predictions of cell type-specific chromatin accessibility in the mouse motor cortex from DNA sequence alone.

## Model Overview

DeepBICCN2 is a deep learning model trained on single-cell ATAC-seq data from the BRAIN Initiative Cell Census Network (BICCN). The
model predicts chromatin accessibility patterns across 19 distinct mouse motor cortex cell types directly from genomic sequences.

## Supported Cell Types

The model provides predictions for 19 mouse motor cortex cell types:

- Excitatory neurons: L2/3 IT, L5 ET, L5 IT, L5/6 NP, L6 CT, L6 IT, L6b
- Inhibitory neurons: Lamp5, Pvalb, Sncg, Sst, Sst Chodl, Vip
- Glial cells: Astrocytes (Astro), Microglia/PVM, Oligodendrocyte Precursor Cells (OPC), Oligodendrocytes (Oligo)
- Vascular cells: Endothelial cells (Endo), Vascular Leptomeningeal Cells (VLMC)

## Model Specifications

- Input: DNA sequences of 2114 base pairs (sequences are automatically padded or cropped to this length)
- Species: Mouse (Mus musculus)
- Output: Tn5 cut-site counts representing chromatin accessibility
- Output Scale: Linear (log scale available on request)
- Readout Type: Point predictions at sequence center
- Architecture: Convolutional neural network trained on BICCN scATAC-seq data

## API Features

The predictor implements the GAME REST API specification and supports:
- /help endpoint: Model metadata and documentation
- /formats endpoint: Supported request/response formats (JSON, MessagePack)
- /predict endpoint: Sequence-to-accessibility predictions
- Batch predictions for multiple sequences and cell types
- Automatic sequence padding and cropping
- Flexible output scaling (linear or log)

## Model Files

- deepbiccn2.keras: Pre-trained model in Keras format
- deepbiccn2_output_classes.tsv: Cell type index mapping

## Documentation

Full documentation available at: https://crested.readthedocs.io/en/stable/models/BICCN/deepbiccn2.html

## Citations

This predictor is based on the DeepBICCN2 model described in:

Kempynck, N., De Winter, S., et al. CREsted: modeling genomic and synthetic cell type-specific enhancers across tissues and species. Zenodo.      https://doi.org/10.5281/zenodo.13918932
