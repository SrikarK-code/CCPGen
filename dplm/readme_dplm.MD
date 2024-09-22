guidance classifier dplm
# Guided Conditional DPLM

This project implements a Guided Conditional Diffusion Protein Language Model (DPLM) that uses diffusion map representations to guide protein generation.

## Files

- `guided_dplm_model.py`: Contains the main `GuidedConditionalDPLM` model implementation.
- `diffmap_classifiers.py`: Implements the MLP and Transformer-based classifiers for processing diffusion maps.
- `guided_dplm_utils.py`: Utility functions for training, evaluation, and protein generation.
- `main.py`: The main script to run the training and generation process.
- `config.yaml`: Configuration file for model and training parameters.
