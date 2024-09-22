guidance classifier dplm
# Guided Conditional DPLM

This project implements a Guided Conditional Diffusion Protein Language Model (DPLM) that uses diffusion map representations to guide protein generation.

## Files

- `guided_dplm_model.py`: Contains the main `GuidedConditionalDPLM` model implementation.
- `diffmap_classifiers.py`: Implements the MLP and Transformer-based classifiers for processing diffusion maps.
- `guided_dplm_utils.py`: Utility functions for training, evaluation, and protein generation.
- `main.py`: The main script to run the training and generation process.
- `config.yaml`: Configuration file for model and training parameters.





How the DPLM works for protein generation conditional on the X_diffmap:

Input Processing:

The X_diffmap (cell representation in pseudotime) is fed into the DiffMap Classifier.
The classifier outputs an embedding that represents the expected protein characteristics for that cell state.


DPLM Conditioning:

The embedding from the DiffMap Classifier is used to condition the DPLM's generation process.
This is done by adding the classifier's output to the encoder's output features:
encoder_out['feats'] += diffmap_embedding.unsqueeze(1)


Initialization:

The generation starts with a sequence of mask tokens or a partially masked input sequence.


Iterative Refinement:

The DPLM uses an iterative refinement process to generate the protein sequence:
a. In each iteration, the model predicts the probabilities for each amino acid at each position.
b. Tokens are sampled based on these probabilities, with the sampling strategy controlled by parameters like temperature.
c. The process uses a "reparam_decoding" strategy, which helps in maintaining consistency with the conditioned information (X_diffmap).


Guidance from X_diffmap:

Throughout the generation process, the embedding derived from X_diffmap guides the DPLM:

It influences the initial state of the generation.
It affects the attention mechanisms in the DPLM, biasing the model towards generating sequences that are consistent with the given cell state.




Multiple Samples:

The process can generate multiple protein samples for a single X_diffmap, allowing for exploration of the protein space consistent with that cell state.


Post-processing:

The generated sequences are decoded from token IDs to amino acid sequences.



Key Points:

The X_diffmap provides a continuous representation of cell state, which is translated into the protein embedding space by the DiffMap Classifier.
This translation allows the DPLM to generate proteins that are consistent with specific points in the cell's pseudotime trajectory.
The iterative refinement process of the DPLM allows for gradual improvement of the generated sequence, always guided by the X_diffmap-derived embedding.
The stochastic nature of the generation process (controlled by temperature and sampling strategy) allows for diversity in the generated proteins while still maintaining consistency with the input cell state.
By generating multiple samples per X_diffmap, you can explore the range of proteins that might be associated with a particular cell state.
