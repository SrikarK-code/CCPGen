a) Input: scVI embeddings of cell states
b) Output: Generated protein sequences

Process Flow:

scVI embeddings → UNetModel → Flow Matching → Protein Decoder → Protein Sequence

1. Input Preparation:
   - Start with scVI (single-cell Variational Inference) embeddings of cell states.
   - These embeddings are high-dimensional vectors (e.g., 128 dimensions) representing cellular gene expression profiles.
   - The scVI embeddings are generated using a separate scVI model trained on single-cell RNA sequencing data.

2. Model Architecture Overview:
   The system consists of several key components:
   a) UNetModel: The core generative model
   b) FlowMatchingTrainer: Manages the flow matching process
   c) ProtT5Encoder: Encodes protein sequences (for training)
   d) ProtT5Decoder: Decodes latent representations to protein sequences

3. UNetModel Detailed Architecture:
   3.1. Initialization:
   - The UNetModel is initialized with parameters like input/output channels, number of ResBlocks, attention resolutions, etc.
   - Key components are created: time embedder, input blocks, middle block, output blocks.

   3.2. Time Embedding:
   - Function: timestep_embedding
   - Converts a scalar timestep to a high-dimensional vector using sinusoidal functions.
   - This embedding is further processed through a small MLP (self.time_embed).

   3.3. Input Blocks:
   - A series of TimestepEmbedSequential modules, each containing:
     a) ResBlock: Combines feature maps with time embeddings
     b) AttentionBlock or SpatialTransformer: For self-attention mechanisms
     c) Downsample: Reduces spatial dimensions (if applicable)

   3.4. Middle Block:
   - Contains ResBlocks and Attention mechanisms for global reasoning.

   3.5. Output Blocks:
   - Mirror the input blocks, but with Upsample layers instead of Downsample.
   - Use skip connections from input blocks.

   3.6. Final Output Layer:
   - Normalization followed by a convolution to produce the output channels.

4. Detailed Component Breakdown:
   4.1. ResBlock:
   - Residual block that processes features and incorporates time embeddings.
   - Contains normalization layers, convolutions, and optional up/downsampling.
   - Uses checkpoint function for memory-efficient backpropagation.

   4.2. AttentionBlock:
   - Self-attention mechanism allowing interaction between different parts of the sequence.
   - Uses QKVAttention for efficient attention computation.

   4.3. SpatialTransformer:
   - More sophisticated attention mechanism with multiple transformer layers.
   - Each layer contains self-attention and feed-forward networks.

   4.4. CrossAttention:
   - Attention mechanism that can attend to a separate context (used in SpatialTransformer).
   - Splits input into query, key, and value before computing attention.

   4.5. FeedForward:
   - Simple feedforward network used in transformer blocks.
   - Contains two linear layers with GELU activation and dropout.

   4.6. Upsample and Downsample:
   - Handle changes in spatial dimensions of feature maps.
   - Use either interpolation or transposed convolutions.

   4.7. GroupNorm32:
   - Custom group normalization for improved training stability.

   4.8. TimestepEmbedSequential:
   - Sequential module that handles passing of timestep embeddings to appropriate submodules.

5. FlowMatchingTrainer:
   - Manages the training process of the UNetModel.
   - Implements the forward process (adding noise) and reverse process (denoising).
   - Uses a noise schedule to control the amount of noise added at each timestep.
   - Computes loss based on the model's ability to predict the noise added.

6. ProtT5Encoder (for training):
   - Utilizes a pre-trained ProtT5 model to encode protein sequences into a latent space.
   - Processes amino acid sequences into a high-dimensional representation.

7. ProtT5Decoder (for inference):
   - Converts latent representations back into amino acid sequences.
   - Uses beam search or other decoding strategies to generate the final protein sequence.

8. Training Process:
   8.1. Data Preparation:
   - Batch of scVI embeddings and corresponding protein sequences are loaded.
   - Protein sequences are encoded using ProtT5Encoder.

   8.2. Forward Pass:
   - Random timesteps are generated for each sample in the batch.
   - Noise is added to the encoded protein sequences based on the timesteps.
   - The UNetModel processes the noisy encodings, conditioned on scVI embeddings and timesteps.

   8.3. Loss Computation:
   - The model's output is compared to the true noise added.
   - Loss is calculated (usually mean squared error).

   8.4. Backpropagation:
   - Gradients are computed and model parameters are updated.

9. Inference Process:
   9.1. Start with an scVI embedding of a cell state.
   9.2. Generate random noise as the starting point.
   9.3. Gradually denoise using the UNetModel:
      - For each timestep (from most noisy to least):
        - Pass the current noisy sample through the UNetModel.
        - Use the model's prediction to update the sample.
   9.4. The final denoised representation is passed through the ProtT5Decoder.
   9.5. The decoder outputs the generated protein sequence.

10. Utility Functions:
    - conv_nd: Creates 1D convolutions for our sequence data.
    - zero_module: Initializes a module's parameters to zero.
    - normalization: Applies GroupNorm32 normalization.
    - checkpoint: Implements gradient checkpointing for memory efficiency.
    - exists and default: Helper functions for handling optional parameters.

Cool adaptations
    - Adaptation of 2D UNet architecture to 1D protein sequences.
        - Unet in original model was used for the audio representations (spectograms)
        - While we do use protein embeddings (like those from ProtT5), the UNet in our case still operates on a 1D sequence of these embeddings.
        - Each position in this sequence corresponds to an amino acid, but is represented by a high-dimensional vector.
        - The UNet processes this sequence of vectors, maintaining the 1D structure of the protein
            - each element of the sequence is itself a rich high-dimensional representation (1280)
    - Integration of flow matching with protein language models.
    - Use of scVI embeddings as conditional input for targeted protein generation.


Loss:
θ^ = argmin_θ E_t,z_t ||u_θ(z_t, t, c) - v_t||^2
Where:

u_θ is your flow matching model (UNet)
z_t is the scVI embedding at time t
t is the timestep
c is your context (which we'll discuss next)
v_t is the target velocity (z_1 - (1-σ_min)z_0 in the optimal transport formulation)


UNet
This 1D UNet processes the scVI latent representations, which encode cellular states, through a series of downsampling and upsampling operations.
The input blocks progressively reduce the spatial dimensions while increasing the channel depth, capturing hierarchical features.
The middle block, with its deep channel representation, allows for global reasoning across the entire sequence.
The output blocks then gradually upsample the representation back to the original dimensions, utilizing skip connections to preserve fine-grained information.
Time embeddings are crucial, allowing the model to understand its position in the generation process.
These embeddings are added to the input at each step, guiding the transformation from noise to protein sequence.
Attention mechanisms, implemented either as AttentionBlocks or SpatialTransformers, enable the model to capture long-range dependencies critical for protein structure.
The ResBlocks incorporate both the current state and the time embedding, allowing for time-dependent processing at each level.
The context dimension, which could include additional information like pseudotime or motif data, is integrated through the SpatialTransformer blocks, providing extra conditioning for the generation process.
The model's output represents the velocity field in the flow matching framework, predicting how the latent representation should change at each step to transform noise into a meaningful protein sequence representation.

