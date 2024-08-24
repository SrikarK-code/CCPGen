def train(scvi_dim, prot_dim, hidden_dim, K, num_epochs=10, learning_rate=1e-4, batch_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and prepare data
    adata_list = load_adata('path to adata folder')
    adata_dict = prepare_adata_dict(adata_list)

    # Generate random protein sequences
    protein_sequences = {cell_type: [generate_random_protein_sequence() for _ in range(3)] for cell_type in adata_dict.keys()}

    # Initialize encoding modules
    prot_encoder = ProtT5EncodingModule()
    scvi_encoder = SCVIEncodingModule()

    # Encode data
    scvi_latents, scvi_pseudotimes = scvi_encoder.encode(adata_dict)
    latent_tensor, pseudotime_tensor, sequence_tensor = prepare_data(scvi_latents, scvi_pseudotimes, protein_sequences, prot_encoder)

    # Initialize MFM model
    mfm = MetricFlowMatching(scvi_dim, prot_dim, hidden_dim, K).to(device)

    # Train MFM
    train_mfm(mfm, latent_tensor, sequence_tensor, n_epochs=num_epochs, batch_size=batch_size, lr=learning_rate)

    return mfm, prot_encoder
