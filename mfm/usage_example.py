scvi_dim = your_scvi_latent_dim
prot_dim = your_prot_latent_dim
hidden_dim = 256  # just for sake of example
K = 100  # of centeres for rbf_metric
mfm = MetricFlowMatching(scvi_dim, prot_dim, hidden_dim, K)

train_mfm(mfm, scvi_latents, prot_latents)

new_scvi_latent = scvi_encoder(new_scvi_data)  
new_prot_latent = sample_mfm(mfm, new_scvi_latent)

new_protein_sequence = prot_decoder(new_prot_latent)
