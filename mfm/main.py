from train import train
from models import sample_mfm
from encodings import ProtT5DecodingModule

def main():
    # Set hyperparameters
    scvi_dim = 50  # Adjust based on your scVI latent dimension
    prot_dim = 1024  # Adjust based on your ProtT5 latent dimension
    hidden_dim = 256
    K = 100
    num_epochs = 1000
    learning_rate = 1e-4
    batch_size = 128

    # Train the model
    mfm, prot_encoder = train(scvi_dim, prot_dim, hidden_dim, K, num_epochs, learning_rate, batch_size)

    # Initialize ProtT5 decoder
    prot_decoder = ProtT5DecodingModule()

    # Example of generating a new protein sequence
    new_scvi_latent = torch.randn(1, scvi_dim)  # Replace with actual new scVI latent
    new_prot_latent = sample_mfm(mfm, new_scvi_latent)
    new_protein_sequence = prot_decoder(new_prot_latent)

    print("Generated protein sequence:", new_protein_sequence)

if __name__ == "__main__":
    main()
