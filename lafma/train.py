import torch
from torch.optim import Adam
from models import CustomUNet1D
from flow_matching import FlowMatchingTrainer, ProteinFlowMatching
from encoding import ProtT5EncodingModule, ProtT5DecodingModule, SCVIEncodingModule
from data_processing import load_adata, prepare_adata_dict, generate_random_protein_sequence, prepare_data, create_dataloaders

def train(num_epochs=1, learning_rate=1e-5, batch_size=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and prepare data
    adata_list = load_adata('/content/drive/MyDrive/tf-flow-design/combined_adata_folder/')
    adata_dict = prepare_adata_dict(adata_list)

    # Generate random protein sequences
    protein_sequences = {cell_type: [generate_random_protein_sequence() for _ in range(3)] for cell_type in adata_dict.keys()}

    # Initialize encoding modules
    prot_encoder = ProtT5EncodingModule()
    scvi_encoder = SCVIEncodingModule()

    # Encode data
    scvi_latents = scvi_encoder.encode(adata_dict)
    latent_tensor, sequence_tensor = prepare_data(scvi_latents, protein_sequences, prot_encoder)

    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(latent_tensor, sequence_tensor, batch_size)

    # Initialize models
    unet = CustomUNet1D(
        in_channels=50,
        out_channels=1024,
        model_channels=64,
        num_res_blocks=2,
        attention_resolutions=(1,),
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        use_spatial_transformer=True,
        transformer_depth=1,
        context_dim=1,
    )

    flow_matching = FlowMatchingTrainer(unet, sample_N=25)
    decoder = ProtT5DecodingModule()
    model = ProteinFlowMatching(flow_matching, decoder)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for scvi_latent, protein_seq in train_dataloader:
            scvi_latent = scvi_latent.to(device)
            protein_seq = protein_seq.to(device)

            optimizer.zero_grad()
            loss = model(scvi_latent, protein_seq)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for scvi_latent, pseudotime, protein_seq in val_dataloader:
                scvi_latent = scvi_latent.to(device)
                protein_seq = protein_seq.to(device)
                loss = model(scvi_latent, protein_seq)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    # Generate new protein sequences
    model.eval()
    with torch.no_grad():
        scvi_latent = torch.randn(2, scvi_latents[list(scvi_latents.keys())[0]].shape[1]).to(device)  # random scVI latent with batch size
        generated_sequence = model.generate(scvi_latent)
        print("Generated sequence:", generated_sequence)

if __name__ == "__main__":
    train()
