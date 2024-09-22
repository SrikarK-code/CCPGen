import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from guided_dplm_model import GuidedConditionalDPLM
from guided_dplm_utils import train_diffmap_classifier, evaluate_model, generate_proteins
from omegaconf import OmegaConf

def main():
    # Load configuration
    cfg = OmegaConf.load('config.yaml')
    
    # Initialize model
    model = GuidedConditionalDPLM(cfg)
    
    # Prepare data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size)
    
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()
    
    # Train the model
    train_diffmap_classifier(
        model, 
        train_dataloader, 
        val_dataloader, 
        optimizer, 
        criterion, 
        num_epochs=cfg.num_epochs, 
        device=cfg.device
    )
    
    # Evaluate the model
    test_loss = evaluate_model(model, test_dataloader, criterion, device=cfg.device)
    
    # Generate proteins
    sample_diffmap = torch.randn(1, cfg.diffmap_dim)  # Example diffusion map
    generated_proteins = generate_proteins(
        model, 
        sample_diffmap, 
        num_samples=cfg.num_generated_samples, 
        max_iter=cfg.max_iter, 
        temperature=cfg.temperature, 
        device=cfg.device
    )
    
    print("Generated protein sequences:")
    for i, protein in enumerate(generated_proteins):
        print(f"Protein {i+1}:", model.decoder.decode(protein))

if __name__ == "__main__":
    main()
