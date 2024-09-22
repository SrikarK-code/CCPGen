import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_diffmap_classifier(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, device):
    model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            X_diffmap = batch['X_diffmap'].to(device)
            protein_embedding = batch['protein_embedding'].to(device)
            
            optimizer.zero_grad()
            pred_embedding = model.diffmap_classifier(X_diffmap)
            loss = criterion(pred_embedding, protein_embedding)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                X_diffmap = batch['X_diffmap'].to(device)
                protein_embedding = batch['protein_embedding'].to(device)
                
                pred_embedding = model.diffmap_classifier(X_diffmap)
                loss = criterion(pred_embedding, protein_embedding)
                
                val_loss += loss.item()
        
        val_loss /= len(val_dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model")

def evaluate_model(model, test_dataloader, criterion, device):
    model.to(device)
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            X_diffmap = batch['X_diffmap'].to(device)
            protein_embedding = batch['protein_embedding'].to(device)
            
            pred_embedding = model.diffmap_classifier(X_diffmap)
            loss = criterion(pred_embedding, protein_embedding)
            
            test_loss += loss.item()
    
    test_loss /= len(test_dataloader)
    print(f"Test Loss: {test_loss:.4f}")
    
    return test_loss

def generate_proteins(model, X_diffmap, num_samples=5, max_iter=50, temperature=1.0, device='cuda'):
    model.to(device)
    model.eval()
    
    batch = {
        'X_diffmap': X_diffmap.to(device),
        'prev_tokens': torch.randint(0, model.decoder.vocab_size, (X_diffmap.size(0), 100)).to(device),
        'motif_mask': torch.ones(X_diffmap.size(0), 100, dtype=torch.bool).to(device)
    }
    
    with torch.no_grad():
        generated_proteins = model.generate(
            batch,
            max_iter=max_iter,
            temperature=temperature,
            sampling_strategy='gumbel_argmax',
            num_samples=num_samples
        )
    
    return generated_proteins
