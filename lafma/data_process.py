import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import anndata
import scvi
import os
import numpy as np
import random

def load_adata(folder_path, max_files=5):
    file_count = 0
    adata_list = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.h5ad'):
            file_path = os.path.join(folder_path, filename)
            print(filename)
            adata = anndata.read_h5ad(file_path)
            adata_list.append(adata)
            file_count += 1
            if file_count >= max_files:
                break

    print(f'Read and stored {file_count} .h5ad files.')
    return adata_list

def prepare_adata_dict(adata_list):
    adata_dict = {}
    for adata in adata_list:
        cell_types = adata.obs['cell_ontology_class'].unique()
        for cell_type in cell_types:
            if cell_type != 'mesenchymal stem cell':
                if cell_type not in adata_dict:
                    adata_dict[cell_type] = adata
                else:
                    adata_dict[cell_type] = anndata.concat([adata_dict[cell_type], adata])
    return adata_dict

def generate_random_protein_sequence(min_length=100, max_length=200):
    valid_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    length = random.randint(min_length, max_length)
    return ''.join(random.choices(valid_amino_acids, k=length))

def prepare_data(adata_dict, protein_sequences, encoder):
    latent_list = []
    sequence_list = []

    for cell_type, latents in adata_dict.items():
        for latent, sequence in zip(latents, protein_sequences[cell_type]):
            latent_list.append(latents)
            sequence_list.append(sequence)

    # Encode sequences using ProtT5
    sequence_tensor_list = [encoder(sequence) for sequence in sequence_list]

    # Pad sequences
    max_len = max(sequence.shape[1] for sequence in sequence_tensor_list)
    padded_sequence_tensor_list = [
        torch.cat([sequence, torch.zeros(1, max_len - sequence.shape[1], sequence.shape[2], dtype=sequence.dtype)], dim=1)
        if sequence.shape[1] < max_len else sequence for sequence in sequence_tensor_list
    ]
    sequence_tensor = torch.cat(padded_sequence_tensor_list, dim=0)

    # Process latents
    latent_tensor_list = [torch.tensor(latent, dtype=torch.float32) for latent in latent_list]
    max_len = max(latent.shape[0] for latent in latent_tensor_list)
    padded_latent_list = [
        torch.cat([latent, torch.zeros((max_len - latent.shape[0], latent.shape[1]), dtype=latent.dtype)], dim=0)
        if latent.shape[0] < max_len else latent for latent in latent_tensor_list
    ]
    latent_tensor = torch.stack(padded_latent_list, dim=0)

    return latent_tensor, sequence_tensor
    
def create_dataloaders(latent_tensor, pseudotime_tensor, sequence_tensor, batch_size=2):
    full_dataset = TensorDataset(latent_tensor, pseudotime_tensor, sequence_tensor)

    # Split size calculation
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = total_size - train_size

    # Dataset split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Dataloader creation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader
