from transformers import T5EncoderModel, T5Tokenizer
import torch.nn as nn
import re
import torch
from transformers import T5ForConditionalGeneration
import scvi
import scanpy as sc
import numpy as np

class ProtT5EncodingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.protT5_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
        self.protT5_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_bfd")

    def forward(self, sequence):
        processed_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        ids = self.protT5_tokenizer(processed_seq, add_special_tokens=True, return_tensors="pt", padding='longest')
        input_ids = ids['input_ids'].to(self.protT5_model.device)
        attention_mask = ids['attention_mask'].to(self.protT5_model.device)

        with torch.no_grad():
            embedding_repr = self.protT5_model(input_ids=input_ids, attention_mask=attention_mask)

        seq_emb = embedding_repr.last_hidden_state
        return seq_emb

class ProtT5DecodingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.protT5_model = T5ForConditionalGeneration.from_pretrained("Rostlab/prot_t5_xl_bfd")
        self.protT5_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_bfd")

    def forward(self, latent_repr, max_length=200):
        outputs = self.protT5_model.generate(
            inputs_embeds=latent_repr,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )

        decoded_sequences = self.protT5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded_sequences

class SCVIEncodingModule:
    def __init__(self):
        self.latent_representations = {}
        self.pseudotime_representations = {}

    def encode(self, adata_dict):
        for cell_type, adata in adata_dict.items():
            print(f"Training and embedding for cell type: {cell_type}...")

            adata_copy = adata.copy()

            nan_count = np.isnan(adata_copy.X).sum()
            if nan_count > 0:
                print(f"There are {nan_count} NaN values in adata_copy.X for cell type: {cell_type}")
            else:
                print(f"No NaN values found in adata_copy.X for cell type: {cell_type}")

            latent = adata_copy.obsm['X_diffmap']
            pseudotime = adata_copy.obs['dpt_pseudotime']

            # Store latent representation in the dictionary
            self.latent_representations[cell_type] = latent
            self.pseudotime_representations[cell_type] = pseudotime

        print("Encoding completed.")

        return self.latent_representations, self.pseudotime_representations
