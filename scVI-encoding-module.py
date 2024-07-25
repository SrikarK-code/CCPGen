import scvi

class SCVIEncodingModule:
    def __init__(self):
        self.latent_representations = {}

    def encode(self, combined_datasets):
        for idx, adata in enumerate(combined_datasets):
            gene_symbol = adata.var_names[0]  # Assuming gene symbol is stored in var_names
            
            if gene_symbol in self.latent_representations:
                print(f"Embedding for gene symbol: {gene_symbol} already exists. Skipping.")
                continue
            
            print(f"Training and embedding for AnnData object {idx+1} associated with gene symbol: {gene_symbol} ...")

            scvi.model.SCVI.setup_anndata(adata)
            model = scvi.model.SCVI(adata)
            model.train()

            latent = model.get_latent_representation()
            self.latent_representations[gene_symbol] = latent

        return self.latent_representations