import os
import itertools
import multiprocessing
import pickle
import scanpy as sc
import anndata
import bbknn
import palantir
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from functools import partial

# import os
# import subprocess
# import warnings
# import warnings
# from numba.core.errors import NumbaDeprecationWarning
# import pandas as pd
# import networkx as nx
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from numba.core.errors import NumbaWarning
# import scanpy as sc
# import pickle
# import scanpy as sc
# import anndata
# import pandas as pd
# import os
# import pickle
# import palantir
# import bbknn

# Inline plotting
%matplotlib inline
warnings.simplefilter('ignore', category=NumbaWarning)

def combine_data(adata, output_dir):
    """
    Combine data for each unique pair of cell ontology class and organ tissue combinations.
    """
    combined_adatas = {}
    os.makedirs(output_dir, exist_ok=True)

    unique_combinations = adata.obs[['cell_ontology_class', 'organ_tissue']].drop_duplicates()
    total_combinations = unique_combinations.shape[0]
    print(f"Total unique combinations: {total_combinations}")

    # Create a dictionary to store subsets for each unique combination
    subsets = {}
    for idx, (class_name, organ) in enumerate(unique_combinations.itertuples(index=False), 1):
        criteria = (adata.obs['cell_ontology_class'] == class_name) & (adata.obs['organ_tissue'] == organ)
        subset = adata[criteria]

        if subset.shape[0] == 0:
            print(f"No cells found for class {class_name} in organ {organ}. Skipping...")
            continue

        subsets[(class_name, organ)] = subset
        print(f"Processed {idx}/{total_combinations} individual combinations")

    # Generate all possible pairs of combinations
    all_pairs = list(itertools.combinations(subsets.keys(), 2))
    total_pairs = len(all_pairs)
    print(f"Total number of pairs to process: {total_pairs}")

    for idx, ((class1, organ1), (class2, organ2)) in enumerate(all_pairs, 1):
        combined_adata_file = os.path.join(output_dir, f'combined_adata_{class1}_{organ1}__{class2}_{organ2}.h5ad')

        if os.path.exists(combined_adata_file):
            print(f"File already exists for {class1}_{organ1} __ {class2}_{organ2}. Skipping...")
            continue

        subset1 = subsets[(class1, organ1)]
        subset2 = subsets[(class2, organ2)]

        combined_adata = anndata.concat([subset1, subset2])
        combined_adatas[((class1, organ1), (class2, organ2))] = combined_adata
        combined_adata.write(combined_adata_file)

        print(f"Processed pair {idx}/{total_pairs}: {class1}_{organ1} __ {class2}_{organ2}")

    combined_adatas_files = {((class1, organ1), (class2, organ2)):
                             os.path.join(output_dir, f'combined_adata_{class1}_{organ1}__{class2}_{organ2}.h5ad')
                             for ((class1, organ1), (class2, organ2)) in combined_adatas}

    with open(os.path.join(output_dir, 'combined_adatas_files.pkl'), 'wb') as f:
        pickle.dump(combined_adatas_files, f)

    return combined_adatas_files

def run_bbknn_integration(adata, batch_key='organ_tissue'):
    """
    Run BBKNN integration on the AnnData object.
    """
    bbknn.bbknn(adata, batch_key=batch_key)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=[batch_key, 'cell_ontology_class'], save='_cell_types_after_bbknn.png')
    return adata

def perform_pseudotime_analysis(adata, start_class, start_organ):
    """
    Perform pseudotime analysis with a specified starting point.
    """
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)

    sc.tl.umap(adata)
    sc.tl.diffmap(adata, n_comps=10)
    sc.pp.neighbors(adata, use_rep='X_diffmap')

    # Set the root cell based on the specified starting point
    start_mask = (adata.obs['cell_ontology_class'] == start_class) & (adata.obs['organ_tissue'] == start_organ)
    if np.sum(start_mask) == 0:
        raise ValueError(f"No cells found for starting class {start_class} in organ {start_organ}")
    adata.uns['iroot'] = np.flatnonzero(start_mask)[0]
    sc.tl.dpt(adata, n_dcs=10)

    return adata

def run_palantir_analysis(adata, start_class, start_organ):
    """
    Run Palantir analysis with a specified starting point.
    """
    dm_res = palantir.utils.run_diffusion_maps(adata, n_components=10)
    ms_data = palantir.utils.determine_multiscale_space(adata)

    imputed_data = palantir.utils.run_magic_imputation(adata)

    palantir.plot.plot_diffusion_components(adata)
    plt.savefig('diffusion_components.png')
    plt.close()

    # Set the start cell based on the specified starting point
    start_mask = (adata.obs['cell_ontology_class'] == start_class) & (adata.obs['organ_tissue'] == start_organ)
    if np.sum(start_mask) == 0:
        raise ValueError(f"No cells found for starting class {start_class} in organ {start_organ}")
    start_cell = adata.obs_names[start_mask][0]

    # Define terminal states
    terminal_states = pd.Series(index=[], dtype='object')
    for cell_type, organ in adata.obs[['cell_ontology_class', 'organ_tissue']].drop_duplicates().itertuples(index=False):
        if cell_type != start_class or organ != start_organ:
            mask = (adata.obs['cell_ontology_class'] == cell_type) & (adata.obs['organ_tissue'] == organ)
            sampled_cells = adata.obs[mask].sample(n=min(10, mask.sum())).index
            for cell in sampled_cells:
                terminal_states[cell] = f"{cell_type}_{organ}"

    pr_res = palantir.core.run_palantir(adata, start_cell, num_waypoints=500, terminal_states=terminal_states)

    palantir.plot.plot_palantir_results(adata, pr_res)
    plt.savefig('palantir_results.png')
    plt.close()

    return pr_res

def run_grnboost2(adata):
    """
    Run GRNBoost2 on the AnnData object.
    """
    expression_matrix = adata.to_df()
    adjacency_matrix = grnboost2(expression_data=expression_matrix, verbose=True)
    return adjacency_matrix

def prioritize_genes(adj_matrix, start_class, start_organ, end_class, end_organ, output_dir):
    """
    Prioritize genes using PageRank algorithm.
    """
    G = nx.from_pandas_edgelist(df=adj_matrix, source='TF', target='target', edge_attr='importance')
    print(f'Loaded {len(G.nodes):,} genes with {len(G.edges):,} edges.')

    cutoff = 1
    print(f'Removing all edges with weight < {cutoff}...\n')
    bad_edges = [(s,t,w) for (s,t,w) in G.edges.data('importance') if w < cutoff]
    G.remove_edges_from(bad_edges)
    print(f'Graph now has {len(G.nodes):,} genes and {len(G.edges):,} edges.')

    pr = nx.pagerank(G, alpha=0.85, max_iter=50, weight='importance')

    prdf = pd.DataFrame(pd.Series(pr)).reset_index()
    prdf.columns = ['Gene', 'PageRank']

    ranked_genes = prdf.sort_values('PageRank', ascending=False)
    ranked_genes_file = os.path.join(output_dir, f'ranked_genes_{start_class}_{start_organ}_to_{end_class}_{end_organ}.csv')
    ranked_genes.to_csv(ranked_genes_file, index=False)

    return ranked_genes

def integrated_analysis(input_dir, output_dir):
    """
    Perform integrated analysis on all .h5ad files in the input directory.
    """
    all_results = {}
    processed_files = 0

    with open(os.path.join(input_dir, 'combined_adatas_files.pkl'), 'rb') as f:
        combined_adatas_files = pickle.load(f)

    for ((start_class, start_organ), (end_class, end_organ)), file_path in combined_adatas_files.items():
        adata = sc.read_h5ad(file_path)

        # Run BBKNN integration
        adata = run_bbknn_integration(adata)

        # Perform pseudotime analysis in both directions
        for direction in ['forward', 'reverse']:
            if direction == 'forward':
                start_c, start_o, end_c, end_o = start_class, start_organ, end_class, end_organ
            else:
                start_c, start_o, end_c, end_o = end_class, end_organ, start_class, start_organ

            adata_pseudo = perform_pseudotime_analysis(adata.copy(), start_c, start_o)
            sc.pl.umap(adata_pseudo, color=['dpt_pseudotime', 'cell_ontology_class', 'organ_tissue'],
                       save=f'_{start_c}_{start_o}_to_{end_c}_{end_o}_umap_pseudotime.png')

            # Run Palantir analysis
            pr_res = run_palantir_analysis(adata_pseudo, start_c, start_o)

            # Run GRNBoost2 (same for both directions)
            if direction == 'forward':
                adjacency_matrix = run_grnboost2(adata)

            # Prioritize genes
            ranked_genes = prioritize_genes(adjacency_matrix, start_c, start_o, end_c, end_o, output_dir)

            # Save top 10 genes to a file
            top_genes_output_path = os.path.join(output_dir, f"{start_c}_{start_o}_to_{end_c}_{end_o}_top_genes.txt")
            with open(top_genes_output_path, 'w') as f:
                for gene in ranked_genes['Gene'].head(10):
                    f.write(f"{gene}\n")

            all_results[(start_c, start_o, end_c, end_o)] = {
                'adata': adata_pseudo,
                'palantir_results': pr_res,
                'adjacency_matrix': adjacency_matrix,
                'ranked_genes': ranked_genes
            }

        processed_files += 1
        print(f"\nProcessed files: {processed_files}")

    return all_results

def combine_data_parallel(adata, output_dir, num_processes=4):
    """
    Combine data for each unique pair of cell ontology class and organ tissue combinations in parallel.
    """
    os.makedirs(output_dir, exist_ok=True)

    unique_combinations = adata.obs[['cell_ontology_class', 'organ_tissue']].drop_duplicates()
    all_pairs = list(itertools.combinations(unique_combinations.itertuples(index=False), 2))

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(partial(process_pair, adata=adata, output_dir=output_dir), all_pairs)

    combined_adatas_files = {pair: file_path for pair, file_path in results if file_path is not None}

    with open(os.path.join(output_dir, 'combined_adatas_files.pkl'), 'wb') as f:
        pickle.dump(combined_adatas_files, f)

    return combined_adatas_files

def process_pair(pair, adata, output_dir):
    (class1, organ1), (class2, organ2) = pair
    combined_adata_file = os.path.join(output_dir, f'combined_adata_{class1}_{organ1}__{class2}_{organ2}.h5ad')

    if os.path.exists(combined_adata_file):
        return pair, combined_adata_file

    criteria1 = (adata.obs['cell_ontology_class'] == class1) & (adata.obs['organ_tissue'] == organ1)
    criteria2 = (adata.obs['cell_ontology_class'] == class2) & (adata.obs['organ_tissue'] == organ2)
    subset1 = adata[criteria1]
    subset2 = adata[criteria2]

    if subset1.shape[0] == 0 or subset2.shape[0] == 0:
        return pair, None

    combined_adata = anndata.concat([subset1, subset2])
    combined_adata.write(combined_adata_file)

    return pair, combined_adata_file

def integrated_analysis_parallel(input_dir, output_dir, num_processes=4):
    """
    Perform integrated analysis on all .h5ad files in the input directory in parallel.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(input_dir, 'combined_adatas_files.pkl'), 'rb') as f:
        combined_adatas_files = pickle.load(f)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(partial(process_file, output_dir=output_dir), combined_adatas_files.items())

    all_results = {key: value for key, value in results if value is not None}
    return all_results

def process_file(item, output_dir):
    ((start_class, start_organ), (end_class, end_organ)), file_path = item
    adata = sc.read_h5ad(file_path)

    try:
        # Run BBKNN integration
        adata = run_bbknn_integration(adata)

        results = {}
        for direction in ['forward', 'reverse']:
            if direction == 'forward':
                start_c, start_o, end_c, end_o = start_class, start_organ, end_class, end_organ
            else:
                start_c, start_o, end_c, end_o = end_class, end_organ, start_class, start_organ

            adata_pseudo = perform_pseudotime_analysis(adata.copy(), start_c, start_o)
            pr_res = run_palantir_analysis(adata_pseudo, start_c, start_o)

            if direction == 'forward':
                adjacency_matrix = run_grnboost2(adata)

            ranked_genes = prioritize_genes(adjacency_matrix, start_c, start_o, end_c, end_o, output_dir)

            results[direction] = {
                'adata': adata_pseudo,
                'palantir_results': pr_res,
                'ranked_genes': ranked_genes
            }

        results['adjacency_matrix'] = adjacency_matrix
        return ((start_class, start_organ, end_class, end_organ), results)

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return ((start_class, start_organ, end_class, end_organ), None)

# Example usage:
if __name__ == "__main__":
    adata = sc.read_h5ad('/content/drive/MyDrive/Programmable Biology Group/Srikar/tf-flow-design/raw_data/TabulaSapiens.h5ad')
    combined_data_dir = "/content/drive/MyDrive/Programmable Biology Group/Srikar/tf-flow-design/concat_adata/"
    output_dir = '/content/drive/MyDrive/Programmable Biology Group/Srikar/tf-flow-design/data_generation/'
    
    # Parallel AnnData combination
    combined_adatas_files = combine_data_parallel(adata, combined_data_dir, num_processes=8)
    
    # Parallel integrated analysis
    results = integrated_analysis_parallel(combined_data_dir, output_dir, num_processes=8)
