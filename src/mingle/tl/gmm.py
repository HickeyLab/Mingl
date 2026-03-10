import numpy as np
import pandas as pd
from scipy.stats import norm
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import Optional, Union, Dict, Sequence
import anndata as ad
import math
from .knn2 import KNN2
import multiprocessing
#multiprocessing.set_start_method('spawn')

# Helper function to calculate probabilities for each cell (moved outside for parallelization)
def calculate_probabilities_for_cell(args):
    # Unpack the arguments
    cell_index, windows2, centroid_rows, cell_type_features, neighborhood_col = args
    neighborhood_probs = {}

    # Iterate through each centroid (neighborhood)
    for _, centroid_row in centroid_rows:
        neighborhood_name = _#centroid_row[neighborhood_col]
        total_prob = 1

        # For each cell type, calculate the probability
        for cell_type in cell_type_features:
            mean_col = f'{cell_type}_mean'
            std_col = f'{cell_type}_std'

            if mean_col in centroid_row and std_col in centroid_row:
                mean = centroid_row[mean_col] 
                std = centroid_row[std_col]
                #mean = math.trunc(mean * 1_000_000) / 1_000_000
                #std = math.trunc(std * 1_000_000) / 1_000_000
                # Get the value of the current cell for this cell type
                cell_value = windows2.loc[cell_index, cell_type] if cell_type in windows2.columns else np.nan

                # Check if std is zero, calculate probability
                if std == 0:
                    cell_prob = 1 if cell_value == mean else 0
                else:
                    cell_prob = norm.pdf(cell_value, loc=mean, scale=std)

                total_prob *= cell_prob  # Multiply for each cell type

        # Store the neighborhood probability for this cell
        neighborhood_probs[neighborhood_name] = total_prob

    # Normalize the probabilities to sum to 1
    total_prob_sum = sum(neighborhood_probs.values())
    for neighborhood in neighborhood_probs:
        neighborhood_probs[neighborhood] /= total_prob_sum

    return neighborhood_probs

def cpu_gmm_probability(
    CELLS_ADATA: ad.AnnData,
    CENTROIDS_ADATA: ad.AnnData,
    *,
    cluster_col: str = "cell_type",  # Default cluster column in obs
    neighborhood_col: str = "neighborhood",  # Default neighborhood column in obs
    region_key: str = "unique_region",
    ks: Sequence[int] = (10, 20, 100, 300),  # List of k values for neighbors
    k: int = 10,
    threshold: float = 0.25,  # Probability threshold for counting
    num_processes: Optional[int] = None,  # Optional: number of processes for parallelism (defaults to max CPUs)
) -> ad.AnnData:
    """
    Calculate GMM probabilities for each cell's assigned neighborhood and return the AnnData object with probabilities stored in `obsm`.

    Parameters
    ----------
    CELLS_ADATA
        AnnData object containing cell-level data (with cluster labels and coordinates).
    CENTROIDS_ADATA
        AnnData object containing centroid data with neighborhood means and standard deviations for each cell type.
    cluster_col
        Column in `CELLS_ADATA.obs` representing the cluster or cell type (default is 'cell_type').
    neighborhood_col
        Column in `CELLS_ADATA.obs` representing the neighborhood assignment (default is 'neighborhood').
    ks
        Sequence of values for k (the number of neighbors) to compute neighborhood summaries.
    threshold
        Threshold probability value to count neighborhoods (default is 0.25).
    num_processes
        Number of parallel processes to use for computation (default is None, which uses all available CPUs).

    Returns
    -------
    CELLS_ADATA
        AnnData object with computed neighborhood probabilities stored in `obsm["neighborhood_probabilities"]`.
    """

    # Ensure neighborhood columns exist in obs
    if neighborhood_col not in CELLS_ADATA.obs or cluster_col not in CELLS_ADATA.obs:
        raise KeyError(f"One or more required columns ({neighborhood_col}, {cluster_col}) are missing in obs.")

    # Step 1: Get KNN neighborhood windows
    windows = KNN2(CELLS_ADATA, cluster_col=cluster_col, region_key=region_key, ks=ks)
    windows2 = windows[k]
    windows2[cluster_col] = CELLS_ADATA.obs[cluster_col].values

    # Step 2: List of neighborhoods and cell types to loop through
    #neighborhoods_to_loop = CELLS_ADATA.obs[neighborhood_col].unique().tolist()
    cell_type_features = CELLS_ADATA.obs[cluster_col].unique()

    # Extract centroid data as a list of rows (to pass to the multiprocessing pool)
    centroid_rows = CENTROIDS_ADATA.to_df().iterrows() #HERE
    centroid_rows = [(idx, row.to_dict()) for idx, row in centroid_rows]

    # Function to parallelize calculations across all cells
    def parallelize_probability_calculations(windows2, centroid_rows, cell_type_features, neighborhood_col, num_processes):
        if num_processes is None:
            num_processes = cpu_count()

        print(f"Using {num_processes} processes.")

        task_args = [
            (cell_index, windows2, centroid_rows, cell_type_features, neighborhood_col)
            for cell_index in windows2.index
        ]

        with Pool(num_processes) as pool:
            results = pool.map(calculate_probabilities_for_cell, task_args)

        return results

    #print(windows2)

    # Parallelize the calculations
    probabilities_list = parallelize_probability_calculations(windows2, centroid_rows, cell_type_features, neighborhood_col, num_processes)

    # Convert the results into a DataFrame
    probabilities_df = pd.DataFrame(probabilities_list, index=windows2.index)

    # Step 3: Attach probabilities to AnnData object (store in obsm)
    CELLS_ADATA.obsm["neighborhood_probabilities"] = probabilities_df.values
    CELLS_ADATA.uns["neighborhood_probability_neighborhoods"] = list(probabilities_df.columns)

    # Return the updated AnnData object
    return CELLS_ADATA
