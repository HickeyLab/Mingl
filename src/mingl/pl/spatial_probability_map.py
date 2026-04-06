import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..tl.knn2 import KNN2


def spatial_probability_mapping(
    adata,
    centroids,
    cell_type_features,
    *,
    k=300,
    batch_size=20000,
    desired_region="B008_Sigmoid",
    reg = "unique_region",
    cluster_col = "Community",
    X = "x" , # Variable for the X coordinate
    Y = "y",  # Variable for the Y coordinate
    neigh = "Neighborhood",
    tiss_unit = "Tissue Unit",
    cell_type = "Cell Type"
):
    # Equivalent to: df = pd.read_csv(...)
    df = adata.obs.copy()
    df_centroids = centroids

    # KNN
    cells = df
    
    keep_cols = [X, Y, reg, cluster_col, tiss_unit, neigh, cell_type]
    ks = [10,100,300]
    # IMPORTANT CHANGE: KNN takes adata (per your note)
    windows = KNN2(adata, cluster_col=cluster_col, keep_obs_cols=keep_cols, ks = ks)
    k = k
    windows2 = windows[k]

    # Add cell type column to output windows dataframe
    # (Preserves your exact logic: windows2 gets Community from cells)
    windows2[cluster_col] = cells[cluster_col]

    # communties
    

    # Adjust batch size according to your GPU memory (~8GB)
    batch_size = batch_size

    num_cells = len(windows2)
    num_batches = (num_cells + batch_size - 1) // batch_size  # ceiling division

    # Extract neighborhood names and centroids
    neighborhood_names = df_centroids["Tissue Unit"].values
    mean_cols = [f"{ct}_mean" for ct in cell_type_features]
    std_cols = [f"{ct}_std" for ct in cell_type_features]

    # Convert means and stds once to GPU arrays
    means = np.array(df_centroids[mean_cols].values)  # shape (num_neighborhoods, num_cell_types)
    stds = np.array(df_centroids[std_cols].values)  # same shape

    def compute_batch_probs(batch_df):
        # Convert cell data to GPU array
        cell_data = np.array(batch_df[cell_type_features].values)  # (batch_size, num_cell_types)

        # Broadcast dims for probability calc
        cell_data_exp = cell_data[:, np.newaxis, :]  # (batch_size, 1, num_cell_types)
        means_exp = means[np.newaxis, :, :]  # (1, num_neighborhoods, num_cell_types)
        stds_exp = stds[np.newaxis, :, :]  # (1, num_neighborhoods, num_cell_types)

        # Avoid division by zero
        stds_exp_safe = np.where(stds_exp == 0, 1e-10, stds_exp)

        # Calculate Gaussian PDF
        coeff = 1.0 / (stds_exp_safe * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((cell_data_exp - means_exp) / stds_exp_safe) ** 2
        pdf_vals = coeff * np.exp(exponent)

        # Handle std=0 cases explicitly
        zero_std_mask = stds_exp == 0
        equal_mask = cell_data_exp == means_exp
        pdf_vals = np.where(zero_std_mask & equal_mask, 1, pdf_vals)
        pdf_vals = np.where(zero_std_mask & (~equal_mask), 0, pdf_vals)

        # Product over cell types axis to get total probability per cell per neighborhood
        total_probs = np.prod(pdf_vals, axis=2)  # shape: (batch_size, num_neighborhoods)

        # Normalize probabilities per cell
        prob_sums = np.sum(total_probs, axis=1, keepdims=True)
        normalized_probs = total_probs / prob_sums

        # Return to CPU as numpy array
        return normalized_probs  # .get()

    # Process all batches and collect DataFrames
    results = []

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, num_cells)
        batch_df = windows2.iloc[start:end]

        batch_probs = compute_batch_probs(batch_df)
        batch_df_probs = pd.DataFrame(batch_probs, index=batch_df.index, columns=neighborhood_names)
        results.append(batch_df_probs)

        print(f"Processed batch {i + 1}/{num_batches}")

    # Concatenate all batches to get final result
    probabilities_df = pd.concat(results).sort_index()

    print(probabilities_df.head())

    filtered_cells = df[df["unique_region"] == desired_region]

    filtered_probabilities_df = probabilities_df.loc[filtered_cells.index]

    # Step 1: Retrieve assigned neighborhoods and probabilities for the filtered cells
    assigned_neighborhoods = filtered_cells["Tissue Unit"]

    assigned_probabilities = filtered_probabilities_df.reindex(filtered_cells.index).apply(
        lambda row: row[filtered_cells.loc[row.name, "Tissue Unit"]], axis=1
    )

    # Step 2: Create a new DataFrame with x, y, assigned neighborhood, probability, and region
    visualization_df = pd.DataFrame(
        {
            "x": filtered_cells["x"],
            "y": filtered_cells["y"],
            "assigned_tissueunit": assigned_neighborhoods,
            "assigned_probability": assigned_probabilities,
            "unique_region": filtered_cells["unique_region"],
        }
    )

    # Step 3: Filter for the desired unique_region
    filtered_region_df = visualization_df[visualization_df["unique_region"] == desired_region]

    # Step 4: Create the scatter plot without a colorbar
    plt.figure(figsize=(10, 10), dpi=50)
    plt.scatter(
        filtered_region_df["x"],
        filtered_region_df["y"],
        c=filtered_region_df["assigned_probability"],
        cmap="viridis",
        alpha=0.75,
        s=6,
    )

    # Add title
    plt.title(f"Assigned Probability\n({desired_region})", fontsize=35)

    # Hide axes
    plt.axis("off")

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

    return probabilities_df, visualization_df, filtered_region_df
