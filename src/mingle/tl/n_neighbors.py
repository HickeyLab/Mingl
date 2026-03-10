
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import norm
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import anndata as ad

def run_mingle_over_n_clusters(
    adata,
    knn_feature_cols,
    n_range=range(1, 51),
    output_col_template="Neighborhood_{}",
    return_per_cell=True,
    plot_summary=True,
    *,
    x_key="x",
    y_key="y",
    region_key="unique_region",
    results_uns_key="mingle_n_clusters",
):
    for k in (x_key, y_key, region_key):
        if k not in adata.obs.columns:
            raise KeyError(f"adata.obs missing required key '{k}'")

    df = adata.obs.copy()
    df[knn_feature_cols] = df[knn_feature_cols].astype("float32")

    per_cell_df = df[[x_key, y_key, region_key]].copy() if return_per_cell else None
    summary_rows = []

    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(df[knn_feature_cols].values)

    for n in tqdm(n_range, desc="Running KMeans + MINGLE (CPU)"):
        km = MiniBatchKMeans(n_clusters=n, random_state=0)
        labels = km.fit_predict(X_scaled_all)
        cluster_col = output_col_template.format(n)
        df[cluster_col] = labels.astype(str)

        # Centroid computation (same)
        centroids = []
        for label in map(str, range(n)):
            group = df[df[cluster_col] == label][knn_feature_cols]
            centroids.append(
                {
                    "Neighborhood": label,
                    **{f"{c}_mean": group[c].mean() for c in knn_feature_cols},
                    **{f"{c}_std": group[c].std(ddof=0) for c in knn_feature_cols},
                }
            )
        df_centroids = pd.DataFrame(centroids).set_index("Neighborhood")

        means = np.array(
            df_centroids[[f"{c}_mean" for c in knn_feature_cols]].values, dtype=np.float32
        )
        stds = np.array(
            df_centroids[[f"{c}_std" for c in knn_feature_cols]].values, dtype=np.float32
        )
        stds = np.where(stds < 1e-2, 1e-2, stds)

        label_map = {name: i for i, name in enumerate(df_centroids.index)}
        assigned_labels = df[cluster_col].values
        assigned_indices = [label_map[label] for label in assigned_labels]

        batch_size = 20000
        num_cells = len(df)
        num_batches = (num_cells + batch_size - 1) // batch_size
        log_likelihoods = []
        assigned_probs = []

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_cells)
            batch_df = df.iloc[start:end]

            batch_data = np.array(batch_df[knn_feature_cols].values, dtype=np.float32)
            batch_indices = np.array(assigned_indices[start:end], dtype=np.int32)

            batch_means = means[batch_indices]
            batch_stds = stds[batch_indices]

            log_coeff = -np.log(batch_stds) - 0.5 * np.log(2 * np.pi)
            squared_diff = ((batch_data - batch_means) / batch_stds) ** 2
            log_pdf_vals = log_coeff - 0.5 * squared_diff
            log_probs = np.sum(log_pdf_vals, axis=1)
            log_likelihoods.extend(log_probs.tolist())

            coeffs = 1.0 / (stds[np.newaxis, :, :] * np.sqrt(2 * np.pi))
            exponents = (
                -0.5
                * ((batch_data[:, np.newaxis, :] - means[np.newaxis, :, :]) / stds[np.newaxis, :, :])
                ** 2
            )
            pdf_vals = coeffs * np.exp(exponents)

            total_probs = np.prod(pdf_vals, axis=2)
            total_probs = np.nan_to_num(total_probs, nan=1e-300, posinf=1e-300, neginf=1e-300)

            prob_sums = np.sum(total_probs, axis=1, keepdims=True)
            prob_sums = np.where(prob_sums == 0, 1e-300, prob_sums)

            normalized_probs = total_probs / prob_sums
            normalized_probs = np.nan_to_num(normalized_probs, nan=0.0, posinf=0.0, neginf=0.0)

            row_indices = np.arange(end - start)
            batch_probs = normalized_probs[row_indices, batch_indices]
            assigned_probs.extend(batch_probs.tolist())

        avg_log = float(np.mean(log_likelihoods))
        avg_prob = float(np.mean(assigned_probs))
        summary_rows.append(
            {"n_clusters": n, "avg_log_likelihood": avg_log, "avg_assigned_probability": avg_prob}
        )

        adata.obs[cluster_col] = pd.Categorical(labels.astype(str))
        adata.obs[f"log_likelihood_n{n}"] = np.asarray(log_likelihoods, dtype=np.float32)
        adata.obs[f"assigned_prob_n{n}"] = np.asarray(assigned_probs, dtype=np.float32)

        if return_per_cell:
            per_cell_df[f"log_likelihood_n{n}"] = log_likelihoods
            per_cell_df[f"assigned_prob_n{n}"] = assigned_probs
            per_cell_df[cluster_col] = labels

    summary_df = pd.DataFrame(summary_rows)
    adata.uns[results_uns_key] = {"summary_df": summary_df, "knn_feature_cols": list(knn_feature_cols)}

    if plot_summary:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel("Number of Neighborhoods")
        ax1.set_ylabel("Avg Log-Likelihood", color="tab:blue")
        ax1.plot(summary_df["n_clusters"], summary_df["avg_log_likelihood"], color="tab:blue", marker="o")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Avg Assigned Probability", color="tab:green")
        ax2.plot(summary_df["n_clusters"], summary_df["avg_assigned_probability"], color="tab:green", marker="s")
        ax2.tick_params(axis="y", labelcolor="tab:green")

        plt.title("Log-Likelihood and Assigned Probability vs. Neighborhood Count")
        fig.tight_layout()
        plt.show()

    return (summary_df, per_cell_df) if return_per_cell else summary_df

def find_elbow_point(
    y_values,
    x_values=None,
    threshold=0.01,
    window=9,
    polyorder=3,
    *,
    adata=None,
    y_key=None,
    x_key=None,
    uns_key=None,
):
    """
      - If adata is provided, y_values/x_values can be pulled from adata.uns[uns_key]
        (a DataFrame) using y_key/x_key.
      - Otherwise, behaves exactly like before on arrays/Series.

    Returns: (elbow_idx, x_at_elbow, slope)
    """
    import numpy as np
    from scipy.signal import savgol_filter

    if adata is not None:
        if uns_key is None or y_key is None:
            raise ValueError("If adata is provided, uns_key and y_key must be provided.")
        summary_df = adata.uns[uns_key]
        # allow nested dict payloads like adata.uns[uns_key]['summary_df']
        if isinstance(summary_df, dict):
            if "summary_df" not in summary_df:
                raise KeyError(f"adata.uns['{uns_key}'] is a dict but has no 'summary_df'")
            summary_df = summary_df["summary_df"]
        y_values = summary_df[y_key]
        if x_key is not None:
            x_values = summary_df[x_key]

    y = np.asarray(y_values)
    x = np.arange(1, len(y) + 1) if x_values is None else np.asarray(x_values)

    smoothed = savgol_filter(y, window_length=window, polyorder=polyorder)
    slope = np.gradient(smoothed, x)

    flat_idx = np.where(np.abs(slope) < threshold)[0]
    elbow_idx = flat_idx[0] if len(flat_idx) > 0 else np.argmin(slope)

    return elbow_idx, x[elbow_idx], slope

def find_best_unsupervised_plateau(
    log_likelihoods,
    assigned_probs,
    method="harmonic",
    slope_threshold=0.025,
    score_threshold=0.88,
    window_length=9,
    polyorder=3,
    elbow_min=None,
    elbow_max=None,
    *,
    adata=None,
    uns_key=None,
    ll_key=None,
    prob_key=None,
    out_uns_key=None,
):
    """
      - If adata is provided, log_likelihoods/assigned_probs can be pulled from
        adata.uns[uns_key] (DataFrame) using ll_key/prob_key.
      - If out_uns_key is provided, stores composite_df, best_n, ranked_plateaus
        into adata.uns[out_uns_key] (no effect on computations).

    Returns: (composite_df, best_n, ranked_plateaus)
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from scipy.signal import savgol_filter
    from itertools import groupby
    from operator import itemgetter

    if adata is not None:
        if uns_key is None or ll_key is None or prob_key is None:
            raise ValueError("If adata is provided, uns_key, ll_key, and prob_key must be provided.")
        summary_df = adata.uns[uns_key]
        if isinstance(summary_df, dict):
            if "summary_df" not in summary_df:
                raise KeyError(f"adata.uns['{uns_key}'] is a dict but has no 'summary_df'")
            summary_df = summary_df["summary_df"]
        log_likelihoods = summary_df[ll_key]
        assigned_probs = summary_df[prob_key]

    log_y = np.asarray(log_likelihoods).reshape(-1, 1)
    prob_y = np.asarray(assigned_probs).reshape(-1, 1)
    n_clusters = np.arange(1, len(log_y) + 1)

    scaler = MinMaxScaler()
    norm_log = scaler.fit_transform(log_y).flatten()
    norm_prob = scaler.fit_transform(prob_y).flatten()

    if method == "harmonic":
        composite = 2 * (norm_log * norm_prob) / (norm_log + norm_prob + 1e-8)
    elif method == "weighted":
        composite = 0.5 * norm_log + 0.5 * norm_prob
    else:
        raise ValueError("Invalid method")

    composite_smooth = savgol_filter(composite, window_length, polyorder)
    slope = np.gradient(composite_smooth)
    max_score = composite_smooth.max()

    score_mask = composite_smooth >= (score_threshold * max_score)
    slope_mask = np.abs(slope) < slope_threshold
    plateau_mask = score_mask & slope_mask

    if elbow_min is not None and elbow_max is not None:
        elbow_mask = (n_clusters > elbow_min) & (n_clusters < elbow_max)
        plateau_mask = plateau_mask & elbow_mask

    runs = []
    for k, g in groupby(enumerate(plateau_mask), key=lambda x: x[1]):
        if k:
            idxs = list(map(itemgetter(0), g))
            run = {
                "start_idx": idxs[0],
                "length": len(idxs),
                "mean_score": float(np.mean(composite_smooth[idxs])),
                "mean_slope": float(np.mean(np.abs(slope[idxs]))),
                "idxs": idxs,
            }
            runs.append(run)

    if not runs:
        best_idx = int(np.argmax(composite_smooth))
    else:
        runs.sort(
            key=lambda r: (
                r["mean_slope"],
                -r["mean_score"],
                -r["length"],
                -r["start_idx"],
            )
        )
        best_idx = runs[0]["start_idx"]

    best_n = int(n_clusters[best_idx])

    composite_df = pd.DataFrame(
        {
            "n_clusters": n_clusters,
            "norm_log_likelihood": norm_log,
            "norm_assigned_probability": norm_prob,
            "composite_score": composite,
            "composite_smooth": composite_smooth,
            "composite_slope": slope,
        }
    )

    ranked_plateaus = pd.DataFrame(runs)
    if not ranked_plateaus.empty:
        ranked_plateaus["start_n"] = [int(n_clusters[r["start_idx"]]) for r in runs]
        ranked_plateaus["rank"] = np.arange(1, len(ranked_plateaus) + 1)
        ranked_plateaus = ranked_plateaus[
            ["rank", "start_n", "length", "mean_score", "mean_slope", "start_idx"]
        ]
    else:
        ranked_plateaus = pd.DataFrame(
            columns=["rank", "start_n", "length", "mean_score", "mean_slope", "start_idx"]
        )

    print(f"📍 Best plateau starts at n = {best_n} (score = {composite_smooth[best_idx]:.4f})")
    if adata is not None and out_uns_key is not None:
        adata.uns[out_uns_key] = {
            "composite_df": composite_df,
            "best_n": best_n,
            "ranked_plateaus": ranked_plateaus,
        }

    return composite_df, best_n, ranked_plateaus

def plot_stable_composite(df, best_n, ll_n=None, prob_n=None):
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 25})

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    h1, = ax.plot(
        df["n_clusters"],
        df["composite_score"],
        label="Composite Score",
        color="black",
        marker="o",
        linewidth=3,
        markersize=10,
    )
    h2, = ax.plot(
        df["n_clusters"],
        df["composite_slope"],
        label="Slope",
        linestyle="--",
        color="gray",
        linewidth=3,
    )

    h3 = ax.axvline(best_n, color="red", linestyle=":", label=f"Selected n = {best_n}", linewidth=5)

    h4 = None
    h5 = None
    if ll_n is not None:
        h4 = ax.axvline(ll_n, color="blue", linestyle="--", label=f"Log Likelihood Elbow n = {ll_n}", linewidth=5)
    if prob_n is not None:
        h5 = ax.axvline(prob_n, color="orange", linestyle="--", label=f"Probability Elbow n = {prob_n}", linewidth=5)

    ax.set_xlabel("Number of Neighborhoods", fontsize=25)
    ax.set_ylabel("", fontsize=35)
    ax.tick_params(axis="both", which="major", labelsize=25)
    ax.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [h1, h2, h3]
    labels = ["Composite Score", "Slope", f"Selected n = {best_n}"]
    if h4 is not None:
        handles.append(h4)
        labels.append(f"Log Likelihood Elbow n = {ll_n}")
    if h5 is not None:
        handles.append(h5)
        labels.append(f"Probability Elbow n = {prob_n}")

    plt.tight_layout()
    plt.show()

    legend_fig = plt.figure(figsize=(8, 2.5), dpi=300)
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, labels=labels, loc="center", frameon=False, fontsize=28, ncol=1)
    for spine in legend_ax.spines.values():
        spine.set_visible(False)

    legend_fig.tight_layout()
    plt.show()

    return fig, legend_fig

'''
file_path = r"/Volumes/data/MINGLE/Data/Intestine/intestine_all_information_2.csv"#r"/Volumes/data/MINGLE/Data/Intestine/05_25_huBMAP_tunit.csv"
#cells = mg.pp.read_file(file_path)

adata = read_file(file_path)

X = "x"
Y = "y"
reg = "unique_region"
cluster_col = "Cell Type"

sum_cols = list(adata.obs[cluster_col].unique())
keep_cols = [X, Y, reg, cluster_col]

windows = KNN2(adata, cluster_col=cluster_col, keep_obs_cols=keep_cols)
k = 10
windows2 = windows[k]
windows2[cluster_col] = adata.obs[cluster_col].values

adata_windows = ad.AnnData(X=None, obs=windows2.copy())
adata_windows.obs_names = adata_windows.obs.index.astype(str)

adata_windows.obsm[f"knn_windows_k{k}"] = adata_windows.obs[sum_cols].astype("float32").values
adata_windows.uns["knn_windows"] = {
    "k": k,
    "cols": list(sum_cols),
    "source": "dummy-count neighborhood windows",
}

summary_df, per_cell_df = run_mingle_over_n_clusters(
    adata=adata_windows,          # <-- AnnData in, AnnData updated in-place
    knn_feature_cols=sum_cols,    # same as your call
    n_range=range(1, 51),
    return_per_cell=True,
    plot_summary=True,
    x_key="x",
    y_key="y",
    region_key="unique_region",
)

ll_idx, ll_n, _ = find_elbow_point(
    y_values=None,
    x_values=None,
    adata=adata_windows,
    uns_key="mingle_n_clusters",
    y_key="avg_log_likelihood",
    x_key="n_clusters",
)

prob_idx, prob_n, _ = find_elbow_point(
    y_values=None,
    x_values=None,
    adata=adata_windows,
    uns_key="mingle_n_clusters",
    y_key="avg_assigned_probability",
    x_key="n_clusters",
)

# Step 2: Constrained plateau search (pulled from adata.uns)
composite_df, best_n, ranked_plateaus = find_best_unsupervised_plateau(
    log_likelihoods=None,
    assigned_probs=None,
    elbow_min=min(ll_n, prob_n),
    elbow_max=max(ll_n, prob_n),
    adata=adata_windows,
    uns_key="mingle_n_clusters",
    ll_key="avg_log_likelihood",
    prob_key="avg_assigned_probability",
    out_uns_key="mingle_plateau_selection",  # optional; remove if you don't want storage
)

# Step 3: Plot (unchanged)
plot_stable_composite(composite_df, best_n, ll_n, prob_n)

# Step 4: View ranked plateau table
print(ranked_plateaus)

'''