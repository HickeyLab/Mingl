import numpy as np
import pandas as pd
import anndata as ad
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
import time

# =========================
# (A) Your Neighborhoods class, but AnnData/scverse-compatible wrapper
# =========================

class Neighborhoods(object):
    def __init__(self, cells,ks,cluster_col,sum_cols,keep_cols,neigh,X='X:X',Y = 'Y:Y',reg = 'Exp',add_dummies = True):
        self.cells_nodumz = cells
        self.X = X
        self.Y = Y
        self.reg = reg
        self.keep_cols = keep_cols
        self.sum_cols = sum_cols
        self.ks = ks
        self.cluster_col = cluster_col
        self.n_neighbors = max(ks)
        self.exps = list(self.cells_nodumz[self.reg].unique())
        self.bool_add_dummies = add_dummies
        self.neigh = neigh
        
    def add_dummies(self):
        
        c = self.cells_nodumz
        dumz = pd.get_dummies(c[self.cluster_col], dtype=int)
        keep = c[self.keep_cols]
        
        self.cells = pd.concat([keep,dumz],axis = 1)
        
        
        
    def get_tissue_chunks(self):
        self.tissue_group = self.cells[[self.X,self.Y,self.reg]].groupby(self.reg)
        
        tissue_chunks = [(time.time(),self.exps.index(t),t,a) for t,indices in self.tissue_group.groups.items() for a in np.array_split(indices,1)] 
        return tissue_chunks
    
    def make_windows(self,job):
        

        start_time,idx,tissue_name,indices = job
        job_start = time.time()

        print ("Starting:", str(idx+1)+'/'+str(len(self.exps)),': ' + self.exps[idx])

        tissue = self.tissue_group.get_group(tissue_name)
        to_fit = tissue.loc[indices][[self.X,self.Y]].values

        fit = NearestNeighbors(n_neighbors=self.n_neighbors+1).fit(tissue[[self.X,self.Y]].values)
        m = fit.kneighbors(to_fit)
        
        #don't include index cell in window (can always easily add to windows again later by just adding 1)
        m = m[0][:,1:], m[1][:,1:]


        #sort_neighbors
        args = m[0].argsort(axis = 1)
        add = np.arange(m[1].shape[0])*m[1].shape[1]
        sorted_indices = m[1].flatten()[args+add[:,None]]
        sorted_dists = m[0].flatten()[args+add[:,None]]
        neighbors = tissue.index.values[sorted_indices].astype(np.int32)
        end_time = time.time()

        print ("Finishing:", str(idx+1)+"/"+str(len(self.exps)),": "+ self.exps[idx],end_time-job_start,end_time-start_time)
        return neighbors,sorted_dists
    
    def k_windows(self,distance_max = 'none'):
        if self.bool_add_dummies:
            self.add_dummies()
        else:
            self.cells =self.cells_nodumz
        sum_cols = list(self.sum_cols)
        for col in sum_cols:
            if col in self.keep_cols:
                self.cells[col+'_sum'] = self.cells[col]
                self.sum_cols.remove(col)
                self.sum_cols+=[col+'_sum']

        values = self.cells[self.sum_cols].values
        tissue_chunks = self.get_tissue_chunks()
        tissues = [self.make_windows(job) for job in tissue_chunks]
        
        out_dict = {}
        if distance_max == 'none':
            k_dists = {k:np.inf for k in self.ks}
        else:
            k_dists = {k: distance_max for k in self.ks}

        print("k_dists:", k_dists, "Type:", type(k_dists))
        print("distance_max:", distance_max, "Type:", type(distance_max))

        for k in self.ks:
            for (neighbors,sorted_dists),job in zip(tissues,tissue_chunks):
                chunk = np.arange(len(neighbors))#indices
                tissue_name = job[2]
                indices = job[3]
                window = values[neighbors[chunk,:k].flatten()].reshape(len(chunk),k,len(self.sum_cols))

                mask = sorted_dists>k_dists[k]
                no_cells_masked = mask.sum(1)
                avg_cells_excluded = no_cells_masked[no_cells_masked!=0]
                if len(avg_cells_excluded)>0:
                    avg_cells_excluded = avg_cells_excluded.mean()
                else:
                    avg_cells_excluded = 0
                print ('{}:{}--{}/{} cells had cells excluded.  Avg excluded={}'.format(tissue_name,k,(no_cells_masked>0).sum(),len(no_cells_masked),avg_cells_excluded))

                mask = np.repeat(mask[:, :k, np.newaxis], len(sum_cols), axis=2)# don't sum cells that are outside max_distance
                masked_window = np.ma.array(window,mask = mask)
                summed = masked_window.sum(1).data

                out_dict[(tissue_name,k)] = (summed.astype(np.float16),indices)
        
        windows = {}
        for k in self.ks:

            window = pd.concat([pd.DataFrame(out_dict[(exp,k)][0],index = out_dict[(exp,k)][1].astype(int),columns = self.sum_cols) for exp in self.exps],axis=0)
            window = window.loc[self.cells.index.values]
            window = pd.concat([self.cells[self.keep_cols],window],axis=1)
            windows[k] = window
        return windows
    
def assign_probability_level_with_edges(series, n_bins=5, labels=None, use_quantiles=True):
    if labels is None:
        labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    labels = list(labels)

    s = series.dropna()
    vmin = float(s.min())
    vmax = float(s.max())

    if np.isclose(vmin, vmax):
        cat = pd.Series(["Medium"] * len(series), index=series.index)
        return cat.astype(pd.CategoricalDtype(categories=labels, ordered=True)), np.array([vmin, vmax])

    if use_quantiles:
        try:
            cat = pd.qcut(series, q=n_bins, labels=labels, duplicates="raise")
            edges = series.quantile(np.linspace(0, 1, n_bins + 1)).values
            return cat, edges
        except ValueError:
            pass

    edges = np.unique(np.linspace(vmin, vmax, n_bins + 1))
    labels_use = labels[: len(edges) - 1]
    cat = pd.cut(series, bins=edges, labels=labels_use, include_lowest=True)
    return cat, edges


def _col_score(lbl: str) -> float:
    import re
    s = str(lbl).strip().lower()
    if s == "very high" or "very high" in s:
        return 10000.0
    if s == "high" or s.endswith(" high") or s.startswith("high"):
        return 8000.0
    if s == "medium" or "medium" in s or s == "med":
        return 5000.0
    if s == "low" and "very" not in s:
        return 2000.0
    if s == "very low" or "very low" in s:
        return 100.0

    m = re.search(r"bin[_\s]*(\d+)", s)
    if m:
        return float(m.group(1))

    m2 = re.search(r"([0-9]*\.?[0-9]+)\s*[-–—]\s*([0-9]*\.?[0-9]+)", s)
    if m2:
        a = float(m2.group(1)); b = float(m2.group(2))
        return (a + b) / 2.0

    m3 = re.search(r"^\d*\.?\d+$", s)
    if m3:
        return float(s)

    m4 = re.search(r"(\d+)$", s)
    if m4:
        return float(m4.group(1))

    return -9999.0


def mingle_neighborhoods_scverse(
    adata: ad.AnnData,
    *,
    tu1="Inner Follicle",
    tu2="Outer Follicle",
    extra_cols=("x","y","unique_region","Neighborhood","Cell Type","Tissue Unit","Community"),
    # Neighborhoods() params
    ks=(10,20,30,50,100),
    k=20,
    distance_max=1000,
    # keys
    X="x",
    Y="y",
    reg="unique_region",
    neigh="Neighborhood",
    cluster_col="Probability_Level",
    celltype="Cell Type",
    tiss="Tissue Unit",
    comm="Community",
    # binning
    n_bins=5,
    bin_labels=("Very Low","Low","Medium","High","Very High"),
    use_quantiles=True,
    # clustering
    n_clusters=5,
    target_neighborhoods=("Inner Follicle","Outer Follicle"),
    random_state=0,
    # outputs
    out_score_key="Score",
    out_prob_level_key="Probability_Level",
    out_neighborhood_key=None,              # default becomes f"neighborhood{k}"
    out_prob_cluster_key="Probability_Bin_Cluster",
    store_windows_key="mingle_windows",
    store_fc_key="mingle_fc",
):
    """
    Wraps your exact logic into an AnnData-friendly function.

    Writes:
      adata.obs[out_score_key]
      adata.obs[out_prob_level_key]
      adata.obs[out_neighborhood_key]  (=-1 outside target neighborhoods)
      adata.obs[out_prob_cluster_key]  ("0".."n_clusters-1" for target rows else NaN)
    Stores:
      adata.uns[store_windows_key][k] = windows df for that k
      adata.uns[store_fc_key] = fc dataframe
      adata.uns["mingle_probability_edges"] = edges_used
    Returns:
      adata, df_sub (obs subset), fc, windows_k, edges_used
    """
    if out_neighborhood_key is None:
        out_neighborhood_key = f"neighborhood{k}"

    # ---- Build df2 exactly like you did, but from adata.obs
    obs = adata.obs.copy()

    needed = [tu1, tu2, *extra_cols]
    missing = [c for c in needed if c not in obs.columns]
    if missing:
        raise KeyError(f"Missing required adata.obs columns: {missing}")

    df2 = obs[[tu1, tu2, *extra_cols]].copy()

    eps = 1e-9
    df2["ratio"] = (df2[tu1].astype(float) + eps) / (df2[tu2].astype(float) + eps)
    df2["log_ratio"] = np.log(df2["ratio"])
    df2["max_prob"] = df2[[tu1, tu2]].astype(float).max(axis=1)
    df2[out_score_key] = df2["log_ratio"] * df2["max_prob"]

    df2[out_prob_level_key], edges_used = assign_probability_level_with_edges(
        df2[out_score_key], n_bins=n_bins, labels=list(bin_labels), use_quantiles=use_quantiles
    )

    # ---- Neighborhood windows (your class)
    df = df2.copy().reset_index(drop=True)

    sum_cols = list(df[out_prob_level_key].unique())
    keep_cols = [X, Y, reg, neigh, celltype, comm, tiss, out_prob_level_key]

    Neigh = Neighborhoods(
        df,
        ks=list(ks),
        cluster_col=out_prob_level_key,
        sum_cols=sum_cols,
        keep_cols=keep_cols,
        neigh=neigh,
        X=X, Y=Y, reg=reg,
        add_dummies=True,
    )
    cluster_name_windows = Neigh.k_windows(distance_max=distance_max)
    windows_k = cluster_name_windows[k].copy()

    # ---- dummies + sum_cols2 exactly like you did
    df = pd.concat([df, pd.get_dummies(df[out_prob_level_key], dtype=int)], axis=1)
    sum_cols2 = pd.get_dummies(df[out_prob_level_key], dtype=int).columns.tolist()

    # ---- align indices (same assumption as your code)
    if not windows_k.index.equals(df.index):
        if len(windows_k) == len(df):
            windows_k = windows_k.reset_index(drop=True)
            df = df.reset_index(drop=True)
        else:
            raise ValueError("windows_k and df indices differ and lengths differ; align them before filtering.")

    # ---- subset to target neighborhoods
    mask = df[neigh].isin(list(target_neighborhoods))
    windows_sub = windows_k.loc[mask].copy()
    df_sub = df.loc[mask].copy()

    # ---- cluster
    km = MiniBatchKMeans(n_clusters=int(n_clusters), random_state=random_state)

    if set(sum_cols2).issubset(windows_sub.columns):
        feature_matrix = windows_sub[sum_cols2].values
    else:
        feature_matrix = df_sub[sum_cols2].values

    labels_sub = km.fit_predict(feature_matrix)

    # ---- write labels back like you did
    df[out_neighborhood_key] = -1
    df.loc[mask, out_neighborhood_key] = labels_sub
    df_sub[out_neighborhood_key] = labels_sub

    # ---- fold-change
    niche_clusters = km.cluster_centers_
    values_sub = df_sub[sum_cols2].values
    tissue_avgs = values_sub.mean(axis=0)

    eps_fc = 1e-12
    tissue_row = tissue_avgs.reshape(1, -1)
    niche_plus = niche_clusters + tissue_row
    row_sums = niche_plus.sum(axis=1, keepdims=True)
    normed = niche_plus / (row_sums + eps_fc)
    fc_array = np.log2((normed + eps_fc) / (tissue_row + eps_fc))
    fc = pd.DataFrame(fc_array, columns=sum_cols2)

    # ---- order fc exactly like you did
    desired_col_order = sorted(list(fc.columns), key=lambda c: -_col_score(c))
    if len(set(_col_score(c) for c in fc.columns)) == 1:
        desired_col_order = list(sum_cols2[::-1])
    high_bin = desired_col_order[0]
    ordered_rows = fc[high_bin].sort_values(ascending=False).index.tolist()
    fc = fc.reindex(index=ordered_rows, columns=desired_col_order)

    # ---- Probability_Bin_Cluster map 0..n_clusters-1 to strings, same as your n_conversion_20
    conv = {i: str(i) for i in range(int(n_clusters))}
    df_sub[out_prob_cluster_key] = df_sub[out_neighborhood_key].map(conv)

    # ---- write results back into AnnData (match original row order)
    # df is reset_index(drop=True) so its order corresponds to adata.obs order if adata.obs was not permuted.
    # safest: align by position.
    adata.obs[out_score_key] = df2[out_score_key].values
    adata.obs[out_prob_level_key] = df2[out_prob_level_key].values
    adata.obs[out_neighborhood_key] = df[out_neighborhood_key].values

    # full-length Probability_Bin_Cluster: NaN for non-target rows
    full_prob_cluster = pd.Series([np.nan] * len(df), index=df.index, dtype="object")
    full_prob_cluster.loc[mask] = df_sub[out_prob_cluster_key].values
    adata.obs[out_prob_cluster_key] = full_prob_cluster.values

    # store artifacts
    adata.uns["mingle_probability_edges"] = edges_used
    adata.uns.setdefault(store_windows_key, {})
    (adata.uns[store_windows_key])[int(k)] = windows_k
    adata.uns[store_fc_key] = fc

    return adata, df_sub, fc, windows_k, edges_used



