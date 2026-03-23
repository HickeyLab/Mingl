# utils_adata.py
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from anndata import AnnData


def _is_prob_like_series(s: pd.Series) -> bool:
    """
    Heuristic: numeric series, many distinct values, and values mostly between 0 and 1.
    """
    if not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_float_dtype(s)):
        return False
    vals = s.dropna().astype(float)
    if vals.size < 3:
        return False
    pct_in_0_1 = ((vals >= 0.0) & (vals <= 1.0)).mean()
    distinct = vals.nunique()
    return (distinct >= 3) and (pct_in_0_1 >= 0.8)


def _fuzzy_col_matches(cols: List[str], name: str) -> Optional[str]:
    """
    Return a single fuzzy-matched column name from cols that best matches `name`,
    using lower/underscore normalization and substring testing (not strict).
    """
    lowname = name.lower().replace(" ", "_")
    for c in cols:
        if lowname == c.lower().replace(" ", "_"):
            return c
    for c in cols:
        if lowname in c.lower().replace(" ", "_"):
            return c
    # try tokens in name
    for c in cols:
        cl = c.lower()
        if ("prob" in cl or "probability" in cl or "%" in cl) and any(tok in cl for tok in name.lower().split()):
            return c
    return None


def build_df_probs_from_adata(
    adata: AnnData,
    prob_key: Optional[str] = "neighborhood_probabilities",
    cell_type_col: str = "Cell Type",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (df_probs, numeric_probs_df) from an AnnData:
      - prefer adata.obsm[prob_key] if present (DataFrame or array)
      - else search adata.obsm for DataFrame-like entries with probability-like columns
      - else search adata.obs for columns that look like probability columns (name contains 'prob' or numeric values in [0,1])
      - also try adata.uns["neighborhood_probability_neighborhoods"] for column names when an array is found

    Returns:
      - df_probs: probabilities DataFrame (index = adata.obs_names) with the detected probability columns and an attached cell_type_col (if found)
      - numeric_df: same as df_probs but only numeric probability columns (floats)
    """
    # 1) try explicit obsm key
    df_probs: Optional[pd.DataFrame] = None

    if prob_key and prob_key in adata.obsm:
        obj = adata.obsm[prob_key]
        if isinstance(obj, pd.DataFrame):
            df_probs = obj.reindex(adata.obs_names).copy()
        else:
            arr = np.asarray(obj)
            if arr.shape[0] == adata.n_obs:
                nb_names = adata.uns.get("neighborhood_probability_neighborhoods", None)
                if nb_names is not None and len(nb_names) == arr.shape[1]:
                    cols = list(nb_names)
                else:
                    cols = [f"N{i}" for i in range(arr.shape[1])]
                df_probs = pd.DataFrame(arr, index=adata.obs_names, columns=cols)

    # 2) if not found, scan other adata.obsm entries for DataFrame-like probability matrices
    if df_probs is None:
        for k, v in adata.obsm.items():
            if isinstance(v, pd.DataFrame):
                # prefer frames that contain 'prob' in any column
                if any(("prob" in str(c).lower() or "probability" in str(c).lower() or "%" in str(c)) for c in v.columns):
                    df_probs = v.reindex(adata.obs_names).copy()
                    break
                # or if columns look numeric and in [0,1]
                numeric_cols = [c for c in v.columns if _is_prob_like_series(v[c])]
                if numeric_cols:
                    df_probs = v.reindex(adata.obs_names)[numeric_cols].copy()
                    break
            else:
                # array-like: prefer arrays with shape (n_obs, n_neigh) and if adata.uns has names
                try:
                    arr = np.asarray(v)
                    if arr.ndim == 2 and arr.shape[0] == adata.n_obs:
                        nb_names = adata.uns.get("neighborhood_probability_neighborhoods", None)
                        if nb_names is not None and len(nb_names) == arr.shape[1]:
                            cols = list(nb_names)
                        else:
                            cols = [f"{k}_{i}" for i in range(arr.shape[1])]
                        df_probs = pd.DataFrame(arr, index=adata.obs_names, columns=cols)
                        break
                except Exception:
                    continue

    # 3) fallback: search adata.obs for prob-like columns
    if df_probs is None:
        candidate_cols = []
        for c in adata.obs.columns:
            cl = str(c).lower()
            if ("prob" in cl) or ("probability" in cl) or ("%" in cl):
                candidate_cols.append(c)
        if not candidate_cols:
            # try numeric columns that look like probabilities
            for c in adata.obs.columns:
                try:
                    s = adata.obs[c]
                    if _is_prob_like_series(pd.Series(s.values)):
                        candidate_cols.append(c)
                except Exception:
                    continue
        if candidate_cols:
            df_probs = adata.obs[candidate_cols].reindex(adata.obs_names).copy()

    # if still nothing, raise an informative KeyError
    if df_probs is None:
        raise KeyError(
            f"No probability matrix found in adata.obsm[{prob_key!r}] or other obsm/obs candidates. "
            "Try computing probabilities into adata.obsm or pass a DataFrame to the plotting utilities."
        )

    # ensure index aligns and is unique
    df_probs.index = df_probs.index.astype(str)
    df_probs = df_probs.reindex(adata.obs_names).copy()

    # attach/normalize cell type column
    if cell_type_col in adata.obs.columns:
        df_probs[cell_type_col] = adata.obs[cell_type_col].reindex(df_probs.index).astype(str)
    else:
        # fuzzy fallbacks
        alts = [c for c in adata.obs.columns if c.lower().replace(" ", "") in ("celltype","celllabel","cell_types","cell_label")]
        if alts:
            df_probs[cell_type_col] = adata.obs[alts[0]].reindex(df_probs.index).astype(str)

    # Coerce prob columns to numeric: assume everything except cell_type_col is a probability column
    prob_cols = [c for c in df_probs.columns if c != cell_type_col]
    numeric_df = df_probs[prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

    return df_probs, numeric_df