# compute_proportions.py
from typing import Optional, Union
import pandas as pd
import numpy as np
from anndata import AnnData
from .utils_adata import build_df_probs_from_adata, _fuzzy_col_matches


def compute_grouped_proportions(
    df_or_adata: Union[pd.DataFrame, AnnData],
    n1: str,
    n2: str,
    *,
    cell_type_col: str = "Cell Type",
    threshold: float = 0.25,
    prob_key: str = "neighborhood_probabilities",
) -> pd.DataFrame:

    if isinstance(df_or_adata, AnnData):
        df_probs, numeric_df = build_df_probs_from_adata(df_or_adata, prob_key=prob_key, cell_type_col=cell_type_col)
        df = df_probs
    else:
        df = df_or_adata.copy()
        numeric_df = df.select_dtypes(include=[np.number])

    # detect the cell_type column (allow fuzzy names)
    if cell_type_col not in df.columns:
        ct_candidates = [c for c in df.columns if c.lower().replace(" ", "") in ("celltype","celllabel","cell_label","cell_types")]
        if ct_candidates:
            cell_type_col_local = ct_candidates[0]
            df[cell_type_col] = df[cell_type_col_local].astype(str)
        else:
            raise KeyError(f"Couldn't detect a cell-type column in input. Expected {cell_type_col!r} or alternatives.")
    else:
        cell_type_col_local = cell_type_col

    # fuzzy match n1/n2 to probability column names (prefer numeric columns)
    def _find_column_for_name(name: str) -> Optional[str]:
        # first try direct in df
        if name in df.columns:
            return name
        # try numeric_df (if available)
        num_cols = list(numeric_df.columns) if hasattr(numeric_df, "columns") else []
        # try exact/fuzzy match among numeric columns
        m = _fuzzy_col_matches(num_cols, name)
        if m:
            return m
        # try fuzzy among all df columns
        return _fuzzy_col_matches(list(df.columns), name)

    col_n1 = _find_column_for_name(n1)
    col_n2 = _find_column_for_name(n2)

    if col_n1 is None or col_n2 is None:
        # if missing, attempt to detect label-based subsets (like 'Neighborhood' or 'subset'). If unavailable, error.
        label_cols = [c for c in df.columns if c.lower() in ("subset","neighborhood","label","assigned","cn","neighborhood_label")]
        if label_cols:
            label_col = label_cols[0]
            mask1 = df[label_col].astype(str) == n1
            mask2 = df[label_col].astype(str) == n2
            only_1 = df[mask1 & ~mask2]
            both = df[mask1 & mask2]
            only_2 = df[~mask1 & mask2]
        else:
            raise KeyError(f"Could not find probability columns for {n1!r} and/or {n2!r}. Tried columns: {list(df.columns)[:50]}")
    else:
        # coerce to numeric probabilities
        p1 = pd.to_numeric(df[col_n1], errors="coerce").fillna(0.0).astype(float)
        p2 = pd.to_numeric(df[col_n2], errors="coerce").fillna(0.0).astype(float)
        pos_1 = p1 > float(threshold)
        pos_2 = p2 > float(threshold)
        only_1 = df.loc[pos_1 & ~pos_2]
        both = df.loc[pos_1 & pos_2]
        only_2 = df.loc[~pos_1 & pos_2]

    def summarize(sub_df: pd.DataFrame, label: str) -> pd.DataFrame:
        if sub_df.empty:
            return pd.DataFrame({cell_type_col: [], "Proportion": [], "Subset": []})
        counts = sub_df[cell_type_col].astype(str).value_counts(normalize=True)
        out = counts.reset_index()
        out.columns = [cell_type_col, "Proportion"]
        out["Subset"] = label
        return out

    df1 = summarize(only_1, f"{n1} only")
    df2 = summarize(both, f"{n1} + {n2}")
    df3 = summarize(only_2, f"{n2} only")

    all_df = pd.concat([df1, df2, df3], ignore_index=True)
    if all_df.empty:
        # produce empty structured result with zero proportions for all detected cell types (best-effort)
        celltypes = sorted(df[cell_type_col].astype(str).unique())
        groups = [f"{n1} only", f"{n1} + {n2}", f"{n2} only"]
        full_index = pd.MultiIndex.from_product([celltypes, groups], names=[cell_type_col, "Subset"])
        result = pd.DataFrame(index=full_index).reset_index()
        result["Proportion"] = 0.0
        return result[[cell_type_col, "Subset", "Proportion"]]

    celltypes = np.unique(all_df[cell_type_col].values)
    groups = [f"{n1} only", f"{n1} + {n2}", f"{n2} only"]
    full_index = pd.MultiIndex.from_product([celltypes, groups], names=[cell_type_col, "Subset"])
    result = all_df.set_index([cell_type_col, "Subset"]).reindex(full_index, fill_value=0.0).reset_index()
    return result[[cell_type_col, "Subset", "Proportion"]]