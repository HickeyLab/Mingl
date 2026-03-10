from pathlib import Path
from typing import Union
import anndata as ad
import numpy as np
import pandas as pd

def read_file(path: Union[str, Path]) -> ad.AnnData:
    """
    Read a .csv or .h5ad file and return an AnnData object.

    For .csv files, the entire table is stored in ``adata.obs`` and ``adata.X``
    is an empty matrix (n_obs x 0). This works well for workflows that use
    only metadata / coordinates from `.obs` (e.g. KNN on x/y, neighborhoods).

    Parameters
    ----------
    path
        Path to the input file.

    Returns
    -------
    AnnData
        AnnData loaded from .h5ad, or constructed from .csv.

    Raises
    ------
    ValueError
        If the extension is not .csv or .h5ad.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)

        # X is an empty matrix; all info lives in obs
        X = np.zeros((df.shape[0], 0), dtype=np.float32)
        adata = ad.AnnData(X=X, obs=df.copy())
        return adata

    elif ext == ".h5ad":
        return ad.read_h5ad(path)
   
    else:
        raise ValueError(f"Unsupported file type: {ext}. Expected .csv or .h5ad")
    
    '''
    elif ext == ".h5mu":
        mdata = mu.read(path)
        mod = list(mdata.mod.keys())[0]
        adata = mdata.mod[mod]
        if "cellid" not in adata.obs.columns:
            adata.obs["cellid"] = adata.obs_names.astype(str)
        return adata
    '''