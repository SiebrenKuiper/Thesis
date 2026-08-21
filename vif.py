import pandas as pd
import numpy as np

def compute_vif(design_matrix, exclude_cols):
    if exclude_cols is None:
        exclude_cols = []
    intercept_candidates = [c for c in design_matrix.columns
                            if c in ("constant", "intercept")]
    exclude_cols = list(set(exclude_cols + intercept_candidates))
    target_cols = [c for c in design_matrix.columns if c not in exclude_cols]
    X_full = design_matrix.values.astype(float)
    vif_values = {}
    for col in target_cols:
        col_idx = design_matrix.columns.get_loc(col)
        y = X_full[:, col_idx]
        other_idx = [i for i in range(X_full.shape[1]) if i != col_idx]
        Xi = X_full[:, other_idx]
        b, _, _, _ = np.linalg.lstsq(Xi, y, rcond=None)
        fits = Xi @ b
        var_fits = np.var(fits, ddof=1)
        var_y    = np.var(y,    ddof=1)
        if var_y < 1e-12:
            vif_values[col] = np.nan
            continue
        r2 = min(var_fits / var_y, 0.9999999)
        vif_values[col] = 1.0 / (1.0 - r2)
    return pd.Series(vif_values, name="VIF")