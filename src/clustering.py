"""
clustering.py —— HDBSCAN clustering (keeps prediction_data for soft clustering).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def hdbscan_cluster(
    df_embed: pd.DataFrame,
    embed_cols: List[str],
    min_cluster_size: int = 8,
    min_samples:      int = 8,
    cluster_selection_method: str = "eom",
) -> Tuple[pd.DataFrame, object]:
    """
    Run HDBSCAN on df_embed[embed_cols].

    `embed_cols` can be either UMAP columns or PCA columns, depending on whether
    UMAP is enabled upstream.
    """
    import hdbscan

    X = df_embed[embed_cols].values
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = min_cluster_size,
        min_samples      = min_samples,
        metric           = "euclidean",
        cluster_selection_method = cluster_selection_method,
        prediction_data  = True,
    )
    labels = clusterer.fit_predict(X)

    out = df_embed.copy()
    out["cluster"]       = labels
    out["confidence"]    = getattr(clusterer, "probabilities_",  np.ones(len(labels)))
    out["outlier_score"] = getattr(clusterer, "outlier_scores_", np.full(len(labels), np.nan))

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    print(f"\n[HDBSCAN] params = (mcs={min_cluster_size}, ms={min_samples}, "
          f"method={cluster_selection_method})")
    print(f"[HDBSCAN] clusters: {n_clusters} | "
          f"noise: {n_noise}/{len(labels)} ({n_noise/len(labels):.1%})")
    return out, clusterer
