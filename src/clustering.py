"""
clustering.py —— HDBSCAN 聚类（保留 prediction_data 以便软聚类）。
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
    在 df_embed[embed_cols] 上跑 HDBSCAN。
    embed_cols 既可以是 UMAP 列也可以是 PCA 列 —— 取决于是否启用 UMAP。
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
