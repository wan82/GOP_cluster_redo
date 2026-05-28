"""
reduction.py —— 标准化 → PCA → (可选) UMAP。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import umap  # 顶层 import: 与原项目保持一致, 避免 numba JIT 预热时机不同导致 PCA/UMAP 数值漂移
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def standardize_and_pca(
    df_features: pd.DataFrame,
    var_ratio:   float = 0.95,
) -> Tuple[np.ndarray, PCA, StandardScaler]:
    """对 df_features 做 StandardScaler + PCA(保留 var_ratio 方差)。"""
    print(f"\n[PCA] input dim: {df_features.shape[1]}")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_features.values)

    pca   = PCA(n_components=var_ratio, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled)

    print(f"[PCA] kept dim: {X_pca.shape[1]} | "
          f"explained var: {pca.explained_variance_ratio_.sum():.4f}")
    return X_pca, pca, scaler


def attach_pca_to_keys(X_pca: np.ndarray, df_key: pd.DataFrame) -> pd.DataFrame:
    out = df_key.reset_index(drop=True).copy()
    for i in range(X_pca.shape[1]):
        out[f"pca_{i}"] = X_pca[:, i]
    return out


def umap_reduce(
    X:            np.ndarray,
    df_key:       pd.DataFrame,
    n_components: int   = 7,
    n_neighbors:  int   = 25,
    min_dist:     float = 0.2,
    random_state: int   = 42,
) -> Tuple[pd.DataFrame, object]:
    """对 X 做 UMAP，把坐标附到 df_key 上。"""
    assert len(X) == len(df_key), "X 与 df_key 行数不一致"

    reducer = umap.UMAP(
        n_components = n_components,
        n_neighbors  = n_neighbors,
        min_dist     = min_dist,
        metric       = "euclidean",
        random_state = random_state,
    )
    X_umap = reducer.fit_transform(np.asarray(X))

    out = df_key.reset_index(drop=True).copy()
    for i in range(n_components):
        out[f"umap_{i}"] = X_umap[:, i]

    print(f"[UMAP] output shape: {X_umap.shape}")
    return out, reducer
