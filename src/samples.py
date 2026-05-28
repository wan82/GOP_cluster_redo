"""
samples.py —— produce cluster_samples.csv (the intermediate artefact).

Shape: one row per input-CSV (videoSequence, baseQP, GOP_id) triple;
columns = 72 base features z-scored via StandardScaler + cluster + 3 key columns.
Cluster labels are broadcast back from the multi-view clustering result by
joining on (videoSequence, GOP_id).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_cluster_samples(
    df_raw:       pd.DataFrame,
    feature_cols: List[str],
    df_clustered: pd.DataFrame,
    join_keys:    List[str],
    key_cols:     List[str],
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df_raw        : input long-format table (one row per (videoSequence, baseQP, GOP_id) triple)
    feature_cols  : the 72 base feature column names
    df_clustered  : multi-view clustering result (must contain `join_keys` and 'cluster')
    join_keys     : columns used to join cluster labels back to the long table
                    (typically [videoSequence, GOP_id])
    key_cols      : key columns to retain in the output table
                    (typically [videoSequence, baseQP, GOP_id])

    Returns
    -------
    DataFrame: 72 z-scored features + cluster + key_cols
    """
    # 1) 对原始 72 维特征做 z-score 标准化（与原版 cluster_samples.csv 对齐）
    scaler = StandardScaler()
    X = df_raw[feature_cols].values
    X_z = scaler.fit_transform(X)
    df_z = pd.DataFrame(X_z, columns=feature_cols, index=df_raw.index)

    # 2) 拼回 key 列
    df_z = pd.concat([df_z, df_raw[key_cols].reset_index(drop=True)], axis=1)

    # 3) 把多视图聚类的 cluster 标签按 join_keys join 回来
    df_z = df_z.merge(
        df_clustered[join_keys + ["cluster"]],
        on=join_keys,
        how="left",
    )
    # 缺失 cluster（即聚类阶段被丢弃的 GOP）用 -1 表示
    df_z["cluster"] = df_z["cluster"].fillna(-1).astype(int)

    # 4) 列顺序：先特征，再 cluster，再 key
    ordered = feature_cols + ["cluster"] + key_cols
    return df_z[ordered]
