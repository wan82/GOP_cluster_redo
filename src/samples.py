"""
samples.py —— 生成 cluster_samples.csv（中间件）。

形态：每行 = 输入 CSV 的一个原始 (videoSequence, baseQP, GOP_id) 行；
列 = 72 个 base 特征做完 StandardScaler 后的 z-score 值 + cluster + 3 个 key。
聚类标签按 (videoSequence, GOP_id) 从多视图聚类结果广播回来。
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
    df_raw        : 输入长表（每行 = 一个 (videoSequence, baseQP, GOP_id) 三元组）
    feature_cols  : 72 个 base 特征列名
    df_clustered  : 多视图聚类结果（必须含 join_keys + 'cluster'）
    join_keys     : 用来把 cluster 标签 join 回原表的键（通常 [videoSequence, GOP_id]）
    key_cols      : 输出表里要保留的 key 列（通常 [videoSequence, baseQP, GOP_id]）

    Returns
    -------
    DataFrame: 72 维 z-score + cluster + key_cols
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
