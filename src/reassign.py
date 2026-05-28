"""
reassign.py —— 少数派归并 (cluster correction)。

思路（来自 readme）:
    若某一 cluster 内部多数来自某个 videoSequence，少数来自另一个，
    则将少数样本并入"它的 videoSequence 在哪个 cluster 占多数"的那个 cluster。

实现:
    1. 仅在非噪声样本里建 video → best_cluster 映射;
       best_cluster = argmax_{c != -1} count(videoSequence==v, cluster==c)
    2. 非噪声样本依此映射重新打标; 噪声样本保持 cluster = -1。
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def reassign_minority(
    df_samples: pd.DataFrame,
    video_col:  str = "videoSequence",
    cluster_col: str = "cluster",
    out_col:    str = "cluster",
) -> Tuple[pd.DataFrame, Dict[str, int], int]:
    """
    Parameters
    ----------
    df_samples : 含 video_col 与 cluster_col 的 DataFrame（通常是 cluster_samples）

    Returns
    -------
    df_out  : 复制后并把 out_col 改成归并后 cluster 的 DataFrame
    mapping : video → best_cluster 字典
    n_changed : 被改动 cluster 的样本数
    """
    df = df_samples.copy()

    valid = df[df[cluster_col] != -1]
    if valid.empty:
        return df, {}, 0

    # video × cluster 频次表
    counts = valid.groupby([video_col, cluster_col]).size().reset_index(name="n")
    # 每个 video 选 n 最大的 cluster；并列时取 cluster id 较小的
    counts = counts.sort_values([video_col, "n", cluster_col],
                                ascending=[True, False, True])
    mapping = (
        counts.drop_duplicates(subset=[video_col], keep="first")
              .set_index(video_col)[cluster_col]
              .to_dict()
    )

    new_labels = df[cluster_col].copy()
    mask_nonnoise = df[cluster_col] != -1
    mapped = df.loc[mask_nonnoise, video_col].map(mapping)
    # 万一某 video 全部是噪声（不可能但兜底）保持原值
    mapped = mapped.fillna(df.loc[mask_nonnoise, cluster_col]).astype(int)
    new_labels.loc[mask_nonnoise] = mapped

    n_changed = int((new_labels != df[cluster_col]).sum())
    df[out_col] = new_labels.astype(int)
    return df, mapping, n_changed
