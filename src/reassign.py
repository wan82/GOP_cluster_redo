"""
reassign.py —— minority reassignment (cluster correction).

Idea (from the original readme):
    If a cluster is mostly composed of one videoSequence but contains a few
    samples from another, move those minority samples into the cluster that
    holds the majority of THEIR videoSequence.

Implementation:
    1. Build a video -> best_cluster mapping using only non-noise samples;
       best_cluster = argmax_{c != -1} count(videoSequence == v, cluster == c).
    2. Relabel every non-noise sample via this mapping; noise samples stay -1.
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
    df_samples : DataFrame containing `video_col` and `cluster_col`
                 (usually the cluster_samples table)

    Returns
    -------
    df_out    : a copy of df_samples with `out_col` set to the reassigned cluster
    mapping   : video -> best_cluster dictionary
    n_changed : number of samples whose cluster label changed
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
