"""
zscore.py —— 簇语义分析:
    1. 计算每个 cluster 的多视图特征均值 (cluster_feature_mean.csv)
    2. 相对全局均值/std 做 z-score, 再按 |z| 排序取 Top-K (区分特征)
    3. 输出:
        cluster_feature_mean.csv
        cluster_feature_zscore_full.csv
        cluster_{c}_top{K}_zscore_features.csv (含 noise → noise_top{K}…)
        cluster_zscore_topk_summary.txt
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def analyze_zscore(
    df_mv:        pd.DataFrame,
    df_mv_key:    pd.DataFrame,
    df_clustered: pd.DataFrame,
    join_keys:    list,
    top_k:        int,
    out_dir:      str,
    fn_feature_mean:    str = "cluster_feature_mean.csv",
    fn_zscore_full:     str = "cluster_feature_zscore_full.csv",
    fn_summary_txt:     str = "cluster_zscore_topk_summary.txt",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
    """
    Returns
    -------
    df_cluster_mean : 每簇 × 特征 均值
    df_zscore       : 每簇 × 特征 z-score
    cluster_topk    : {cluster_id: top-K DataFrame}
    """
    os.makedirs(out_dir, exist_ok=True)

    df_full = (
        df_mv.reset_index(drop=True)
              .join(df_mv_key.reset_index(drop=True))
              .merge(df_clustered[join_keys + ["cluster"]], on=join_keys, how="inner")
    )

    feature_cols = df_mv.columns.tolist()

    global_mean = df_full[feature_cols].mean()
    global_std  = df_full[feature_cols].std(ddof=0)
    global_std[global_std == 0] = 1e-6

    df_cluster_mean = df_full.groupby("cluster")[feature_cols].mean()
    df_zscore       = (df_cluster_mean - global_mean) / global_std
    df_cluster_mean.index.name = "cluster"
    df_zscore.index.name       = "cluster"

    # 落盘
    df_cluster_mean.to_csv(os.path.join(out_dir, fn_feature_mean))
    df_zscore.to_csv(os.path.join(out_dir, fn_zscore_full))

    cluster_topk: dict = {}
    summary_path = os.path.join(out_dir, fn_summary_txt)
    with open(summary_path, "w", encoding="utf-8") as f:
        for c in sorted(df_zscore.index):
            z = df_zscore.loc[c]
            top_idx = z.abs().sort_values(ascending=False).head(top_k).index
            df_top = (
                pd.DataFrame({"feature": top_idx, "z_score": z[top_idx].values})
                  .sort_values("z_score", key=lambda x: x.abs(), ascending=False)
            )
            cluster_topk[int(c)] = df_top

            tag = "noise" if c == -1 else f"cluster_{c}"
            df_top.to_csv(
                os.path.join(out_dir, f"{tag}_top{top_k}_zscore_features.csv"),
                index=False,
            )

            f.write("=" * 30 + "\n")
            label = "Cluster -1 (NOISE)" if c == -1 else f"Cluster {c}"
            f.write(f"{label}\n")
            f.write("=" * 30 + "\n")
            f.write(f"Top-{top_k} Z-score features:\n")
            for i, row in enumerate(df_top.itertuples(index=False), 1):
                f.write(f"  {i:2d}. {row.feature:<40s} z = {row.z_score:+.3f}\n")
            f.write("\n")

    print(f"[Z-score] saved -> {summary_path}")
    return df_cluster_mean, df_zscore, cluster_topk
