"""
features.py —— 多视图特征聚合。
按 (videoSequence, GOP_id) 把 4 个 QP 下的特征向量横向拼接，
得到形状 (n_gops, 72*len(qp_list)) 的多视图矩阵。

输出列名风格: {base_feature}_qp{qp}, 例如 mean_bits_qp22。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def build_multiview_features(
    df:           pd.DataFrame,
    feature_cols: List[str],
    join_keys:    Tuple[str, ...] = ("videoSequence", "GOP_id"),
    qp_col:       str             = "baseQP",
    qp_list:      Tuple[int, ...] = (22, 27, 32, 37),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameters
    ----------
    df            : 长表（每行 = 一个 (videoSequence, baseQP, GOP_id) 三元组）
    feature_cols  : 72 个 base 特征列名
    join_keys     : 唯一标识一个 GOP 的列
    qp_col        : QP 所在列
    qp_list       : 所有要求齐全的 QP 值；任何一个 QP 缺失会整组丢弃

    Returns
    -------
    df_mv     : 多视图特征 DataFrame, shape = (n_valid_gops, 72*|qp_list|)
    df_mv_key : 与 df_mv 同行对应的 key 列 DataFrame
    """
    records:     list = []
    key_records: list = []
    skipped = 0

    for key_vals, group in df.groupby(list(join_keys)):
        qp_values_present = set(group[qp_col].values)
        if not all(qp in qp_values_present for qp in qp_list):
            skipped += 1
            continue

        feature_list = []
        valid = True
        for qp in qp_list:
            sub = group[group[qp_col] == qp]
            if len(sub) != 1:
                print(f"[WARN] skip key={key_vals}, qp={qp}, rows={len(sub)} (expected 1)")
                valid = False
                break
            feature_list.append(sub.iloc[0][feature_cols].values)
        if not valid:
            skipped += 1
            continue

        records.append(np.concatenate(feature_list, axis=0))
        key_records.append({k: v for k, v in zip(join_keys, key_vals)})

    mv_cols = [f"{feat}_qp{qp}" for qp in qp_list for feat in feature_cols]
    df_mv     = pd.DataFrame(records, columns=mv_cols)
    df_mv_key = pd.DataFrame(key_records)

    print(f"[Multi-view] valid GOPs: {len(df_mv)} | feature dim: {df_mv.shape[1]} "
          f"| skipped (incomplete QP coverage): {skipped}")
    return df_mv, df_mv_key
