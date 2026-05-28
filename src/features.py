"""
features.py —— multi-view feature aggregation.

For every (videoSequence, GOP_id), horizontally concatenate the feature vectors
across the 4 QP values into a single multi-view row, producing a matrix of shape
(n_gops, 72 * len(qp_list)).

Output column naming convention: ``{base_feature}_qp{qp}``, e.g. ``mean_bits_qp22``.
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
    df            : long-format table (one row per (videoSequence, baseQP, GOP_id) triple)
    feature_cols  : the 72 base feature column names
    join_keys     : columns that uniquely identify one GOP
    qp_col        : the column that holds the QP value
    qp_list       : every QP value that must be present; a GOP missing any QP is dropped

    Returns
    -------
    df_mv     : multi-view feature DataFrame, shape = (n_valid_gops, 72 * |qp_list|)
    df_mv_key : key-column DataFrame aligned row-by-row with df_mv
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
