"""
representative.py —— representative-GOP selection + video distribution + cluster_report.txt.

Representative-point strategy (medoid + farthest point; matches the original project):
    size <= small_threshold        ->  1 representative (medoid only)
    small < size <= mid_threshold  ->  2 representatives
    size > mid_threshold           ->  3 representatives

    First rep : the global medoid in PCA space.
    Later reps: farthest-point selection from the candidate pool of points whose
                distance to the medoid is within the `candidate_ratio` quantile.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------- helpers ----------

def _n_reps_for_size(n: int, small: int, mid: int) -> int:
    if n <= small:
        return 1
    if n <= mid:
        return 2
    return 3


def _compute_medoid(X: np.ndarray) -> Tuple[int, np.ndarray]:
    diff = X[:, None, :] - X[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return int(np.argmin(dist.sum(axis=1))), dist


def _select_reps(
    df_cluster:      pd.DataFrame,
    pca_cols:        List[str],
    n_reps:          int,
    candidate_ratio: float,
) -> pd.DataFrame:
    X = df_cluster[pca_cols].values
    n = len(X)
    if n == 0:
        return df_cluster.iloc[[]].copy()

    if n_reps >= n:
        out = df_cluster.copy()
        roles = ["core_rep"] + [f"coverage_rep_{i}" for i in range(1, n)]
        out["rep_role"] = roles[:n]
        return out

    med_idx, dist = _compute_medoid(X)
    d_to_med = dist[:, med_idx]
    threshold = np.quantile(d_to_med, candidate_ratio)
    candidates = list(np.where(d_to_med <= threshold)[0])
    if med_idx not in candidates:
        candidates.append(med_idx)

    selected = [med_idx]
    while len(selected) < n_reps:
        remaining = [i for i in candidates if i not in selected]
        if not remaining:
            break
        min_d = [min(dist[i, s] for s in selected) for i in remaining]
        selected.append(remaining[int(np.argmax(min_d))])

    out = df_cluster.iloc[selected].copy()
    out["rep_role"] = ["core_rep"] + [f"coverage_rep_{i}" for i in range(1, len(out))]
    return out


# ---------- public API ----------

def merge_pca_and_clusters(
    df_pca:       pd.DataFrame,
    df_clustered: pd.DataFrame,
    join_keys:    List[str],
) -> pd.DataFrame:
    """Inner-join the PCA coordinates with the cluster labels on `join_keys`."""
    cols = join_keys + ["cluster", "confidence", "outlier_score"]
    return df_pca.merge(df_clustered[cols], on=join_keys, how="inner")


def generate_cluster_report(
    df_pca_clustered: pd.DataFrame,
    pca_cols:         List[str],
    small_threshold:  int,
    mid_threshold:    int,
    candidate_ratio:  float,
    cluster_col:      str = "cluster",
    video_col:        str = "videoSequence",
    gop_col:          str = "GOP_id",
) -> Tuple[str, pd.DataFrame]:
    """
    Returns
    -------
    report_text : human-readable txt (format-compatible with the original cluster_report.txt)
    summary_df  : one row per representative point
    """
    summary_rows: list = []
    lines: list = []

    df_valid = df_pca_clustered[df_pca_clustered[cluster_col] != -1].copy()

    for c, sub in df_valid.groupby(cluster_col):
        n_reps = _n_reps_for_size(len(sub), small_threshold, mid_threshold)
        reps = _select_reps(sub, pca_cols, n_reps, candidate_ratio)

        lines.append(f"\n===== Cluster {c} | Size = {len(sub)} =====")
        lines.append(f"Representatives ({len(reps)}):")
        for row in reps.itertuples(index=False):
            lines.append(
                f"  [{row.rep_role}]  "
                f"videoSequence={getattr(row, video_col)}, "
                f"GOP_id={getattr(row, gop_col)}, "
                f"confidence={row.confidence:.4f}"
            )
            summary_rows.append({
                "cluster":           int(c),
                "cluster_size":      len(sub),
                "rep_role":          row.rep_role,
                "rep_videoSequence": getattr(row, video_col),
                "rep_GOP_id":        getattr(row, gop_col),
                "rep_confidence":    float(row.confidence),
                "rep_outlier_score": float(row.outlier_score)
                                     if pd.notna(row.outlier_score) else np.nan,
            })

        lines.append("Video distribution:")
        for v, cnt in sub[video_col].value_counts().items():
            lines.append(f"  {v}: {cnt}")

    noise = df_pca_clustered[df_pca_clustered[cluster_col] == -1]
    if not noise.empty:
        lines.append(f"\n===== Cluster -1 (NOISE) | Size = {len(noise)} =====")
        lines.append("Video distribution:")
        for v, cnt in noise[video_col].value_counts().items():
            lines.append(f"  {v}: {cnt}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["cluster", "rep_role"])
    return "\n".join(lines), summary_df
