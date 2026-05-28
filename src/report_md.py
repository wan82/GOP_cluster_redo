"""
report_md.py —— aggregate every artefact into final_report.md.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


def render_markdown(
    *,
    cfg,
    n_input_rows:        int,
    n_feature_cols:      int,
    n_videos:            int,
    qp_list:             List[int],
    n_mv_gops:           int,
    mv_feature_dim:      int,
    pca_kept_dim:        int,
    pca_var_explained:   float,
    umap_enabled:        bool,
    embed_dim:           int,
    n_clusters:          int,
    n_noise:             int,
    n_total:             int,
    df_pca_clustered:    pd.DataFrame,
    cluster_topk:        Dict[int, pd.DataFrame],
    reassign_mapping:    Dict[str, int],
    n_reassigned:        int,
    cluster_report_text: str,
) -> str:
    """Render the markdown report as a string."""

    L: List[str] = []
    L.append("# GOP Clustering Report\n")
    L.append(f"_Auto-generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
    L.append("---\n")

    # ---------- 1. Pipeline & hyper-parameters ----------
    L.append("## 1. Pipeline and Hyper-parameters\n")
    L.append("```")
    L.append("StandardScaler → PCA(95%) "
             + ("→ UMAP " if umap_enabled else "")
             + "→ HDBSCAN → Minority Reassignment → Representative GOPs + Z-score")
    L.append("```\n")
    L.append("| Stage | Parameter | Value |")
    L.append("|---|---|---|")
    L.append(f"| PCA | var_ratio | {cfg.pca.var_ratio} |")
    if umap_enabled:
        L.append(f"| UMAP | n_components | {cfg.umap.n_components} |")
        L.append(f"| UMAP | n_neighbors  | {cfg.umap.n_neighbors} |")
        L.append(f"| UMAP | min_dist     | {cfg.umap.min_dist} |")
        L.append(f"| UMAP | random_state | {cfg.umap.random_state} |")
    L.append(f"| HDBSCAN | min_cluster_size | {cfg.hdbscan.min_cluster_size} |")
    L.append(f"| HDBSCAN | min_samples      | {cfg.hdbscan.min_samples} |")
    L.append(f"| HDBSCAN | method           | {cfg.hdbscan.cluster_selection_method} |")
    L.append(f"| Reassign | enabled | {cfg.reassign.enabled} |")
    L.append(f"| Reassign | strategy | {cfg.reassign.strategy} |")
    L.append(f"| Rep | small_threshold | {cfg.representative.small_threshold} |")
    L.append(f"| Rep | mid_threshold   | {cfg.representative.mid_threshold} |")
    L.append(f"| Rep | candidate_ratio | {cfg.representative.candidate_ratio} |")
    L.append(f"| Z-score | top_k | {cfg.zscore.top_k} |\n")

    # ---------- 2. Input data ----------
    L.append("## 2. Input Data\n")
    L.append(f"- Input CSV: `{cfg.data.input_csv}`")
    L.append(f"- Rows (each row = one (videoSequence, baseQP, GOP_id) triple): **{n_input_rows}**")
    L.append(f"- Feature columns: **{n_feature_cols}**")
    L.append(f"- Number of video sequences: **{n_videos}**")
    L.append(f"- QP list: **{qp_list}**\n")

    # ---------- 3. Aggregation & dimensionality reduction ----------
    L.append("## 3. Multi-view Aggregation and Dimensionality Reduction\n")
    L.append(f"- Valid GOPs after multi-view aggregation: **{n_mv_gops}**")
    L.append(f"- Multi-view feature dimension: **{mv_feature_dim}**  ({n_feature_cols} features × {len(qp_list)} QPs)")
    L.append(f"- PCA dimension kept: **{pca_kept_dim}** (cumulative variance {pca_var_explained:.4f})")
    if umap_enabled:
        L.append(f"- UMAP output dimension: **{embed_dim}**")
    else:
        L.append("- UMAP: disabled — HDBSCAN runs directly on PCA output\n")
    L.append("")

    # ---------- 4. Clustering overview ----------
    L.append("## 4. Clustering Overview\n")
    L.append(f"- Clusters (excluding noise): **{n_clusters}**")
    L.append(f"- Noise samples: **{n_noise} / {n_total}** ({n_noise/n_total:.1%})\n")

    df_valid = df_pca_clustered[df_pca_clustered["cluster"] != -1]
    cluster_sizes = (df_valid.groupby("cluster")
                              .size()
                              .reset_index(name="size")
                              .sort_values("cluster"))

    L.append("| Cluster | Size | Top videoSequences |")
    L.append("|---|---|---|")
    for c, sub in df_valid.groupby("cluster"):
        vc = sub["videoSequence"].value_counts().head(3)
        top_videos = ", ".join(f"{v}({n})" for v, n in vc.items())
        L.append(f"| {int(c)} | {len(sub)} | {top_videos} |")
    if (df_pca_clustered["cluster"] == -1).any():
        noise = df_pca_clustered[df_pca_clustered["cluster"] == -1]
        L.append(f"| -1 (noise) | {len(noise)} | — |")
    L.append("")

    # ---------- 5. Representative GOPs & video distribution (raw txt) ----------
    L.append("## 5. Representative GOPs and Video Distribution\n")
    L.append("```text")
    L.append(cluster_report_text.strip())
    L.append("```\n")

    # ---------- 6. Minority reassignment ----------
    L.append("## 6. Minority Reassignment\n")
    if not cfg.reassign.enabled:
        L.append("_(disabled)_\n")
    else:
        L.append(f"- Strategy: `{cfg.reassign.strategy}`")
        L.append(f"- Samples whose cluster label was reassigned: **{n_reassigned}**\n")
        L.append("**videoSequence → reassigned cluster mapping**:\n")
        L.append("| videoSequence | best_cluster |")
        L.append("|---|---|")
        for v, c in sorted(reassign_mapping.items()):
            L.append(f"| {v} | {int(c)} |")
        L.append("")

    # ---------- 7. Top-K discriminative features ----------
    L.append(f"## 7. Top-{cfg.zscore.top_k} Discriminative Features (ranked by |Z-score|)\n")
    for c in sorted(cluster_topk.keys()):
        label = "Cluster -1 (NOISE)" if c == -1 else f"Cluster {c}"
        L.append(f"### {label}\n")
        L.append("| # | feature | z_score |")
        L.append("|---|---|---|")
        for i, row in enumerate(cluster_topk[c].itertuples(index=False), 1):
            L.append(f"| {i} | `{row.feature}` | {row.z_score:+.3f} |")
        L.append("")

    # ---------- 8. Artefact list ----------
    L.append("## 8. Artefacts\n")
    L.append(f"All written to `{cfg.data.output_dir}/`:\n")
    L.append("```")
    for fn in [
        cfg.filenames.cluster_samples,
        cfg.filenames.cluster_samples_reassigned,
        cfg.filenames.cluster_report,
        cfg.filenames.cluster_representatives,
        cfg.filenames.cluster_feature_mean,
        cfg.filenames.cluster_feature_zscore,
        cfg.filenames.cluster_zscore_summary,
        cfg.filenames.final_report,
    ]:
        L.append(f"  {fn}")
    L.append(f"  cluster_<id>_top{cfg.zscore.top_k}_zscore_features.csv  (one per cluster)")
    L.append("```\n")

    return "\n".join(L)


def write_markdown(text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"[Report] markdown saved -> {out_path}")
