"""
report_md.py —— 把所有产物汇总成 final_report.md。
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
    """生成 markdown 报告字符串。"""

    L: List[str] = []
    L.append("# GOP 聚类分析报告\n")
    L.append(f"_自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
    L.append("---\n")

    # ---------- 1. Pipeline & 参数 ----------
    L.append("## 1. Pipeline 与超参数\n")
    L.append("```")
    L.append("StandardScaler → PCA(95%) "
             + ("→ UMAP " if umap_enabled else "")
             + "→ HDBSCAN → 少数派归并 → 代表点 + Z-score")
    L.append("```\n")
    L.append("| 阶段 | 参数 | 值 |")
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

    # ---------- 2. 输入数据 ----------
    L.append("## 2. 输入数据\n")
    L.append(f"- 输入 CSV: `{cfg.data.input_csv}`")
    L.append(f"- 行数（每行 = 一个 (videoSequence, baseQP, GOP_id) 三元组）: **{n_input_rows}**")
    L.append(f"- 特征列数: **{n_feature_cols}**")
    L.append(f"- 视频序列数: **{n_videos}**")
    L.append(f"- QP 列表: **{qp_list}**\n")

    # ---------- 3. 降维 ----------
    L.append("## 3. 多视图聚合与降维\n")
    L.append(f"- 多视图聚合后有效 GOP 数: **{n_mv_gops}**")
    L.append(f"- 多视图特征维度: **{mv_feature_dim}**  ({n_feature_cols} 特征 × {len(qp_list)} QP)")
    L.append(f"- PCA 保留维度: **{pca_kept_dim}**（累计方差 {pca_var_explained:.4f}）")
    if umap_enabled:
        L.append(f"- UMAP 输出维度: **{embed_dim}**")
    else:
        L.append("- UMAP: 未启用，HDBSCAN 直接在 PCA 输出上做聚类\n")
    L.append("")

    # ---------- 4. 聚类总览 ----------
    L.append("## 4. 聚类总览\n")
    L.append(f"- 簇数（不含噪声）: **{n_clusters}**")
    L.append(f"- 噪声样本: **{n_noise} / {n_total}** ({n_noise/n_total:.1%})\n")

    df_valid = df_pca_clustered[df_pca_clustered["cluster"] != -1]
    cluster_sizes = (df_valid.groupby("cluster")
                              .size()
                              .reset_index(name="size")
                              .sort_values("cluster"))

    L.append("| Cluster | Size | 主要 videoSequence |")
    L.append("|---|---|---|")
    for c, sub in df_valid.groupby("cluster"):
        vc = sub["videoSequence"].value_counts().head(3)
        top_videos = ", ".join(f"{v}({n})" for v, n in vc.items())
        L.append(f"| {int(c)} | {len(sub)} | {top_videos} |")
    if (df_pca_clustered["cluster"] == -1).any():
        noise = df_pca_clustered[df_pca_clustered["cluster"] == -1]
        L.append(f"| -1 (noise) | {len(noise)} | — |")
    L.append("")

    # ---------- 5. 代表 GOP & 视频分布（原始 txt 嵌入） ----------
    L.append("## 5. 代表 GOP 与视频分布\n")
    L.append("```text")
    L.append(cluster_report_text.strip())
    L.append("```\n")

    # ---------- 6. 少数派归并 ----------
    L.append("## 6. 少数派归并\n")
    if not cfg.reassign.enabled:
        L.append("_（已禁用）_\n")
    else:
        L.append(f"- 策略: `{cfg.reassign.strategy}`")
        L.append(f"- 被改动 cluster 标签的样本数: **{n_reassigned}**\n")
        L.append("**videoSequence → 归并后 cluster 映射**:\n")
        L.append("| videoSequence | best_cluster |")
        L.append("|---|---|")
        for v, c in sorted(reassign_mapping.items()):
            L.append(f"| {v} | {int(c)} |")
        L.append("")

    # ---------- 7. Top-K Z-score ----------
    L.append(f"## 7. Top-{cfg.zscore.top_k} 区分特征 (按 |Z-score|)\n")
    for c in sorted(cluster_topk.keys()):
        label = "Cluster -1 (NOISE)" if c == -1 else f"Cluster {c}"
        L.append(f"### {label}\n")
        L.append("| # | feature | z_score |")
        L.append("|---|---|---|")
        for i, row in enumerate(cluster_topk[c].itertuples(index=False), 1):
            L.append(f"| {i} | `{row.feature}` | {row.z_score:+.3f} |")
        L.append("")

    # ---------- 8. 产物清单 ----------
    L.append("## 8. 产物清单\n")
    L.append(f"全部写入 `{cfg.data.output_dir}/`:\n")
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
    L.append(f"  cluster_<id>_top{cfg.zscore.top_k}_zscore_features.csv  (每簇一份)")
    L.append("```\n")

    return "\n".join(L)


def write_markdown(text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"[Report] markdown saved -> {out_path}")
