"""
main.py —— pipeline entry point.

Usage:
    python -m src.main --config config/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 支持 `python src/main.py` 和 `python -m src.main`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config         import load_config
from src.features       import build_multiview_features
from src.reduction      import standardize_and_pca, attach_pca_to_keys, umap_reduce
from src.clustering     import hdbscan_cluster
from src.samples        import build_cluster_samples
from src.reassign       import reassign_minority
from src.representative import merge_pca_and_clusters, generate_cluster_report
from src.zscore         import analyze_zscore
from src.report_md      import render_markdown, write_markdown


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml",
                    help="Path to the YAML config file")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    out_dir = cfg.data.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = cfg.filenames

    # ---------- 1. 读数据 ----------
    print(f"\n[Load] {cfg.data.input_csv}")
    df_raw = pd.read_csv(cfg.data.input_csv).dropna().reset_index(drop=True)
    feature_cols = [c for c in df_raw.columns if c not in cfg.data.key_cols]
    n_videos = df_raw["videoSequence"].nunique()
    print(f"[Load] {len(df_raw)} rows | {len(feature_cols)} base features "
          f"| {n_videos} videoSequences")

    # ---------- 2. 多视图聚合 ----------
    df_mv, df_mv_key = build_multiview_features(
        df_raw, feature_cols,
        join_keys=tuple(cfg.data.join_keys),
        qp_col=cfg.data.qp_col,
        qp_list=tuple(cfg.data.qp_list),
    )

    # ---------- 3. StandardScaler + PCA ----------
    X_pca, _, _ = standardize_and_pca(df_mv, var_ratio=cfg.pca.var_ratio)
    df_pca = attach_pca_to_keys(X_pca, df_mv_key)

    # ---------- 4. UMAP (可选) ----------
    if cfg.umap.enabled:
        df_embed, _ = umap_reduce(
            X_pca, df_mv_key,
            n_components=cfg.umap.n_components,
            n_neighbors=cfg.umap.n_neighbors,
            min_dist=cfg.umap.min_dist,
            random_state=cfg.umap.random_state,
        )
        embed_cols = [c for c in df_embed.columns if c.startswith("umap_")]
    else:
        df_embed = df_pca.copy()
        embed_cols = [c for c in df_embed.columns if c.startswith("pca_")]

    # ---------- 5. HDBSCAN ----------
    df_clustered, _ = hdbscan_cluster(
        df_embed, embed_cols,
        min_cluster_size=cfg.hdbscan.min_cluster_size,
        min_samples=cfg.hdbscan.min_samples,
        cluster_selection_method=cfg.hdbscan.cluster_selection_method,
    )

    # ---------- 6. 中间件: cluster_samples ----------
    df_samples = build_cluster_samples(
        df_raw=df_raw,
        feature_cols=feature_cols,
        df_clustered=df_clustered,
        join_keys=cfg.data.join_keys,
        key_cols=cfg.data.key_cols,
    )
    p_samples = out_dir / fn.cluster_samples
    df_samples.to_csv(p_samples, index=False)
    print(f"[Samples] saved -> {p_samples}  shape={df_samples.shape}")

    # ---------- 7. 少数派归并 ----------
    mapping: dict = {}
    n_changed = 0
    if cfg.reassign.enabled:
        df_samples_reassigned, mapping, n_changed = reassign_minority(df_samples)
        p_reassign = out_dir / fn.cluster_samples_reassigned
        df_samples_reassigned.to_csv(p_reassign, index=False)
        print(f"[Reassign] mapped {len(mapping)} videos | "
              f"changed {n_changed} samples -> {p_reassign}")

        # 把归并后的 cluster 也同步回 df_clustered（让代表点 + zscore 用归并后的）
        sample_label_map = (
            df_samples_reassigned[cfg.data.join_keys + ["cluster"]]
            .drop_duplicates(subset=cfg.data.join_keys)
            .rename(columns={"cluster": "cluster_reassigned"})
        )
        df_clustered = df_clustered.merge(
            sample_label_map, on=cfg.data.join_keys, how="left"
        )
        df_clustered["cluster"] = (
            df_clustered["cluster_reassigned"]
            .fillna(df_clustered["cluster"])
            .astype(int)
        )
        df_clustered = df_clustered.drop(columns=["cluster_reassigned"])

    # ---------- 8. 代表 GOP + cluster_report ----------
    df_pca_clustered = merge_pca_and_clusters(
        df_pca, df_clustered, cfg.data.join_keys
    )
    pca_cols = [c for c in df_pca_clustered.columns if c.startswith("pca_")]

    report_text, summary_df = generate_cluster_report(
        df_pca_clustered, pca_cols,
        small_threshold=cfg.representative.small_threshold,
        mid_threshold=cfg.representative.mid_threshold,
        candidate_ratio=cfg.representative.candidate_ratio,
    )

    (out_dir / fn.cluster_report).write_text(report_text, encoding="utf-8")
    summary_df.to_csv(out_dir / fn.cluster_representatives, index=False)
    print(f"[Report] cluster_report.txt + cluster_representatives.csv saved")

    # ---------- 9. 簇均值 + Top-K Z-score ----------
    df_cluster_mean, df_zscore, cluster_topk = analyze_zscore(
        df_mv=df_mv, df_mv_key=df_mv_key, df_clustered=df_clustered,
        join_keys=cfg.data.join_keys,
        top_k=cfg.zscore.top_k,
        out_dir=str(out_dir),
        fn_feature_mean=fn.cluster_feature_mean,
        fn_zscore_full=fn.cluster_feature_zscore,
        fn_summary_txt=fn.cluster_zscore_summary,
    )

    # ---------- 10. final_report.md ----------
    labels = df_clustered["cluster"]
    md = render_markdown(
        cfg=cfg,
        n_input_rows=len(df_raw),
        n_feature_cols=len(feature_cols),
        n_videos=n_videos,
        qp_list=cfg.data.qp_list,
        n_mv_gops=len(df_mv),
        mv_feature_dim=df_mv.shape[1],
        pca_kept_dim=X_pca.shape[1],
        pca_var_explained=float((X_pca.var(axis=0).sum())
                                / (df_mv.shape[1])),  # approx
        umap_enabled=cfg.umap.enabled,
        embed_dim=len(embed_cols),
        n_clusters=int(labels[labels != -1].nunique()),
        n_noise=int((labels == -1).sum()),
        n_total=int(len(labels)),
        df_pca_clustered=df_pca_clustered,
        cluster_topk=cluster_topk,
        reassign_mapping=mapping,
        n_reassigned=n_changed,
        cluster_report_text=report_text,
    )
    write_markdown(md, out_dir / fn.final_report)

    print("\n[Done] all outputs ->", out_dir)


if __name__ == "__main__":
    main()
