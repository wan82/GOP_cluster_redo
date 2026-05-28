"""
config.py —— 读取 YAML 配置，转成 dataclass 方便 IDE 提示。
所有路径都相对 project root（config.yaml 所在目录的上一级）解析。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


# ---------- dataclass 定义 ----------

@dataclass
class DataCfg:
    input_csv:  Path
    output_dir: Path
    key_cols:   List[str]
    join_keys:  List[str]
    qp_col:     str
    qp_list:    List[int]


@dataclass
class PCACfg:
    var_ratio: float = 0.95


@dataclass
class UMAPCfg:
    enabled:      bool = True
    n_components: int  = 7
    n_neighbors:  int  = 25
    min_dist:     float = 0.2
    random_state: int  = 42


@dataclass
class HDBSCANCfg:
    min_cluster_size:         int = 8
    min_samples:              int = 8
    cluster_selection_method: str = "eom"


@dataclass
class ReassignCfg:
    enabled:  bool = True
    strategy: str  = "video_majority"


@dataclass
class RepresentativeCfg:
    small_threshold: int   = 50
    mid_threshold:   int   = 100
    candidate_ratio: float = 0.8


@dataclass
class ZScoreCfg:
    top_k: int = 10


@dataclass
class FileNames:
    cluster_samples:            str = "cluster_samples.csv"
    cluster_samples_reassigned: str = "cluster_samples_reassigned.csv"
    cluster_report:             str = "cluster_report.txt"
    cluster_representatives:    str = "cluster_representatives.csv"
    cluster_feature_mean:       str = "cluster_feature_mean.csv"
    cluster_feature_zscore:     str = "cluster_feature_zscore_full.csv"
    cluster_zscore_summary:     str = "cluster_zscore_topk_summary.txt"
    final_report:               str = "final_report.md"


@dataclass
class Config:
    project_root:   Path
    data:           DataCfg
    pca:            PCACfg
    umap:           UMAPCfg
    hdbscan:        HDBSCANCfg
    reassign:       ReassignCfg
    representative: RepresentativeCfg
    zscore:         ZScoreCfg
    filenames:      FileNames


# ---------- loader ----------

def _abs(root: Path, p: str | Path) -> Path:
    """相对路径相对 project root 解析；绝对路径原样返回。"""
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def load_config(yaml_path: str | Path) -> Config:
    yaml_path = Path(yaml_path).resolve()
    project_root = yaml_path.parent.parent  # config/ 上一级

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    d = raw.get("data", {})
    data = DataCfg(
        input_csv  = _abs(project_root, d["input_csv"]),
        output_dir = _abs(project_root, d["output_dir"]),
        key_cols   = list(d.get("key_cols",  ["videoSequence", "baseQP", "GOP_id"])),
        join_keys  = list(d.get("join_keys", ["videoSequence", "GOP_id"])),
        qp_col     = d.get("qp_col",  "baseQP"),
        qp_list    = list(d.get("qp_list", [22, 27, 32, 37])),
    )

    pca       = PCACfg(**raw.get("pca", {}))
    umap_cfg  = UMAPCfg(**raw.get("umap", {}))
    hdb       = HDBSCANCfg(**raw.get("hdbscan", {}))
    reassign  = ReassignCfg(**raw.get("reassign", {}))
    rep       = RepresentativeCfg(**raw.get("representative", {}))
    zscore    = ZScoreCfg(**raw.get("zscore", {}))
    filenames = FileNames(**raw.get("filenames", {}))

    return Config(
        project_root  = project_root,
        data          = data,
        pca           = pca,
        umap          = umap_cfg,
        hdbscan       = hdb,
        reassign      = reassign,
        representative= rep,
        zscore        = zscore,
        filenames     = filenames,
    )
