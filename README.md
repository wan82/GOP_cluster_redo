# GOP_cluster_redo

复刻原 `GopCluster` 项目的聚类流水线，基于 `gop_all_featuresRename.csv` 重新做一遍：
`StandardScaler → PCA → UMAP → HDBSCAN → 少数派归并 → 代表 GOP + Top-K Z-score → md 报告`。

默认参数完全沿用原项目（PCA 95% / UMAP 7D, n=25, d=0.2 / HDBSCAN 8,8,eom），
所有可调参数都集中在 `config/config.yaml`。

---

> ## ⚠️ 重要：跑出来是 6 cluster 还是 7 cluster？
>
> **正确答案是 7 cluster**（与原版 GopCluster 一致）。
>
> 如果你跑出来是 **6 cluster**，几乎可以肯定是**部署环境问题**，**不是代码 bug** ——
> 用了 `pip install` 而不是 `conda install` 装数学库（numpy / scipy / umap-learn）。
>
> **原因**：UMAP + HDBSCAN 的输出会被底层 BLAS/LAPACK 二进制影响。即便包**版本号一字不差**：
> - PyPI 上的 wheel 链接的是 wheel 自带的 OpenBLAS
> - conda-forge 编译的 .so 链接的是 conda-forge 编译的 OpenBLAS
>
> 两者浮点累积顺序不同，UMAP 嵌入会有纳米级漂移。
>
> **具体被影响的是哪个簇？**
>
> 在 conda 路径下（正确的 7 cluster），有一个独立簇 `Cluster 2` 专门容纳 `BQTerrace` 序列的 19 个 GOP，这个簇代表 BQTerrace 自身的编码行为模式。
>
> 在 pip 路径下（错误的 6 cluster），这 19 个 BQTerrace GOP 会被吸进**大杂烩簇**（在 7-cluster 视图里是 `Cluster 3`，size = 154），跟下面这 11 个视频序列混在一起：
>
> ```
> BQMall, BQSquare, BasketballDrill, BasketballDrive, BasketballPass,
> BlowingBubbles, PartyScene, RaceHorses, RaceHorsesC, Kimono, ParkScene
> ```
>
> 也就是说，pip 路径下大杂烩簇会从 154 个 GOP 涨到 173 个，BQTerrace 不再有自己的代表 GOP。
> 其它六个高 confidence 簇（Campfire / ToddlerFountain2 / Cactus / TrafficFlow / DaylightRoad2 群 等）在两种路径下都稳定，不受影响。
>
> **解决办法**：用 `make all`（默认走 conda），不要走 `make install-venv` 路径。
> 详见下面"[环境注意事项](#环境注意事项-)"。

---

## 目录结构

```
GOP_cluster_redo/
├── config/
│   └── config.yaml            # 所有超参数 (改这里就够了)
├── data/
│   └── gop_all_featuresRename.csv
├── src/
│   ├── config.py              # 读 yaml → dataclass
│   ├── features.py            # 多视图聚合 (4 QP → 288 维)
│   ├── reduction.py           # StandardScaler + PCA + UMAP
│   ├── clustering.py          # HDBSCAN
│   ├── samples.py             # 生成 cluster_samples.csv (中间件)
│   ├── reassign.py            # 少数派归并 (video_majority)
│   ├── representative.py      # 代表 GOP (medoid + 最远点)
│   ├── zscore.py              # 簇均值 + Top-K Z-score
│   ├── report_md.py           # final_report.md 渲染
│   └── main.py                # pipeline 入口
├── outputs/                   # 所有产物 (自动生成)
├── environment.yml            # ★ 推荐: conda 环境定义
├── requirements.txt           #   回退: pip 装 (不保证数值复现)
├── Makefile
└── README.md
```

---

## 快速开始（推荐：conda）

```bash
# 一行搞定: 建 conda 环境 + 跑 pipeline
make all
```

或者分步：

```bash
make install   # 从 environment.yml 建/更新 conda 环境 (默认名 GOP_cluster_redo)
make run       # 跑 pipeline (默认读 config/config.yaml)
```

跑别的配置：

```bash
make run CONFIG=config/my_other.yaml
```

清理：

```bash
make clean        # 删 outputs/ 里的产物
make distclean    # 再删 conda 环境
```

> 如果不想用 conda，可以走 `make install-venv` / `make run-venv` 的 pip 回退路径。
> 但 **pip 回退不保证 cluster 数完全复现**，详见下面"环境注意事项"。

---

## 环境注意事项 ⚠

**为什么 README 推荐 conda 而不是 pip？**
UMAP / HDBSCAN 的输出会受 **底层 BLAS/LAPACK 二进制** 的影响。即便 `pip list` 显示的包版本号完全相同，conda-forge 编译的 `numpy` / `scipy` / `umap-learn` 跟 PyPI wheel 链接的不是同一份 OpenBLAS，浮点累积顺序不同，UMAP 嵌入会有纳米级漂移。在这个 264 个 GOP 的小数据集上，那 19 个 BQTerrace 边界点的归属会因此被翻面，**7 cluster 变成 6 cluster**。

实测对比：

| 安装方式 | Python | 数学库装法 | HDBSCAN 输出 |
|---|---|---|---|
| pip + venv | 3.12 | PyPI wheel (OpenBLAS) | 6 cluster |
| pip + venv | 3.10 | PyPI wheel (OpenBLAS) | 6 cluster |
| conda env（本项目推荐） | 3.10 | conda-forge OpenBLAS | **7 cluster ✓** |

所以排查清单按这个顺序：

1. **Python 解释器版本必须 3.10**（umap-learn 0.5.7 + numba 0.61 在 3.10 上验证过）
2. **所有数值包用 conda-forge 装**，不要 `pip install` 顶替
3. **不要混用 conda + pip 装 numpy/scipy/umap-learn**

---

## 产物清单

| 文件 | 内容 |
|---|---|
| `outputs/cluster_samples.csv` | **中间件**: 1071 行 × (72 z-score 特征 + cluster + 3 个 key) |
| `outputs/cluster_samples_reassigned.csv` | 少数派归并后版本 (cluster 列已修正) |
| `outputs/cluster_report.txt` | 每簇 size + 代表 GOP + videoSequence 分布 |
| `outputs/cluster_representatives.csv` | 代表点的结构化表格 |
| `outputs/cluster_feature_mean.csv` | 每簇 × 288 维特征的均值 |
| `outputs/cluster_feature_zscore_full.csv` | 每簇 × 288 维特征的 Z-score |
| `outputs/cluster_<id>_top10_zscore_features.csv` | 每簇 Top-K 区分特征明细 |
| `outputs/cluster_zscore_topk_summary.txt` | 上面那个的人可读汇总 |
| **`outputs/final_report.md`** | **总报告 (包含以上所有内容)** |

---

## Pipeline 流程图

```
gop_all_featuresRename.csv     (1071 行 × 75 列, 4 QP × 268 GOP)
        │
        ▼
build_multiview_features       按 (videoSequence, GOP_id) 把 4 QP 横向拼接
        │                       → (264, 288)  (5 个 GOP 因 QP 不全被丢弃)
        ▼
StandardScaler + PCA(95%)      → (264, ~38)
        │
        ▼
UMAP (n=25, d=0.2, 7D)         → (264, 7)             [可在 yaml 关掉]
        │
        ▼
HDBSCAN (mcs=8, ms=8, eom)     → 7 cluster + 0 噪声
        │
        ├─→ build_cluster_samples       1071 行 × (72 z-score + cluster + key)
        │       │
        │       └─→ reassign_minority   按 videoSequence 主导 cluster 归并
        │
        ├─→ generate_cluster_report     代表 GOP + 视频分布
        │
        ├─→ analyze_zscore              每簇 mean / z-score / Top-K
        │
        └─→ render_markdown             final_report.md
```

---

## 调参指南

打开 `config/config.yaml`，常用调点：

| 想干的事 | 改哪 |
|---|---|
| 关掉 UMAP, 直接 PCA 喂 HDBSCAN | `umap.enabled: false` |
| 让聚类更密 (更多小簇) | `hdbscan.min_cluster_size: 5` |
| 让聚类更稀 (更少大簇) | `hdbscan.min_cluster_size: 15` |
| 改用 leaf 而不是 eom | `hdbscan.cluster_selection_method: leaf` |
| 关掉少数派归并 | `reassign.enabled: false` |
| 每簇出更多代表点 | 调低 `representative.small_threshold` / `mid_threshold` |
| 看更多 Top-K 区分特征 | `zscore.top_k: 20` |

---

## 与原版的差异

| 维度 | 原 GopCluster | 本项目 |
|---|---|---|
| 入口 | `python src/main.py` | `python -m src.main --config …` |
| 配置 | 硬编码在 `path.py` 顶部常量 | 集中到 `config/config.yaml` |
| 中间件 | 仅 `cluster_samples.csv` (历史产物) | `cluster_samples.csv` + `..._reassigned.csv` 都明确为 pipeline 产物 |
| 报告 | TXT + 多个 CSV 分开看 | 自动汇总成 **final_report.md** |
| 少数派归并 | 仅在概念上提过, 未在 main pipeline 里 | 已正式实现为 `reassign.py`, yaml 可开关 |
| 依赖管理 | 全局 conda | `environment.yml` 自动建 conda 环境 |
| 入口工具 | 手动跑 Python | `Makefile` |

聚类**结果**与原版 7 cluster 完全一致（在推荐的 conda 环境下）。

---

## 依赖

**推荐**（conda，见 `environment.yml`）：

```yaml
python=3.10
numpy=2.2.4    pandas=2.2.3    scipy=1.15.2
scikit-learn=1.7.0
umap-learn=0.5.7  numba=0.61.2  llvmlite=0.44.0  pynndescent=0.5.13
hdbscan=0.8.40
joblib=1.5.1   threadpoolctl=3.6.0   pyyaml=6.0.3
```

**回退**（pip，见 `requirements.txt`）：版本一致但**链接的 BLAS 不同，数值结果可能漂移**。
