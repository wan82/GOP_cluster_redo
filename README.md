# GOP_cluster_redo

A re-implementation of the original `GopCluster` pipeline, running on `gop_all_featuresRename.csv`:
`StandardScaler → PCA → UMAP → HDBSCAN → Minority Reassignment → Representative GOPs + Top-K Z-score → Markdown report`.

Default hyper-parameters follow the original project (PCA 95% / UMAP 7D, n=25, d=0.2 / HDBSCAN 8,8,eom). All tunable knobs are centralised in `config/config.yaml`.

---

## Directory Layout

```
GOP_cluster_redo/
├── config/
│   └── config.yaml            # All hyper-parameters (edit here only)
├── data/
│   └── gop_all_featuresRename.csv
├── src/
│   ├── config.py              # YAML → dataclass loader
│   ├── features.py            # Multi-view aggregation (4 QP → 288-D)
│   ├── reduction.py           # StandardScaler + PCA + UMAP
│   ├── clustering.py          # HDBSCAN
│   ├── samples.py             # Builds cluster_samples.csv (intermediate artefact)
│   ├── reassign.py            # Minority reassignment (video_majority)
│   ├── representative.py      # Representative-GOP selection (medoid + farthest-point)
│   ├── zscore.py              # Per-cluster mean + Top-K Z-score
│   ├── report_md.py           # Renders final_report.md
│   └── main.py                # Pipeline entry point
├── outputs/                   # All artefacts (auto-created)
├── environment.yml            # ★ Recommended: conda environment definition
├── requirements.txt           #   Fallback: pip install (does NOT guarantee numerical reproducibility)
├── Makefile
└── README.md
```

---

## Quick Start (Recommended: conda)

```bash
# One-liner: create conda env + run pipeline
make all
```

Or step by step:

```bash
make install   # Create/update conda env from environment.yml (default name: GOP_cluster_redo)
make run       # Run pipeline (reads config/config.yaml by default)
```

Run with a different config:

```bash
make run CONFIG=config/my_other.yaml
```

Clean up:

```bash
make clean        # Remove outputs/ artefacts
make distclean    # Also remove the conda environment
```

> If you don't want to use conda, you can fall back to `make install-venv` / `make run-venv` (pip path).
> But the **pip fallback does NOT guarantee identical cluster output** — see "Environment Caveats" below.

---

> ## ⚠️ Did you get 6 or 7 clusters?
>
> **The correct answer is 7 clusters** (matching the original GopCluster).
>
> If you got **6 clusters**, it is almost certainly a **deployment-environment issue, NOT a code bug** — you most likely installed the numerical stack via `pip install` instead of `conda install` (numpy / scipy / umap-learn).
>
> **Why this happens:** UMAP + HDBSCAN output is sensitive to the underlying BLAS/LAPACK binaries. Even when the package version numbers are **byte-identical**:
> - PyPI wheels link against the wheel's own bundled OpenBLAS
> - conda-forge binaries link against conda-forge's OpenBLAS
>
> The two produce slightly different floating-point accumulation orders, and UMAP embeddings drift by nanometric amounts.
>
> **Which cluster gets affected?**
>
> Under the conda path (correct 7-cluster result), there is an independent cluster `Cluster 2` containing exactly the 19 GOPs of the `BQTerrace` sequence — this cluster represents BQTerrace's distinct encoding-behaviour pattern.
>
> Under the pip path (incorrect 6-cluster result), those 19 BQTerrace GOPs get absorbed into the **catch-all cluster** (`Cluster 3` in the 7-cluster view, size = 154), mixing with the following 11 sequences:
>
> ```
> BQMall, BQSquare, BasketballDrill, BasketballDrive, BasketballPass,
> BlowingBubbles, PartyScene, RaceHorses, RaceHorsesC, Kimono, ParkScene
> ```
>
> In other words: under the pip path the catch-all cluster grows from 154 to 173 GOPs, and BQTerrace loses its own representative GOP. The other six high-confidence clusters (Campfire / ToddlerFountain2 / Cactus / TrafficFlow / the DaylightRoad2 group, etc.) are stable under both paths.
>
> **Fix:** use `make all` (which defaults to the conda path); do NOT use `make install-venv`. See "[Environment Caveats](#environment-caveats-)" for details.

---

## Environment Caveats ⚠

**Why does this README recommend conda over pip?**
UMAP / HDBSCAN output is affected by the **underlying BLAS/LAPACK binaries**. Even if `pip list` shows identical version numbers, the `numpy` / `scipy` / `umap-learn` binaries compiled by conda-forge link against a different OpenBLAS than the PyPI wheels do. Their floating-point accumulation orders differ, causing nanometric drift in the UMAP embedding. On this small 264-GOP dataset, that drift is enough to flip the 19 BQTerrace boundary points and **collapse 7 clusters into 6**.

Empirical comparison:

| Install path | Python | Math-lib provenance | HDBSCAN output |
|---|---|---|---|
| pip + venv | 3.12 | PyPI wheel (OpenBLAS) | 6 clusters |
| pip + venv | 3.10 | PyPI wheel (OpenBLAS) | 6 clusters |
| conda env (project default) | 3.10 | conda-forge OpenBLAS | **7 clusters ✓** |

Troubleshooting checklist, in order:

1. **Python interpreter version must be 3.10** (umap-learn 0.5.7 + numba 0.61 verified on 3.10).
2. **Install all numerical packages via conda-forge**, not pip wheels.
3. **Do NOT mix conda + pip** for numpy/scipy/umap-learn.

---

## Output Artefacts

| File | Contents |
|---|---|
| `outputs/cluster_samples.csv` | **Intermediate artefact**: 1071 rows × (72 z-scored features + cluster + 3 key columns) |
| `outputs/cluster_samples_reassigned.csv` | Same as above, but `cluster` column is updated by minority reassignment |
| `outputs/cluster_report.txt` | Per-cluster size + representative GOPs + videoSequence distribution |
| `outputs/cluster_representatives.csv` | Structured table of all representative points |
| `outputs/cluster_feature_mean.csv` | Per-cluster × 288-D feature means |
| `outputs/cluster_feature_zscore_full.csv` | Per-cluster × 288-D feature z-scores |
| `outputs/cluster_<id>_top10_zscore_features.csv` | Per-cluster Top-K discriminative features |
| `outputs/cluster_zscore_topk_summary.txt` | Human-readable summary of the above |
| **`outputs/final_report.md`** | **Aggregate report bundling everything above** |

---

## Pipeline Diagram

```
gop_all_featuresRename.csv     (1071 rows × 75 cols, 4 QP × 268 GOPs)
        │
        ▼
build_multiview_features       Horizontally concatenate the 4 QP rows
        │                       per (videoSequence, GOP_id) → (264, 288)
        │                       (5 GOPs dropped due to incomplete QP coverage)
        ▼
StandardScaler + PCA(95%)      → (264, ~38)
        │
        ▼
UMAP (n=25, d=0.2, 7D)         → (264, 7)             [can be disabled in yaml]
        │
        ▼
HDBSCAN (mcs=8, ms=8, eom)     → 7 clusters + 0 noise
        │
        ├─→ build_cluster_samples       1071 rows × (72 z-scored + cluster + key)
        │       │
        │       └─→ reassign_minority   Reassign by majority-videoSequence rule
        │
        ├─→ generate_cluster_report     Representative GOPs + video distribution
        │
        ├─→ analyze_zscore              Per-cluster mean / z-score / Top-K
        │
        └─→ render_markdown             final_report.md
```

---

## Tuning Guide

Open `config/config.yaml`. Common knobs:

| What you want | Where to change |
|---|---|
| Disable UMAP, feed PCA output directly into HDBSCAN | `umap.enabled: false` |
| Denser clustering (more small clusters) | `hdbscan.min_cluster_size: 5` |
| Sparser clustering (fewer, larger clusters) | `hdbscan.min_cluster_size: 15` |
| Use `leaf` instead of `eom` | `hdbscan.cluster_selection_method: leaf` |
| Disable minority reassignment | `reassign.enabled: false` |
| More representatives per cluster | Lower `representative.small_threshold` / `mid_threshold` |
| More Top-K discriminative features | `zscore.top_k: 20` |

---

## Differences From the Original

| Dimension | Original GopCluster | This Project |
|---|---|---|
| Entry point | `python src/main.py` | `python -m src.main --config …` |
| Configuration | Hard-coded constants in `path.py` | Centralised in `config/config.yaml` |
| Intermediate | Only `cluster_samples.csv` (historical artefact) | `cluster_samples.csv` + `..._reassigned.csv`, both first-class pipeline outputs |
| Report | TXT + multiple CSVs, viewed separately | Auto-aggregated into **final_report.md** |
| Minority reassignment | Mentioned conceptually, not in main pipeline | Implemented as `reassign.py`, toggleable in YAML |
| Dependency management | Global conda env | `environment.yml` provisions a dedicated conda env |
| Entry tooling | Manual `python` invocation | `Makefile` |

Clustering **results** match the original 7-cluster output (under the recommended conda environment).

---

## Dependencies

**Recommended** (conda, see `environment.yml`):

```yaml
python=3.10
numpy=2.2.4    pandas=2.2.3    scipy=1.15.2
scikit-learn=1.7.0
umap-learn=0.5.7  numba=0.61.2  llvmlite=0.44.0  pynndescent=0.5.13
hdbscan=0.8.40
joblib=1.5.1   threadpoolctl=3.6.0   pyyaml=6.0.3
```

**Fallback** (pip, see `requirements.txt`): same version numbers, but **different BLAS binaries — numerical results may drift**.
