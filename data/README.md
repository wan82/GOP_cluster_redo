# Data

`gop_all_featuresRename.csv` is the input data to this clustering pipeline. It is shipped together with the repository (~1.3 MB).

Each row represents one (videoSequence, baseQP, GOP) triple of GOP-level behavioural features. In `src/features.py`, the 4 rows of the same `(videoSequence, GOP_id)` across baseQP ∈ {22, 27, 32, 37} are horizontally concatenated into a 288-D multi-view vector.

---

## 1. Dataset Size

| Property | Value |
|---|---|
| Rows | 1071 (excluding the header row) |
| Columns | 75 (3 key + 72 feature) |
| Number of video sequences | 21 (JVET standard test set, resolution 416×240 ~ 3840×2160, bit depth 8 / 10) |
| QP values | 22, 27, 32, 37 |
| GOPs after multi-view aggregation | 264 (merged across 4 QP per `(videoSequence, GOP_id)`; 5 GOPs are dropped by `features.py` due to incomplete QP coverage) |

---

## 2. Data Collection Pipeline (How the CSV Was Built)

```
Raw YUV                VTM encoder log         Decoded CTU qt-depth
    │                      │                          │
    │  ITU-T P.910         │  Parse per-frame         │  CTU quad-tree
    │  per-frame SI/TI     │  Bits/PSNR/ET/QP/POC     │  depth statistics
    ▼                      ▼                          ▼
yuvName_SITI.csv     video_log_summary.csv    qtDepth_summary_by_POC.csv
  (POC-level SI/TI)     (POC-level encoding)      (POC-level qt-depth)
    │                      │                          │
    └──────────────────────┴──────────────────────────┘
                           │
                           ▼   Aggregate the 32-frame POC time-series within
                               each GOP into 8 summary statistics, grouped
                           │   by (videoSequence, baseQP)
                           ▼
            gop_all_featuresRename.csv     ← this file
```

The complete collection pipeline lives in the original `GopInfoCol/` project (not part of this repository).

---

## 3. Column Schema

### 3.1 Key columns (first 3 columns)

| Column | Meaning |
|---|---|
| `videoSequence` | Video-sequence name (one of 21 JVET sequences) |
| `baseQP` | Base QP used during encoding, one of 22 / 27 / 32 / 37 |
| `GOP_id` | GOP index within that video sequence (each sequence is split into GOPs of 32 frames) |

The triple `(videoSequence, baseQP, GOP_id)` uniquely identifies a row.

### 3.2 Feature columns (remaining 72 columns)

72 feature columns = **9 signals** × **8 statistics**.
Naming convention: `{statistic}_{signal}`, e.g. `mean_bits`, `maxJump_upsnr`, `specEntropy_qtdContrast`.

#### The 9 signals (time-series across the 32 frames inside one GOP)

| # | Signal | Physical meaning |
|---|---|---|
| 1 | `bits` | Number of bits used to encode the frame |
| 2 | `ypsnr` | Y-component PSNR (luma quality) |
| 3 | `upsnr` | U-component PSNR (chroma U quality) |
| 4 | `vpsnr` | V-component PSNR (chroma V quality) |
| 5 | `et` | Per-frame encoding time (after normalisation) |
| 6 | `qpSI` | Spatial Information of the decoded YUV (ITU-T P.910 SI) — spatial detail of the reconstructed frame |
| 7 | `qpTI` | Temporal Information of the decoded YUV (ITU-T P.910 TI) — inter-frame motion intensity |
| 8 | `qtdFullness` | Overall depth level of the quad-tree partition = `qtD_mean / qtD_max`. Larger means CTUs are on average split more deeply. |
| 9 | `qtdContrast` | Shape contrast of the quad-tree depth distribution = `sign(depth_asym) · log(1 + depth_spread)`. Larger means a more complex / contrasted depth distribution. |

> Mathematical definitions for `qtdFullness` / `qtdContrast`:
> ```
> depth_asym   = (max - mean) / std − (mean - min) / std
> depth_spread = (max - min) · std
> qtdFullness  = mean / max
> qtdContrast  = sign(depth_asym) · log(1 + depth_spread)
> ```

#### The 8 statistics (computed over the 32-frame waveform of each signal)

Grouped by physical meaning into three layers:

**Layer 1 — Overall magnitude**

| Statistic | Meaning |
|---|---|
| `mean` | Average of the 32-frame waveform — overall magnitude |

**Layer 2 — Dynamic stability**

| Statistic | Meaning | Interpretation |
|---|---|---|
| `std` | Standard deviation | Larger = less stable |
| `mad` | Mean absolute difference between consecutive frames | Larger = more violent inter-frame variation |
| `maxJump` | Maximum absolute difference between consecutive frames | Larger = stronger spikes (scene cut / large motion) |
| `zcr` | Zero-crossing rate of the first derivative | Larger = higher motion frequency |

**Layer 3 — Motion character**

| Statistic | Meaning | Interpretation |
|---|---|---|
| `range` | max − min | Dynamic range; presence of ROI-style peaks |
| `trend` | Slope of a linear fit | Positive = rising / negative = falling |
| `specEntropy` | Spectral entropy (uniformity of energy in the frequency domain) | Larger = more complex / less predictable |

---

## 4. Video-Sequence List (21 sequences)

From the JVET standard test set:

```
416×240   (Class D) : BlowingBubbles, BasketballPass, BQSquare, RaceHorses
832×480   (Class C) : BasketballDrill, BQMall, PartyScene, RaceHorsesC
1280×720  (other)   : Drums100
1920×1080 (Class B) : BQTerrace, BasketballDrive, Cactus, Kimono, ParkScene
3840×2160 (Class A) : Campfire, CatRobot, DaylightRoad2, Rollercoaster2,
                      Tango2, ToddlerFountain2, TrafficFlow
```

Mix of 8-bit and 10-bit. Sequence-name normalisation is handled by the fuzzy-matcher in `GopInfoCol/unite_sequnceName.py` of the original project.

---

## 5. Naming Difference vs the Older `gop_all_features.csv`

Historically, the column names in `gop_all_features.csv` used snake_case (e.g. `max_jump_bits`, `qtD_level`).
The file used by this project, `gop_all_featuresRename.csv`, has unified to camelCase and semantically clearer names:

| Old column name | Current column name |
|---|---|
| `max_jump_*` | `maxJump_*` |
| `spec_entropy_*` | `specEntropy_*` |
| `qp_SI` / `qp_TI` | `qpSI` / `qpTI` |
| `qtD_level` | **`qtdFullness`** |
| `qtD_shape` | **`qtdContrast`** |

**The numerical values in both CSVs are byte-identical** — only the column names changed.
