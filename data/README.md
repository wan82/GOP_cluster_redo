# Data

`gop_all_featuresRename.csv` 是本项目聚类管道的输入数据，直接随 repo 提交（~1.3 MB）。

每一行代表一个 (视频序列, baseQP, GOP) 三元组下的 GOP 行为特征。
其中 4 个 baseQP（22 / 27 / 32 / 37）下的同一个 (videoSequence, GOP_id) 会在 `src/features.py` 里被横向拼接成 288 维多视图向量。

---

## 1. 数据规模

| 维度 | 数值 |
|---|---|
| 行数 | 1071（含 1 行表头） |
| 列数 | 75（3 个 key + 72 个特征） |
| 视频序列数 | 21（JVET 标准测试集，分辨率 416×240 ~ 3840×2160，位深 8 bit / 10 bit） |
| QP 取值 | 22, 27, 32, 37 |
| 多视图聚合后 GOP 数 | 264（按 (videoSequence, GOP_id) 合并 4 个 QP；5 个 GOP 因 QP 不全被 `features.py` 丢弃） |

---

## 2. 数据采集流水线（数据是怎么来的）

```
原始 YUV            VTM 编码 log          解码 CTU qt-depth
    │                   │                       │
    │  ITU-T P.910      │  解析每帧            │  CTU 四叉树
    │  逐帧 SI / TI     │  Bits/PSNR/ET/QP/POC │  深度统计
    ▼                   ▼                       ▼
yuvName_SITI.csv    video_log_summary.csv   qtDepth_summary_by_POC.csv
   (POC 级 SI/TI)     (POC 级编码统计)        (POC 级 qt-depth)
    │                   │                       │
    └───────────────────┴───────────────────────┘
                        │
                        ▼   按 (videoSequence, baseQP) 把每个 GOP 内
                            32 帧的 POC 时序聚合成 8 个统计量
                        │
                        ▼
            gop_all_featuresRename.csv     ← 本文件
```

完整采集逻辑见原项目 `GopInfoCol/` 仓库（不包含在本项目里）。

---

## 3. 列定义

### 3.1 Key 列（前 3 列）

| 列 | 含义 |
|---|---|
| `videoSequence` | 视频序列名（21 个 JVET 序列之一） |
| `baseQP` | 编码时设定的基础 QP，取 22 / 27 / 32 / 37 之一 |
| `GOP_id` | 该视频序列下 GOP 的编号（每段视频被切成多个 GOP=32 帧的单元） |

三元组 `(videoSequence, baseQP, GOP_id)` 唯一标识一行。

### 3.2 特征列（后 72 列）

72 个特征列 = **9 个信号** × **8 个统计量**。
列名格式：`{统计量}_{信号}`，如 `mean_bits`、`maxJump_upsnr`、`specEntropy_qtdContrast`。

#### 9 个信号（一个 GOP 内 32 帧的时序波形）

| # | 信号 | 物理含义 |
|---|---|---|
| 1 | `bits` | 该帧的编码比特数 |
| 2 | `ypsnr` | Y 分量 PSNR（亮度质量） |
| 3 | `upsnr` | U 分量 PSNR（色度 U 质量） |
| 4 | `vpsnr` | V 分量 PSNR（色度 V 质量） |
| 5 | `et` | 该帧编码时间（归一化后） |
| 6 | `qpSI` | 编码后 YUV 的空间复杂度（ITU-T P.910 SI），代表编码后画面的空间细节量 |
| 7 | `qpTI` | 编码后 YUV 的时间复杂度（ITU-T P.910 TI），代表编码后帧间运动剧烈度 |
| 8 | `qtdFullness` | 四叉树深度的整体等级 = `qtD_mean / qtD_max`，越大代表 CTU 平均切得越深 |
| 9 | `qtdContrast` | 四叉树深度的形态对比 = `sign(depth_asym) · log(1 + depth_spread)`，越大代表深度分布越复杂、对比越强 |

> 关于 `qtdFullness` / `qtdContrast` 的数学定义：
> ```
> depth_asym   = (max - mean) / std − (mean - min) / std
> depth_spread = (max - min) · std
> qtdFullness  = mean / max
> qtdContrast  = sign(depth_asym) · log(1 + depth_spread)
> ```

#### 8 个统计量（对 GOP 内 32 帧的信号波形求得）

按物理含义分三层：

**层 1 — 总体强度**

| 统计量 | 含义 |
|---|---|
| `mean` | 32 帧波形的平均值 — 整体强度 |

**层 2 — 动态稳定性**

| 统计量 | 含义 | 解读方向 |
|---|---|---|
| `std` | 标准差 | 越大越不稳定 |
| `mad` | 帧间一阶差分的平均值（mean abs diff） | 越大帧间变化越剧烈 |
| `maxJump` | 帧间一阶差分的最大值（max abs diff） | 越大突发变化越明显（场景切换 / 大位移） |
| `zcr` | 过零率（zero-crossing rate of 1st derivative） | 越大运动频率越高 |

**层 3 — 运动特性**

| 统计量 | 含义 | 解读方向 |
|---|---|---|
| `range` | 最大值 − 最小值 | 动态范围、是否含 ROI 峰值 |
| `trend` | 线性拟合斜率 | 正值上升 / 负值下降 |
| `specEntropy` | 频谱熵（频域能量分布均匀程度） | 越大越复杂、可预测性越差 |

---

## 4. 视频序列清单（21 条）

来自 JVET 标准测试集：

```
416×240   (Class D)  ：BlowingBubbles, BasketballPass, BQSquare, RaceHorses
832×480   (Class C)  ：BasketballDrill, BQMall, PartyScene, RaceHorsesC
1280×720  (其他)     ：Drums100
1920×1080 (Class B)  ：BQTerrace, BasketballDrive, Cactus, Kimono, ParkScene
3840×2160 (Class A)  ：Campfire, CatRobot, DaylightRoad2, Rollercoaster2,
                       Tango2, ToddlerFountain2, TrafficFlow
```

位深覆盖 8 bit / 10 bit。序列名标准化由原 `GopInfoCol/unite_sequnceName.py` 的模糊匹配统一处理。

---

## 5. 与原版命名差异

历史上 `gop_all_features.csv` 用过下划线命名（如 `max_jump_bits`、`qtD_level`），
本项目使用的 `gop_all_featuresRename.csv` 已统一为驼峰且语义化：

| 原列名 | 当前列名 |
|---|---|
| `max_jump_*` | `maxJump_*` |
| `spec_entropy_*` | `specEntropy_*` |
| `qp_SI` / `qp_TI` | `qpSI` / `qpTI` |
| `qtD_level` | **`qtdFullness`** |
| `qtD_shape` | **`qtdContrast`** |

两份 CSV 的**数值完全一致**，只是列名变了。
