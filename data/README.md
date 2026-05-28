# Data

`gop_all_featuresRename.csv` 是 pipeline 的**输入数据**，直接随 repo 一起提交（~1.3 MB）。

## 数据格式

| 列 | 含义 |
|---|---|
| `videoSequence` | 视频序列名 (21 个 JVET 标准序列) |
| `baseQP` | 编码 QP 值 (22 / 27 / 32 / 37 之一) |
| `GOP_id` | 该序列内 GOP 的编号 |
| 其余 72 列 | GOP 级特征 (9 信号 × 8 统计量) |

特征列名按 `{统计量}_{信号}` 命名，例如 `mean_bits`、`maxJump_upsnr`、`specEntropy_qtdContrast`。

更详细的数据来源 / 含义见原 `GopCluster/README.md` 与 `GopInfoCol/README.md`。
