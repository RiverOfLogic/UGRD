# Chronos2 → LMP-Net 五种子复现

完整方法用冻结的 Chronos2 蒸馏 LMP-Net。训练目标为 MSE、QBD、REG 和限带增量损失，教师历史 512 步、学生历史 96 步，教师分位点固定为 0.2 / 0.8。

## 环境和输入

建议 Python 3.12。安装与驱动匹配的 CUDA PyTorch 后，在交付目录根运行：

```text
python -m pip install -r extensions/final_method/requirements.txt
```

放入本地 Chronos2 的完整权重和配置目录 `models/chronos2/`，以及下列 CSV。可在 `extensions/final_method_standard/configs/study.yaml` 修改 `teacher_path`、`data_root`、`cache_dir`、`output_dir` 和显存调度参数。相对路径以交付目录根解析。

```text
dataset/ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
dataset/weather/weather.csv
dataset/electricity/electricity.csv
```

每个训练进程用一张 GPU，支持多卡并发。编号以 `CUDA_VISIBLE_DEVICES` 可见顺序为准。全量教师缓存建议预留至少 50 GiB 磁盘空间，并为检查点留出空间。并发复现时给各运行指定独立缓存目录。

## 运行

```text
python -m extensions.final_method_standard.run
python -m extensions.final_method_standard.run --execute --gpus 0
```

第一条预览任务，第二条执行六数据集 × H48/96/192 × 种子 2026–2030，共 90 个模型。可加 `--datasets ETTh1 --horizons 96 --run-name etth1_h96` 只运行指定组合。

流程依次生成训练集教师缓存、测量显存、用固定配置从头训练五个种子、按验证 MSE 锁定各自最佳检查点，最后在原测试集评估。训练使用 FP32、AdamW、100 轮上限和 30 轮早停。标准化只用原训练段拟合，每个组合均匀抽取最多 8192 个训练窗口并对齐其固定 batch。

配置采用六数据集 Table I 固定方案，包括 ECL 的 W25。W、batch、学习率和损失系数按组合固定。QBD、增量系数和分段权重优先使用与 Table I 显示值一致的本地全精度记录，缺失时使用表中显示精度。ETTh2、ETTm1、ETTm2 的 QBD 和增量系数采用表中显示精度。

## 输出

默认写到 `outputs/final_method_standard/table_i_5seed/`。根目录保存 `validation_lock.json`、逐种子 `test_results.csv` 和按数据集及预测长度汇总的 `summary.csv`，报告 MAE/MSE 的均值、样本标准差和种子数。组合子目录保留生效配置、逐轮指标、最佳检查点、日志和纯学生 `student.pt`。

同一配置可重跑命令继续，已完成任务会跳过。中断时若已写入检查点但任务未完成，先归档该任务目录再重试。修改实验参数时使用新的运行名。

已验证交付目录导入、18 组合计划及 7 项现有损失与调度检查。ETTh1 H96 用 32 个训练窗口、每种子 1 轮、验证和测试各 1 批完成单卡五种子端到端检查。90 个正式训练任务待实际运行。
