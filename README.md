# UGRD

UGRD distills a frozen Chronos2 teacher into LMP-Net for multivariate time-series forecasting. A teacher's point forecast alone does not convey its uncertainty, the relationships between variables, or how the forecast evolves over the prediction horizon. UGRD transfers these signals during training while keeping inference limited to a compact point-forecasting student.

## Method

The student learns from the observed targets through mean squared error (MSE) and from the teacher through three additional losses.

- **Quantile-band distillation (QBD)** aligns the student with the teacher's median and penalizes excursions beyond the teacher's 0.2–0.8 quantile band. Four horizon segments have separate loss weights.
- **Cross-channel relation matching (REG)** aligns pairwise channel similarities in the student's seasonal representation with those in the teacher's encoder representation.
- **Band-limited increment matching (INC)** projects the student forecast and teacher median onto selected low-frequency DCT components, then matches their lagged increments using a Huber loss.

The teacher sees 512 history steps during cache preparation. The student uses 96 steps and produces point forecasts without the teacher at inference time. The training objective is implemented in [objective.py](extensions/final_method/objective.py).

## Reproduce the experiments

Use Python 3.12 and a CUDA-compatible PyTorch installation. From the repository root, install the remaining dependencies:

```bash
python -m pip install -r extensions/final_method/requirements.txt
```

Place the Chronos2 weights and configuration in `models/chronos2/`, and put the CSV files at the following paths. The data, teacher, cache, and output paths are configurable in [study.yaml](extensions/final_method_standard/configs/study.yaml).

```text
dataset/ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
dataset/weather/weather.csv
dataset/electricity/electricity.csv
```

Preview the run or launch it on the first visible GPU:

```bash
python -m extensions.final_method_standard.run
python -m extensions.final_method_standard.run --execute --gpus 0
```

The default run uses the Table I configuration for six datasets, horizons 48/96/192, and seeds 2026–2030. It prepares training-only teacher caches, selects each checkpoint by validation MSE, then evaluates on the test split. The runner writes per-seed scores to `test_results.csv` and reports mean and sample standard deviation for MAE and MSE in `summary.csv` under `outputs/final_method_standard/table_i_5seed/`. Allow at least 50 GiB for the full cache and additional space for checkpoints.

To run one dataset and horizon, add `--datasets ETTh1 --horizons 96 --run-name etth1_h96`. Use `--gpus 0,1` to schedule independent jobs across two visible GPUs.

The configuration uses full-precision Table I coefficients where available. The QBD and INC coefficients for ETTh2, ETTm1, and ETTm2 use the precision printed in Table I. The five-seed pipeline was smoke-tested end to end on ETTh1 at horizon 96 with a reduced training and evaluation budget.
