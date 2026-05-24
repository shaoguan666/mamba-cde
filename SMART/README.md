# SMART / SMILE Research Code

This directory started from the official implementation of [SMART: Towards Pre-trained Missing-Aware Model for Patient Health Status Prediction](https://openreview.net/pdf?id=7UenF4kx4j). The local version also contains experimental missingness-aware variants used in this workspace: TimeFiLM, MNAR, SMILE, SMILE-v2, SMILE-Lean, SMILE-Lean-v2, PMAE-style proportional masking, and ablations.

## Code Map

- `main_pretrain.py` trains encoder checkpoints with fixed, SMILE curriculum, stratified, or proportional variable masking.
- `main_finetune.py` fine-tunes classifiers or LoS heads from pretrained checkpoints.
- `run_all_experiments.py` orchestrates pretrain, fine-tune, and optional visualization runs across datasets, seeds, models, and ablations.
- `models/smart.py` defines the SMART-family encoders and classifiers.
- `analysis/mnar_verification.py` audits missingness patterns and writes CSV/figure outputs under `analysis/results/`.
- `analyze_pmae.py` inspects proportional masking behavior and optional per-variable reconstruction loss.
- `visualize.py` generates missingness and representation figures from trained checkpoints.

## Data Preparation

- Cardiology: https://physionet.org/content/challenge-2012/1.0.0/
- Sepsis: https://physionet.org/content/challenge-2019/1.0.0/
- MIMIC-III: https://physionet.org/content/mimiciii/1.4/

You need to follow the instructions on the PhysioNet website to access the data.

## Data Preprocessing

For Cardiology and Sepsis, please follow the jupyter notebook in the `data` folder to preprocess the data. Please move the files from zips in Cardiology to `raw` folder before running the notebook.

For MIMIC-III, please follow the instructions in the [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks) repository to extract data from the MIMIC-III database. Notably, we adopt different settings when generating decompensation and length-of-stay data. Please replace the original `mimic3benchmark/scripts/create_decompensation.py` and `mimic3benchmark/scripts/create_length_of_stay.py` with the scripts in the `data/MIMIC-III` folder. After that, you can run the scripts in the jupyter notebook in the `data/MIMIC-III` folder to generate the tasks.

## Supported Datasets

CLI dataset names:

- `c12`
- `c19`
- `mimic_mortality`
- `mimic_phenotyping`
- `mimic_decompensation`
- `mimic_lengthofstay`

## Model Variants

The batch runner accepts the baseline and experimental names listed in `run_all_experiments.py`, including:

- `smart`
- `smart-film`
- `smart-mnar`
- `smart-smile`
- `smart-smile-film`
- `smart-smile-v2`
- `smart-smile-v2-film`
- `smart-smile-lean`
- `smart-smile-lean-v2`
- `smart-smile-lean-samepretrain`
- `smart-smile-lean-pmae`
- `smart-smile-stratified`

Architecture ablations include `no-density`, `no-mnar-bias`, `no-film`, `no-time-mnar`, `no-time-pe`, `no-policy`, `no-dynamic-mnar`, and `no-dual-head` variants where supported.

## Training and Evaluation

Run a dry run first to inspect generated commands:

```bash
python run_all_experiments.py --dry-run
```

Run a focused batch:

```bash
python run_all_experiments.py --models smart smart-smile-lean-v2 --datasets c12 c19 --seeds 1 42
```

Use local distributed launch:

```bash
python run_all_experiments.py --use-torchrun --devices 0,1 --nproc-per-node 2
```

One-off pretrain and fine-tune:

```bash
python main_pretrain.py --dataset c12 --use-smile-lean-v2 --save-last
python main_finetune.py --dataset c12 --use-smile-lean-v2
```

For MIMIC length-of-stay, the local runner uses the ROC-style classification protocol by default:

```bash
python main_finetune.py --dataset mimic_lengthofstay --los-task classification --los-label-unit auto --los-save-metric auc_micro
```

Check missingness evidence:

```bash
python analysis/mnar_verification.py --output-dir analysis/results
```

It is recommended to use fixed seeds when preprocessing datasets and running experiments.

## Outputs

Checkpoints and logs are written under:

```text
export/<dataset>/<model>/seed_<seed>/
```

The key checkpoints are `checkpoint-mse.pth` after pretraining and `checkpoint-prc.pth` after fine-tuning. `training.log` captures per-run logs.

## Citation
```
@inproceedings{yu2024smart,
  title={SMART: Towards Pre-trained Missing-Aware Model for Patient Health Status Prediction},
  author={Yu, Zhihao and Chu, Xu and Jin, Yujie and Wang, Yasha and Zhao, Junfeng},
  booktitle={NeurIPS 2024},
  year={2024}
}
```
