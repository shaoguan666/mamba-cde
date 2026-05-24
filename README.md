# mamba-cde Workspace

This repository is a research workspace for clinical time-series modelling experiments. It contains the active SMART/SMILE implementation, reference baselines, prepared datasets, experiment outputs, and notes.

## Primary Project

The actively developed code lives in `SMART/`.

- `SMART/main_pretrain.py` pretrains SMART-family encoders.
- `SMART/main_finetune.py` fine-tunes and evaluates task heads.
- `SMART/run_all_experiments.py` launches multi-dataset, multi-model experiment batches.
- `SMART/analysis/mnar_verification.py` runs statistical audits of missingness patterns.
- `SMART/models/smart.py` contains SMART, TimeFiLM, MNAR, SMILE, SMILE-v2, SMILE-Lean, and SMILE-Lean-v2 model classes.
- `SMART/export/` and root `export/` contain generated checkpoints, logs, and experiment outputs.

See `SMART/README.md` for project-specific setup, datasets, model variants, and command examples.

## Other Top-Level Directories

- `mimic3-benchmarks/`, `NeuralCDE-master/`, `apricotM-main/`, `PMAE/`, `remasker/`, `mode-mamba-ts/`, and `positional-encoding-benchmark/` are reference or comparison projects.
- `raw/`, `physionet.org/`, `DG6111742_x64/`, and compressed archives hold local datasets or downloaded artifacts.
- `analysis/`, `logs/`, and `export/` hold generated analysis products and run logs.

## Common SMART Commands

Run commands from `SMART/` unless the command explicitly says otherwise.

```bash
python run_all_experiments.py --dry-run
python run_all_experiments.py --models smart smart-smile-lean-v2 --datasets c12 c19 --seeds 1 42 --use-torchrun --devices 0,1
python analysis/mnar_verification.py --output-dir analysis/results
python analyze_pmae.py --dataset c12 --skip_model
```

For one-off runs:

```bash
python main_pretrain.py --dataset c12 --use-smile-lean-v2 --save-last
python main_finetune.py --dataset c12 --use-smile-lean-v2
```

## Data Notes

The SMART code expects preprocessed Challenge 2012, Challenge 2019, and MIMIC-III task files in the locations used by `SMART/data/*.py`. MIMIC-III access requires following the upstream PhysioNet and `mimic3-benchmarks` preparation flow before running SMART training.

Large generated files and experiment outputs are part of this local workspace. Do not assume they are reproducible from git alone.
