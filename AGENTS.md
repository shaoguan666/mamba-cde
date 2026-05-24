# Agent Notes

## Scope

This repository is a mixed research workspace. Treat `SMART/` as the active project unless the user names another subproject. The other top-level model repositories are mostly references or baselines.

## Editing Rules

- Do not modify generated experiment outputs in `export/`, `SMART/export/`, `logs/`, or `SMART/figs/` unless the user explicitly asks.
- Do not rewrite third-party baseline directories just to match local style.
- Preserve existing user changes. This workspace commonly has dirty training logs, checkpoints, and experimental code edits.
- Prefer small, targeted documentation updates over broad cleanup when only one project is involved.

## Tooling

- Do not use `rg` in this environment because it is rejected by the system. Use native PowerShell commands instead, such as `Get-ChildItem` for file discovery and `Select-String` for content search.

## SMART Conventions

- Run SMART commands from `SMART/`.
- Supported datasets are `c12`, `c19`, `mimic_mortality`, `mimic_phenotyping`, `mimic_decompensation`, and `mimic_lengthofstay`.
- Main entry points:
  - `main_pretrain.py` for encoder pretraining.
  - `main_finetune.py` for task fine-tuning and evaluation.
  - `run_all_experiments.py` for batch runs across models, datasets, and seeds.
  - `analysis/mnar_verification.py` for missingness-pattern audits.
- `run_all_experiments.py --dry-run` is the safest first check for generated commands.
- Use `--use-torchrun` with `--devices` for distributed local GPU runs. The runner sets conservative NCCL environment defaults when `SMART_SAFE_NCCL=1`.

## Documentation Hygiene

When changing SMART model behavior, update `SMART/README.md` if the change affects user-facing commands, dataset assumptions, supported variants, output layout, or evaluation protocol. Update the root `README.md` only for workspace-level structure or cross-project notes.
