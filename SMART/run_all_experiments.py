"""
Run all experiments: SMART baseline vs SMART-FiLM across all datasets and seeds.

Usage examples:
    # Run everything (dry run first to check)
    python run_all_experiments.py --dry-run

    # Run all (36 finetune + 36 pretrain = 72 runs total)
    python run_all_experiments.py

    # Only run SMART-FiLM on specific datasets
    python run_all_experiments.py --models smart-film --datasets c12 c19

    # Only run specific seeds
    python run_all_experiments.py --seeds 42 1234 3407

    # Skip pretrain if checkpoint already exists, redo finetune
    python run_all_experiments.py --finetune-only

    # Resume interrupted run (skips done experiments by default)
    python run_all_experiments.py
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

SMART_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_DATASETS = [
    'c12',
    'c19',
    'mimic_mortality',
    'mimic_phenotyping',
    'mimic_decompensation',
    'mimic_lengthofstay',
]
ALL_MODELS = ['smart', 'smart-film', 'smart-smile', 'smart-smile-film', 'smart-mnar',
              'smart-smile-v2', 'smart-smile-v2-film', 'smart-smile-lean',
              'smart-smile-lean-samepretrain', 'smart-smile-lean-pmae',
              'smart-smile-stratified']
# SMILE-Lean ablation variants (architecture ablation)
ABLATION_MODELS = [
    'smart-smile-lean-no-density',
    'smart-smile-lean-no-mnar-bias',
    'smart-smile-lean-no-film',
    'smart-smile-lean-no-time-mnar',
    'smart-smile-lean-no-time-pe',
    'smart-smile-lean-no-mnar-bias-no-time-mnar',  # remove all MNAR signals
]
ALL_SEEDS = [1, 42, 3407, 1234, 2024, 9999]
# Lean models use batch_size=64 and finetune_epochs=25 (same as smart baseline)
# and save_best instead of save_last for pretrain checkpointing.
_LEAN_MODELS = {'smart-smile-lean', 'smart-smile-lean-samepretrain', 'smart-smile-lean-pmae'}
# Ablation models also use lean settings
_LEAN_MODELS.update(ABLATION_MODELS)

# Map ablation model name -> list of --abl-* CLI flags
_ABLATION_FLAGS = {
    'smart-smile-lean-no-density':                   ['--abl-no-density'],
    'smart-smile-lean-no-mnar-bias':                 ['--abl-no-mnar-bias'],
    'smart-smile-lean-no-film':                      ['--abl-no-film'],
    'smart-smile-lean-no-time-mnar':                 ['--abl-no-time-mnar'],
    'smart-smile-lean-no-time-pe':                   ['--abl-no-time-pe'],
    'smart-smile-lean-no-mnar-bias-no-time-mnar':    ['--abl-no-mnar-bias', '--abl-no-time-mnar'],
}


def pretrain_ckpt(dataset, model_name, seed):
    return os.path.join(
        SMART_DIR, 'export', dataset, model_name, f'seed_{seed}', 'checkpoint-mse.pth'
    )


def finetune_ckpt(dataset, model_name, seed):
    return os.path.join(
        SMART_DIR, 'export', dataset, model_name, f'seed_{seed}', 'checkpoint-prc.pth'
    )


def run_cmd(cmd, tag, dry_run):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'\n{"="*70}')
    print(f'[{ts}] {tag}')
    print(f'CMD: {" ".join(cmd)}')
    print('='*70, flush=True)
    if dry_run:
        print('[DRY RUN] skipped')
        return True
    result = subprocess.run(cmd, cwd=SMART_DIR)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without running them')
    _extra_models = [
        'smart-smile-nomnar', 'smart-smile-norandom',
        'smart-smile-temporal-only', 'smart-smile-system-only',
        'smart-smile-film', 'smart-smile-stratified',
    ]
    parser.add_argument('--models', nargs='+', default=ALL_MODELS,
                        choices=ALL_MODELS + ABLATION_MODELS + _extra_models,
                        metavar='MODEL',
                        help='Models to run. Available: ' + ', '.join(
                            ALL_MODELS + ABLATION_MODELS + _extra_models))
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS,
                        choices=ALL_DATASETS, metavar='DATASET')
    parser.add_argument('--seeds', nargs='+', type=int, default=ALL_SEEDS,
                        metavar='SEED')
    parser.add_argument('--pretrain-epochs', type=int, default=25)
    parser.add_argument('--finetune-epochs', type=int, default=35)
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size per GPU. Paper uses total=256 (4 GPU x 64); '
                             'single-GPU should use 256 to match effective batch size.')
    parser.add_argument('--pretrain-only', action='store_true',
                        help='Only run pretraining, skip finetuning')
    parser.add_argument('--finetune-only', action='store_true',
                        help='Only run finetuning (pretrain checkpoint must exist)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if checkpoint already exists')
    parser.add_argument('--visualize', action='store_true',
                        help='Run visualize.py after each successful finetune')
    args = parser.parse_args()

    plan = []
    for model in args.models:
        for dataset in args.datasets:
            for seed in args.seeds:
                plan.append((model, dataset, seed))

    total = len(plan)
    failed = []
    skipped_pre = 0
    skipped_ft = 0

    print(f'Total experiments: {total}')
    print(f'Models:   {args.models}')
    print(f'Datasets: {args.datasets}')
    print(f'Seeds:    {args.seeds}')
    print(f'Pretrain epochs: {args.pretrain_epochs}  |  Finetune epochs: {args.finetune_epochs}')

    for idx, (model, dataset, seed) in enumerate(plan, 1):
        use_film_flag          = ['--use-film']          if model == 'smart-film'          else []
        use_smile_film_flag    = ['--use-smile-film']    if model == 'smart-smile-film'    else []
        use_smile_v2_film_flag = ['--use-smile-v2-film'] if model == 'smart-smile-v2-film' else []
        use_smile_v2_flag      = ['--use-smile-v2']      if model == 'smart-smile-v2'      else []
        _is_lean_ablation = model in _ABLATION_FLAGS
        use_smile_lean_flag              = ['--use-smile-lean']             if model in ('smart-smile-lean', 'smart-smile-lean-pmae') or _is_lean_ablation else []
        use_smile_lean_samepretrain_flag = ['--use-smile-lean-samepretrain'] if model == 'smart-smile-lean-samepretrain' else []
        pmae_pretrain_flag               = ['--pretrain-mask-mode', 'proportional_var'] if model == 'smart-smile-lean-pmae' else []
        pmae_pretrain_dir_flag           = ['--pretrain-dir', os.path.join('./export', dataset, model, f'seed_{seed}')] if model == 'smart-smile-lean-pmae' else []
        _lean_exclude = {'smart-smile-film', 'smart-smile-v2', 'smart-smile-v2-film',
                         'smart-smile-lean', 'smart-smile-lean-samepretrain', 'smart-smile-lean-pmae'}
        _lean_exclude.update(_ABLATION_FLAGS.keys())
        use_smile_flag         = ['--use-smile']         if (model.startswith('smart-smile')
                                                             and model not in _lean_exclude) else []
        use_mnar_flag          = ['--use-mnar']          if model == 'smart-mnar'          else []
        # Ablation extra flags for smile variants
        smile_extra = []
        if model == 'smart-smile-nomnar':
            smile_extra = ['--smile-no-mnar']
        elif model == 'smart-smile-norandom':
            smile_extra = ['--smile-no-curriculum']
        elif model == 'smart-smile-temporal-only':
            smile_extra = ['--smile-mask-type', 'temporal']
        elif model == 'smart-smile-system-only':
            smile_extra = ['--smile-mask-type', 'system']
        elif model == 'smart-smile-stratified':
            smile_extra = ['--smile-stratified']
        # SMILE-Lean architecture ablation flags
        lean_abl_extra = _ABLATION_FLAGS.get(model, [])
        tag_prefix = f'[{idx:>2}/{total}] {model:12s} | {dataset:25s} | seed={seed}'

        # ---- Pretrain ----
        # Lean models: batch_size=64, save_best (same setup as smart baseline)
        cur_batch_size = 64 if model in _LEAN_MODELS else args.batch_size
        cur_ft_epochs = 25 if model in _LEAN_MODELS else args.finetune_epochs
        if not args.finetune_only:
            pre_ckpt = pretrain_ckpt(dataset, model, seed)
            if not args.force and os.path.exists(pre_ckpt):
                print(f'{tag_prefix} | pretrain: SKIP (exists)')
                skipped_pre += 1
            else:
                # LoS/Decomp: save best (curriculum loss stays low); other non-lean: save last
                # Lean models: save best (random masking; val loss is monotone-friendly)
                save_last_flag = ([]
                    if dataset in ('mimic_lengthofstay', 'mimic_decompensation') or model in _LEAN_MODELS
                    else ['--save-last'])
                cmd = [
                    sys.executable, 'main_pretrain.py',
                    '--dataset', dataset,
                    '--seed', str(seed),
                    '--epochs', str(args.pretrain_epochs),
                    '--batch_size', str(cur_batch_size),
                ] + save_last_flag + use_film_flag + use_smile_film_flag + use_smile_v2_film_flag + use_smile_v2_flag + use_smile_lean_flag + use_smile_lean_samepretrain_flag + use_smile_flag + use_mnar_flag + smile_extra + pmae_pretrain_flag + lean_abl_extra
                ok = run_cmd(cmd, f'{tag_prefix} | PRETRAIN', args.dry_run)
                if not ok:
                    failed.append(f'{tag_prefix} pretrain')
                    print(f'[WARN] pretrain failed, skipping finetune for this experiment')
                    continue

        # ---- Finetune ----
        if not args.pretrain_only:
            ft_ckpt = finetune_ckpt(dataset, model, seed)
            if not args.force and os.path.exists(ft_ckpt):
                print(f'{tag_prefix} | finetune: SKIP (exists)')
                skipped_ft += 1
                continue
            pre_ckpt = pretrain_ckpt(dataset, model, seed)
            if not args.dry_run and not os.path.exists(pre_ckpt):
                print(f'[WARN] pretrain checkpoint missing for {model}/{dataset}/seed_{seed}, skipping finetune')
                failed.append(f'{tag_prefix} finetune (no pretrain ckpt)')
                continue
            cmd = [
                sys.executable, 'main_finetune.py',
                '--dataset', dataset,
                '--seed', str(seed),
                '--epochs', str(cur_ft_epochs),
                '--batch_size', str(cur_batch_size),
            ] + use_film_flag + use_smile_film_flag + use_smile_v2_film_flag + use_smile_v2_flag + use_smile_lean_flag + use_smile_lean_samepretrain_flag + use_smile_flag + use_mnar_flag + smile_extra + pmae_pretrain_dir_flag + lean_abl_extra
            ok = run_cmd(cmd, f'{tag_prefix} | FINETUNE', args.dry_run)
            if not ok:
                failed.append(f'{tag_prefix} finetune')
            elif args.visualize:
                viz_cmd = [
                    sys.executable, 'visualize.py',
                    '--dataset', dataset,
                    '--checkpoint', finetune_ckpt(dataset, model, seed),
                    '--seed', str(seed),
                ] + use_film_flag + use_smile_flag
                run_cmd(viz_cmd, f'{tag_prefix} | VISUALIZE', args.dry_run)

    print(f'\n{"="*70}')
    print(f'Finished. Total={total}, Skipped pretrain={skipped_pre}, Skipped finetune={skipped_ft}')
    if failed:
        print(f'Failed ({len(failed)}):')
        for f in failed:
            print(f'  - {f}')
    else:
        print('All experiments completed successfully.')


if __name__ == '__main__':
    main()
