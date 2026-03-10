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
              'smart-smile-v2', 'smart-smile-v2-film']
ALL_SEEDS = [1, 42, 3407, 1234, 2024, 9999]


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
    parser.add_argument('--models', nargs='+', default=ALL_MODELS,
                        choices=ALL_MODELS + [
                            'smart-smile-nomnar', 'smart-smile-norandom',
                            'smart-smile-temporal-only', 'smart-smile-system-only',
                            'smart-smile-film',
                        ], metavar='MODEL',
                        help='Models to run. Available: ' + ', '.join(ALL_MODELS + [
                            'smart-smile-nomnar', 'smart-smile-norandom',
                            'smart-smile-temporal-only', 'smart-smile-system-only',
                        ]))
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS,
                        choices=ALL_DATASETS, metavar='DATASET')
    parser.add_argument('--seeds', nargs='+', type=int, default=ALL_SEEDS,
                        metavar='SEED')
    parser.add_argument('--pretrain-epochs', type=int, default=25)
    parser.add_argument('--finetune-epochs', type=int, default=25)
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
        use_smile_flag         = ['--use-smile']         if (model.startswith('smart-smile')
                                                             and model not in ('smart-smile-film',
                                                                               'smart-smile-v2',
                                                                               'smart-smile-v2-film')) else []
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
        tag_prefix = f'[{idx:>2}/{total}] {model:12s} | {dataset:25s} | seed={seed}'

        # ---- Pretrain ----
        if not args.finetune_only:
            pre_ckpt = pretrain_ckpt(dataset, model, seed)
            if not args.force and os.path.exists(pre_ckpt):
                print(f'{tag_prefix} | pretrain: SKIP (exists)')
                skipped_pre += 1
            else:
                cmd = [
                    sys.executable, 'main_pretrain.py',
                    '--dataset', dataset,
                    '--seed', str(seed),
                    '--epochs', str(args.pretrain_epochs),
                    '--batch_size', '64',
                    '--save-last',
                ] + use_film_flag + use_smile_film_flag + use_smile_v2_film_flag + use_smile_v2_flag + use_smile_flag + use_mnar_flag + smile_extra
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
                '--epochs', str(args.finetune_epochs),
                '--batch_size', '64',
            ] + use_film_flag + use_smile_film_flag + use_smile_v2_film_flag + use_smile_v2_flag + use_smile_flag + use_mnar_flag + smile_extra
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
