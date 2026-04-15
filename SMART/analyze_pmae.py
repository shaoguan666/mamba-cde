"""
PMAE Diagnostic: Compare A1 (fixed) vs A2 (proportional_var) pretrain.
Measures:
  1. target_count: total pretrain_mask tokens per epoch (to verify task difficulty is comparable)
  2. per-variable val loss: breakdown of reconstruction loss by variable index
Usage:
    cd SMART
    python analyze_pmae.py --dataset c12
"""
import argparse
import sys
import torch
import copy
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# Reuse pretrain helpers
from main_pretrain import (
    random_masking, proportional_random_masking,
    compute_variable_density, compute_var_max_ratios,
    smooth_l1_loss, get_masking_fn, system_level_masking,
    get_mask_system_groups,
)
from data.challenge2012 import load_challenge_2012
from data.challenge2019 import load_challenge_2019
from data.mimiciii import (load_mimic_iii_mortality, load_mimic_iii_phenotyping,
                            load_mimic_iii_decompensation, load_mimic_iii_lengthofstay)
from data.dataloader import collate_fn
from utils.variable_order import get_variable_order


def count_targets_one_epoch(dataloader, mask_mode, var_max_ratios=None,
                             min_ratio=0.0, max_ratio=0.75, system_groups=None,
                             total_epochs=25, n_epochs_sim=25):
    """Simulate one full epoch of masking; return average target count per sample."""
    total_targets = 0
    total_samples = 0
    for batch in dataloader:
        x = batch['x'].cuda()
        m = batch['mask'].cuda()
        B = x.shape[0]
        if mask_mode == 'fixed':
            _, _, pretrain_mask = random_masking(x, m, min_ratio, max_ratio)
        else:
            _, _, pretrain_mask = proportional_random_masking(
                x, m, min_ratio, max_ratio, var_max_ratios)
        total_targets += pretrain_mask.sum().item()
        total_samples += B
    return total_targets / total_samples


def per_var_val_loss(encoder, predictor, target_encoder, val_dataloader,
                      min_ratio, max_ratio, use_mnar=True, feature_names=None):
    """Compute per-variable mean reconstruction loss on val set using fixed random masking."""
    V = None
    var_loss_sum = None
    var_count_sum = None
    encoder.eval(); predictor.eval(); target_encoder.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            for k in batch:
                batch[k] = batch[k].cuda()
            enc_original_mask = batch['mask'].clone() if use_mnar else None
            h = target_encoder(**batch, original_mask=enc_original_mask)
            batch['labels'] = batch['x']
            batch['x'], batch['mask'], pretrain_mask = random_masking(
                batch['x'], batch['mask'], min_ratio, max_ratio)
            z = encoder(**batch, original_mask=enc_original_mask)
            z = predictor(z)
            # z: (B, d_model, V+1)  h: (B, d_model, V+1)
            # pretrain_mask: (B, T, V) -> need per-var L1
            # Recompute raw per-variable loss
            # actual z/h shape: (B, V, T+1, d_model); z[:,:,1:] removes CLS temporal token
            pred_v = z[:, :, 1:]   # (B, V, T, d_model)
            tgt_v  = h[:, :, 1:]   # (B, V, T, d_model)
            # pretrain_mask: (B, T, V)
            pm = pretrain_mask      # (B, T, V)
            if V is None:
                V = pm.shape[2]
                var_loss_sum  = torch.zeros(V, device='cuda')
                var_count_sum = torch.zeros(V, device='cuda')
            diff = (pred_v - tgt_v).abs()   # (B, V, T, d_model)
            diff_mean_D = diff.mean(dim=-1)  # (B, V, T) - mean over embedding dim
            pm_bvt = pm.permute(0, 2, 1).float()  # (B, V, T)
            var_loss_sum  += (diff_mean_D * pm_bvt).sum(dim=(0, 2))  # (V,)
            var_count_sum += pm_bvt.sum(dim=(0, 2))                   # (V,)
    per_var = (var_loss_sum / (var_count_sum + 1e-8)).cpu().numpy()
    return per_var


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='c12',
                        choices=['c12', 'c19', 'mimic_mortality', 'mimic_phenotyping',
                                 'mimic_decompensation', 'mimic_lengthofstay'])
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--min_mask_ratio', type=float, default=0.0)
    parser.add_argument('--max_mask_ratio', type=float, default=0.75)
    parser.add_argument('--ratio_temperature', type=float, default=1.0)
    parser.add_argument('--skip_model', action='store_true',
                        help='Skip per-var loss (no checkpoint needed)')
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--time_dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--obs_density_window', type=int, default=5)
    args = parser.parse_args()
    # Defaults for non-distributed run
    args.distributed = False
    args.world_size = 1
    args.rank = 0

    torch.manual_seed(args.seed)

    # Load dataset
    if args.dataset == 'c12':
        args.input_dim = 37; args.demo_dim = 4; args.num_class = 2; args.max_len = 48
        train_dataset, val_dataset, _ = load_challenge_2012()
    elif args.dataset == 'c19':
        args.input_dim = 34; args.demo_dim = 5; args.num_class = 2; args.max_len = 60
        train_dataset, val_dataset, _ = load_challenge_2019()
    elif args.dataset == 'mimic_mortality':
        args.input_dim = 17; args.demo_dim = 0; args.num_class = 2; args.max_len = 48
        train_dataset, val_dataset, _ = load_mimic_iii_mortality()
    else:
        raise ValueError(f'Dataset {args.dataset} not configured here')

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size,
                          sampler=RandomSampler(train_dataset), collate_fn=collate_fn)
    val_dl   = DataLoader(val_dataset,   batch_size=args.batch_size,
                          sampler=SequentialSampler(val_dataset), collate_fn=collate_fn)

    feature_names = getattr(train_dataset, 'feature_names', [])
    system_groups = get_mask_system_groups(args.dataset, feature_names)

    var_order_idx, inv_order_idx = get_variable_order(
        args.dataset.split('_')[0] if args.dataset.startswith('mimic') else args.dataset)
    args.var_order_idx = var_order_idx.cuda()
    args.inv_order_idx = inv_order_idx.cuda()

    # Compute PMAE density
    print('[PMAE] Computing variable density...')
    p_obs = compute_variable_density(train_dl)
    var_max_ratios = compute_var_max_ratios(
        p_obs, args.min_mask_ratio, args.max_mask_ratio,
        temperature=args.ratio_temperature).cuda()

    print(f'[PMAE] p_obs range: [{p_obs.min():.3f}, {p_obs.max():.3f}]  mean={p_obs.mean():.3f}')
    if feature_names:
        print('\n[PMAE] Per-variable density:')
        for i, (name, pv, mr) in enumerate(zip(feature_names, p_obs.tolist(), var_max_ratios.cpu().tolist())):
            print(f'  [{i:2d}] {name:<30s}  p_obs={pv:.3f}  max_ratio={mr:.3f}')

    # ── Section 1: Target count comparison ─────────────────────────────────────
    print('\n' + '='*60)
    print('SECTION 1: Target count (avg masked tokens per sample)')
    print('='*60)
    print('Simulating A1 (fixed)...')
    tc_a1 = count_targets_one_epoch(train_dl, 'fixed',
                                     min_ratio=args.min_mask_ratio,
                                     max_ratio=args.max_mask_ratio)
    print('Simulating A2 (proportional_var)...')
    tc_a2 = count_targets_one_epoch(train_dl, 'proportional_var',
                                     var_max_ratios=var_max_ratios,
                                     min_ratio=args.min_mask_ratio,
                                     max_ratio=args.max_mask_ratio)
    ratio_change = (tc_a2 - tc_a1) / tc_a1 * 100
    print(f'\n  A1 (fixed)           avg targets/sample: {tc_a1:.1f}')
    print(f'  A2 (proportional)    avg targets/sample: {tc_a2:.1f}')
    print(f'  Change: {ratio_change:+.1f}%')
    if abs(ratio_change) < 10:
        print('  [OK] Target counts comparable - loss improvement is real learning')
    elif ratio_change < -10:
        print('  [WARN] A2 has significantly FEWER targets - some loss drop may be due to easier task')
    else:
        print('  [INFO] A2 has more targets - loss improvement is conservative estimate')

    # ── Section 2: Per-variable loss from checkpoints ──────────────────────────
    if args.skip_model:
        print('\nSkipping per-var loss (--skip_model)')
        return

    ckpt_dir_a1 = f'./export/{args.dataset}/smart-smile-lean/seed_{args.seed}'
    ckpt_dir_a2 = f'./export/{args.dataset}/smart-smile-lean/seed_{args.seed}'

    # Check if there are two separate runs - if not, we can only do one
    import os
    ckpt_a1 = os.path.join(ckpt_dir_a1, 'checkpoint-mse.pth')

    if not os.path.exists(ckpt_a1):
        print(f'\n[WARN] Checkpoint not found: {ckpt_a1}')
        print('Run finetune first or check save_dir.')
        return

    print('\n' + '='*60)
    print('SECTION 2: Per-variable val loss from A1 checkpoint')
    print('='*60)
    print('(A2 checkpoint overwrites A1 in same dir - per-var loss shows A2 model)')

    from models.smart import SMILELeanEncoder as Encoder, EmbeddingDecoder
    encoder = Encoder(args).cuda()
    predictor = EmbeddingDecoder(args).cuda()
    target_encoder = copy.deepcopy(encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    ckpt = torch.load(ckpt_a1, weights_only=False)
    encoder.load_state_dict(ckpt['encoder'])
    predictor.load_state_dict(ckpt['predictor'])
    target_encoder.load_state_dict(ckpt['target_encoder'])
    print(f'Loaded checkpoint from epoch {ckpt["epoch"]}')

    per_var = per_var_val_loss(encoder, predictor, target_encoder, val_dl,
                                args.min_mask_ratio, args.max_mask_ratio,
                                use_mnar=True, feature_names=feature_names)

    print('\n[Per-variable val loss] (sorted by p_obs ascending):')
    order = np.argsort(p_obs.numpy())
    print(f'  {"Idx":>4}  {"Variable":<30}  {"p_obs":>6}  {"max_ratio":>9}  {"val_loss":>8}')
    print('  ' + '-'*65)
    for i in order:
        name = feature_names[i] if i < len(feature_names) else f'var_{i}'
        print(f'  [{i:2d}]  {name:<30}  {p_obs[i]:6.3f}  {var_max_ratios[i].item():9.3f}  {per_var[i]:8.4f}')

    print(f'\n  Low-density  (p_obs<0.1) mean loss: {per_var[p_obs.numpy()<0.1].mean():.4f}' if (p_obs.numpy()<0.1).any() else '')
    print(f'  High-density (p_obs>0.5) mean loss: {per_var[p_obs.numpy()>0.5].mean():.4f}' if (p_obs.numpy()>0.5).any() else '')


if __name__ == '__main__':
    main()
