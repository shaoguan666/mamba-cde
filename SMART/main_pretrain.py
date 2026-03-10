import argparse
import copy
import json
import os
import logging
import random
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import numpy as np
from tqdm import tqdm

from data.challenge2012 import load_challenge_2012
from data.challenge2019 import load_challenge_2019
from data.mimiciii import load_mimic_iii_mortality, load_mimic_iii_phenotyping, load_mimic_iii_decompensation, load_mimic_iii_lengthofstay
from data.dataloader import collate_fn
from models.smart import EmbeddingDecoder
from utils.utils import set_seed, distributed_init, init_logging
from utils.variable_order import get_variable_order


def random_masking(x, original_mask, min_mask_ratio, max_mask_ratio):
    """
    Perform per-sample random masking.
    """
    N, L, V = x.shape  # batch, length, var

    # Calculate mask ratios and lengths to keep for each sample in the batch
    mask_ratios = torch.rand(N, device=x.device) * \
        (max_mask_ratio - min_mask_ratio) + min_mask_ratio

    mask = torch.rand_like(x) < mask_ratios.view(-1, 1, 1)
    x = x * (~mask)  # True for reconstruction, False for original
    return x, original_mask * (~mask),  original_mask * mask


# ── Variable groupings (dynamically built from feature names) ─────────────────

CARDIOVASCULAR_NAMES = {
    # C12/C19 short names
    'HR', 'SysBP', 'DiaBP', 'MAP', 'HeartRate',
    'NBPsys', 'NBPdia', 'NBPmean', 'MeanBP', 'SystolicBP', 'DiastolicBP',
    # MIMIC-III YerevaNN benchmark full names
    'Heart Rate', 'Diastolic blood pressure',
    'Mean blood pressure', 'Systolic blood pressure',
}
RESPIRATORY_NAMES = {
    # C12/C19 short names
    'RR', 'SpO2', 'FiO2', 'PaO2', 'PaCO2', 'pH', 'Resp',
    'SaO2', 'EtCO2', 'RespRate', 'O2Sat',
    # MIMIC-III YerevaNN benchmark full names
    'Fraction inspired oxygen', 'Oxygen saturation',
    'Respiratory rate',
}


def build_system_groups(feature_names):
    """Build physiological system groupings from feature names. Returns {} when empty."""
    if not feature_names:
        return {}
    groups = {
        'cardiovascular': [i for i, f in enumerate(feature_names)
                           if f in CARDIOVASCULAR_NAMES],
        'respiratory':    [i for i, f in enumerate(feature_names)
                           if f in RESPIRATORY_NAMES],
    }
    used = set(groups['cardiovascular']) | set(groups['respiratory'])
    groups['lab'] = [i for i in range(len(feature_names)) if i not in used]
    return {k: v for k, v in groups.items() if v}


# ── Masking strategy functions ────────────────────────────────────────────────

def temporal_block_masking(x, original_mask, min_ratio, max_ratio):
    """Randomly select a contiguous time interval per sample, mask all variables.
    Fully vectorized (Bug fix review #4: no Python-level for loop over batch).
    """
    N, L, V = x.shape
    ratios = torch.rand(N, device=x.device) * (max_ratio - min_ratio) + min_ratio
    block_lens = (ratios * L).clamp(min=1).long()                           # (N,)
    max_starts = (L - block_lens).clamp(min=0)                              # (N,)
    starts = (torch.rand(N, device=x.device) * (max_starts + 1)).long().clamp(max=max_starts)
    t_idx = torch.arange(L, device=x.device).unsqueeze(0)                  # (1, L)
    time_mask = (t_idx >= starts.unsqueeze(1)) & \
                (t_idx < (starts + block_lens).unsqueeze(1))                # (N, L)
    mask = time_mask.unsqueeze(2).expand(-1, -1, V)                         # (N, L, V)
    x = x * (~mask)
    return x, original_mask * (~mask), original_mask * mask


def system_level_masking(x, original_mask, min_ratio, max_ratio, system_groups):
    """Randomly select a physiological system and mask all its variables in a time block.
    Fully vectorized over batch dimension; loop only over groups (typically 5-10).
    """
    group_names = list(system_groups.keys())
    N, L, V = x.shape
    device = x.device
    group_idx = torch.randint(0, len(group_names), (N,), device=device)
    ratios = torch.rand(N, device=device) * (max_ratio - min_ratio) + min_ratio
    block_lens = (ratios * L).clamp(min=1).long()
    max_starts = (L - block_lens).clamp(min=0)
    starts = (torch.rand(N, device=device) * (max_starts + 1)).long().clamp(max=max_starts)
    t_idx = torch.arange(L, device=device)
    mask = torch.zeros(N, L, V, dtype=torch.bool, device=device)
    for g, group_name in enumerate(group_names):
        var_idx = torch.tensor(system_groups[group_name], dtype=torch.long, device=device)
        sel_i = (group_idx == g).nonzero(as_tuple=True)[0]
        if sel_i.numel() == 0:
            continue
        time_in_block = (t_idx[None] >= starts[sel_i, None]) & \
                        (t_idx[None] < (starts[sel_i] + block_lens[sel_i]).unsqueeze(1))
        chunk = mask[sel_i]
        chunk[:, :, var_idx] = time_in_block.unsqueeze(-1).expand(-1, -1, len(var_idx))
        mask[sel_i] = chunk
    x = x * (~mask)
    return x, original_mask * (~mask), original_mask * mask


def get_masking_fn(epoch, total_epochs, system_groups):
    """
    Smooth probability interpolation curriculum schedule.
    Gradually increases complex masking probability as training progresses.
    Avoids hard cutpoints; random masking is always retained (~28% at end).

    progress=0:   P(system)=0,    P(temporal)=0,    P(random)=1.00
    progress=0.5: P(system)=0.18, P(temporal)=0.14, P(random)=0.68
    progress=1.0: P(system)=0.48, P(temporal)=0.24, P(random)=0.28

    Bug fix (review #1): original used `rand_prob < system_prob and system_groups`
    short-circuit, causing temporal_prob=0.72 when system_groups={} (3x inflation).
    Fix: independent if check; system_groups={} falls back to random.
    """
    progress = epoch / total_epochs
    rand_prob = random.random()
    system_prob   = max(0.0, (progress - 0.2) * 0.6)   # 0 -> 0.48
    temporal_prob = max(0.0, (progress - 0.1) * 0.8)   # 0 -> 0.72 (includes system range)

    if rand_prob < system_prob:
        if system_groups:
            return lambda x, m, mn, mx: system_level_masking(x, m, mn, mx, system_groups)
        else:
            return random_masking
    elif rand_prob < temporal_prob:
        return temporal_block_masking
    else:
        return random_masking


def apply_mnar_dropout(original_mask, dropout_rate=0.05):
    """
    Fixed lightweight dropout applied throughout training. Prevents shortcut learning
    from (original_mask - input_mask) diff signal; mitigates pretrain->finetune shift.
    0.05 is mild, retaining 95% of the original MNAR signal.
    """
    drop = torch.rand_like(original_mask.float()) < dropout_rate
    return original_mask * (~drop)


def test(args, checkpoint_path, test_dataloader):
    checkpoint = torch.load(os.path.join(args.save_dir, checkpoint_path), weights_only=False)
    save_epoch = checkpoint['epoch']
    log(logger, "last saved model is in epoch {}".format(save_epoch))
    encoder.load_state_dict(checkpoint['encoder'])
    predictor.load_state_dict(checkpoint['predictor'])
    target_encoder.load_state_dict(checkpoint['target_encoder'])
    encoder.eval()
    predictor.eval()
    target_encoder.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            for key in batch:
                batch[key] = batch[key].cuda()
            # Test uses clean mask: no dropout, consistent with finetune test()
            if args.use_mnar or args.use_smile or args.use_smile_film or args.use_smile_v2 or args.use_smile_v2_film:
                original_mask = None if args.smile_no_mnar else batch['mask'].clone()
            else:
                original_mask = None
            with torch.no_grad():
                h = target_encoder(**batch, original_mask=original_mask)
            batch['labels'] = batch['x']
            batch['x'], batch['mask'], pretrain_mask = random_masking(batch['x'], batch['mask'], args.min_mask_ratio, args.max_mask_ratio)
            z = encoder(**batch, original_mask=original_mask)
            z = predictor(z)
            test_loss += criterion(z[:, :, 1:], h[:, :, 1:], pretrain_mask.permute(0, 2, 1).unsqueeze(-1).expand_as(z[:, :, 1:])).item() * batch['x'].shape[0]
    log(logger, 'Test Loss %.4f' % (test_loss / len(test_dataset)))


def smooth_l1_loss(pred, target, pad_mask, beta=1.0):
    diff = torch.abs(pred - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    combined_mask = pad_mask.bool()
    loss = (loss * combined_mask).sum() / (combined_mask.sum() + 1e-6)
    return loss


def log(logger, msg):
    if logger is not None:
        logger.info(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic_decompensation', choices=['c12', 'c19', 'mimic_mortality', 
                            'mimic_phenotyping', 'mimic_decompensation', 'mimic_lengthofstay'])
    parser.add_argument('--data_dropout', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--seed', type=int, default=3407) 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='./export/')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--min_mask_ratio', type=float, default=0.)
    parser.add_argument('--max_mask_ratio', type=float, default=0.75)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--time_dim', type=int, default=16)
    parser.add_argument('--use-film', action='store_true', default=False)
    parser.add_argument('--use-smile', action='store_true', default=False)
    parser.add_argument('--use-smile-film', action='store_true', default=False,
                        help='Use SMILEFiLMEncoder (MNAR + FiLM joint modulation)')
    parser.add_argument('--use-smile-v2', action='store_true', default=False,
                        help='Use SMILEv2Encoder (MNAR attn bias + obs density + cross-attn fusion)')
    parser.add_argument('--use-smile-v2-film', action='store_true', default=False,
                        help='Use SMILEv2FiLMEncoder (SMILEv2 + time-conditional FiLM)')
    parser.add_argument('--use-mnar', action='store_true', default=False,
                        help='Use simplified MNAREncoder (no curriculum masking)')
    parser.add_argument('--save-last', action='store_true', default=True,
                        help='Save final epoch checkpoint instead of best val loss (default: True, because curriculum masking causes loss to monotonically increase)')
    # SMILE ablation control flags
    parser.add_argument('--smile-no-mnar', action='store_true',
                        help='Disable MissingPatternEncoder (curriculum mask only)')
    parser.add_argument('--smile-no-curriculum', action='store_true',
                        help='Disable curriculum masking (MNAR encoder only, random mask)')
    parser.add_argument('--smile-mask-type', choices=['all', 'temporal', 'system'], default='all',
                        help='Which masking types to include in curriculum')
    parser.add_argument('--smile-mnar-dropout', type=float, default=0.05,
                        help='MNAR dropout rate (default 0.05)')
    args = parser.parse_args()
    if args.use_smile_v2_film:
        from models.smart import SMILEv2FiLMEncoder as Encoder
        model_name = 'smart-smile-v2-film'
    elif args.use_smile_v2:
        from models.smart import SMILEv2Encoder as Encoder
        model_name = 'smart-smile-v2'
    elif args.use_mnar:
        from models.smart import MNAREncoder as Encoder
        model_name = 'smart-mnar'
        if not args.save_last:
            args.save_last = False  # default to save-last for mnar
    elif args.use_smile_film:
        from models.smart import SMILEFiLMEncoder as Encoder
        model_name = 'smart-smile-film'
    elif args.use_smile:
        from models.smart import SMILEEncoder as Encoder
        model_name = 'smart-smile'
        if args.smile_no_mnar:
            model_name = 'smart-smile-nomnar'
        elif args.smile_no_curriculum:
            model_name = 'smart-smile-norandom'
        elif args.smile_mask_type == 'temporal':
            model_name = 'smart-smile-temporal-only'
        elif args.smile_mask_type == 'system':
            model_name = 'smart-smile-system-only'
    elif args.use_film:
        from models.smart import TimeFiLMEncoder as Encoder
        model_name = 'smart-film'
    else:
        from models.smart import Encoder
        model_name = 'smart'
    args.save_dir = os.path.join(args.save_dir, args.dataset, model_name, f'seed_{args.seed}')
    if args.local_rank == 0 and args.save_model and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.local_rank == 0:
        logger = logging.getLogger()
        init_logging(logger, args.save_dir if args.save_model else None)
    else:
        logger = None
    log(logger, json.dumps(vars(args), indent=4))
    set_seed(args.seed)
    distributed_init(args)

    if args.dataset == 'c12':
        args.input_dim = 37
        args.demo_dim = 4
        args.num_class = 2
        args.max_len = 48
        train_dataset, val_dataset, test_dataset = load_challenge_2012()
    elif args.dataset == 'c19':
        args.input_dim = 34
        args.demo_dim = 5
        args.num_class = 2
        args.max_len = 60
        train_dataset, val_dataset, test_dataset = load_challenge_2019()
    elif args.dataset == 'mimic_mortality':
        args.input_dim = 17
        args.demo_dim = 0
        args.num_class = 2
        args.max_len = 48
        train_dataset, val_dataset, test_dataset = load_mimic_iii_mortality()
    elif args.dataset == 'mimic_phenotyping':
        args.input_dim = 17
        args.demo_dim = 0
        args.num_class = 25
        args.max_len = 60
        train_dataset, val_dataset, test_dataset = load_mimic_iii_phenotyping()
    elif args.dataset == 'mimic_decompensation':
        args.input_dim = 17
        args.demo_dim = 0
        args.num_class = 2
        args.max_len = 24
        train_dataset, val_dataset, test_dataset = load_mimic_iii_decompensation()
    elif args.dataset == 'mimic_lengthofstay':
        args.input_dim = 17
        args.demo_dim = 0
        args.num_class = 10
        args.max_len = 24
        args.max_mask_ratio = 0.75
        train_dataset, val_dataset, test_dataset = load_mimic_iii_lengthofstay()
    else:
        raise Exception("Dataset not exist!")
    if args.data_dropout > 0:
        train_dataset.dropout_data(args.data_dropout)
        val_dataset.dropout_data(args.data_dropout)
        test_dataset.dropout_data(args.data_dropout)
    log(logger, 'Dataset Loaded.')
    feature_names = getattr(train_dataset, 'feature_names', [])
    system_groups = build_system_groups(feature_names)
    if args.use_smile or args.use_smile_film:
        log(logger, f'SMILE system_groups: { {k: len(v) for k, v in system_groups.items()} }')
    if args.dataset != 'all':
        if args.distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=True)
            val_sampler = SequentialSampler(val_dataset)
            test_sampler = SequentialSampler(test_dataset)
        else:
            train_sampler = RandomSampler(train_dataset)
            val_sampler = SequentialSampler(val_dataset)
            test_sampler = SequentialSampler(test_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, collate_fn=collate_fn)

    var_order_idx, inv_order_idx = get_variable_order(
        args.dataset.split('_')[0] if args.dataset.startswith('mimic') else args.dataset
    )
    args.var_order_idx = var_order_idx.cuda()
    args.inv_order_idx = inv_order_idx.cuda()

    encoder = Encoder(args).cuda()
    predictor = EmbeddingDecoder(args).cuda()
    target_encoder = copy.deepcopy(encoder)
    
    if args.distributed:
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, static_graph=True, device_ids=[args.gpu], output_device=args.local_rank, find_unused_parameters=True)
        predictor = torch.nn.parallel.DistributedDataParallel(predictor, static_graph=True, device_ids=[args.gpu], output_device=args.local_rank, find_unused_parameters=True)
        target_encoder = torch.nn.parallel.DistributedDataParallel(target_encoder, device_ids=[args.gpu], output_device=args.local_rank, find_unused_parameters=True)
    for p in target_encoder.parameters():
        p.requires_grad = False
        
    ema = [0.996, 1]
    ipe = len(train_dataloader)
    ipe_scale = 1.0
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*args.epochs*ipe_scale)
                          for i in range(int(ipe*args.epochs*ipe_scale)+1))
    
    param_groups = [
        {
            'params': encoder.parameters(),
        }, 
        {
            'params': predictor.parameters()
        }
    ]
    optimizer = torch.optim.Adam(param_groups, args.lr)
    criterion = smooth_l1_loss

    best_auc = 0
    best_prc = 0
    best_mse = float('inf')
    epoch_bar = tqdm(range(1, args.epochs + 1), desc='[Pretrain]', unit='epoch')
    for i in epoch_bar:
        train_loss = 0
        val_loss = 0
        encoder.train()
        predictor.train()
        target_encoder.train()
        batch_bar = tqdm(train_dataloader, desc=f'  Ep{i:>3}', leave=False, unit='batch')
        for step, batch in enumerate(batch_bar, 1):
            for key in batch:
                batch[key] = batch[key].cuda()
            # Save original clinical observation pattern with fixed MNAR dropout
            original_mask = apply_mnar_dropout(batch['mask'].clone(),
                                               dropout_rate=args.smile_mnar_dropout)
            # When ablating MNAR encoder, pass None so SMILEEncoder skips it
            enc_original_mask = None if args.smile_no_mnar else original_mask
            # smart-mnar: always random masking, no curriculum
            if args.use_mnar:
                masking_fn = random_masking
            elif args.smile_no_curriculum:
                masking_fn = random_masking
            else:
                mask_sg = {} if args.smile_mask_type == 'temporal' else system_groups
                if args.smile_mask_type == 'system':
                    masking_fn = (lambda x, m, mn, mx: system_level_masking(x, m, mn, mx, system_groups)
                                  if system_groups else random_masking)
                else:
                    masking_fn = get_masking_fn(i, args.epochs, mask_sg)
            with torch.no_grad():
                h = target_encoder(**batch, original_mask=enc_original_mask)
            batch['labels'] = batch['x']
            batch['x'], batch['mask'], pretrain_mask = masking_fn(
                batch['x'], batch['mask'], args.min_mask_ratio, args.max_mask_ratio)
            z = encoder(**batch, original_mask=enc_original_mask)
            z = predictor(z)
            loss = criterion(z[:, :, 1:], h[:, :, 1:], pretrain_mask.permute(0, 2, 1).unsqueeze(-1).expand_as(z[:, :, 1:]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                m = next(momentum_scheduler)
                for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                    param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
            train_loss += loss.item() * batch['x'].shape[0]
            batch_bar.set_postfix(loss=f'{loss.item():.4f}')

        encoder.eval()
        predictor.eval()
        target_encoder.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                for key in batch:
                    batch[key] = batch[key].cuda()
                # Val uses clean mask: stable metric consistent with finetune val
                if args.use_mnar or args.use_smile or args.use_smile_film or args.use_smile_v2 or args.use_smile_v2_film:
                    enc_original_mask = None if args.smile_no_mnar else batch['mask'].clone()
                else:
                    enc_original_mask = None
                h = target_encoder(**batch, original_mask=enc_original_mask)
                batch['labels'] = batch['x']
                batch['x'], batch['mask'], pretrain_mask = random_masking(
                    batch['x'], batch['mask'], args.min_mask_ratio, args.max_mask_ratio)
                z = encoder(**batch, original_mask=enc_original_mask)
                z = predictor(z)
                val_loss += criterion(z[:, :, 1:], h[:, :, 1:], pretrain_mask.permute(0, 2, 1).unsqueeze(-1).expand_as(z[:, :, 1:])).item() * batch['x'].shape[0]
        t_loss = train_loss / len(train_dataset) * args.world_size
        v_loss = val_loss / len(val_dataset)
        epoch_bar.set_postfix(train=f'{t_loss:.4f}', val=f'{v_loss:.4f}')
        log(logger, 'Epoch %d: Train Loss %.4f, Valid Loss %.4f' % (i, t_loss, v_loss))
        cur_mse = v_loss
        if args.save_last:
            # Save at final epoch (ensures model benefits from full training)
            if i == args.epochs and args.local_rank == 0:
                state = {
                    'encoder': encoder.state_dict(),
                    'predictor': predictor.state_dict(),
                    'target_encoder': target_encoder.state_dict(),
                    'epoch': i
                }
                log(logger, '----- Save last epoch model - L1: %.4f -----' % cur_mse)
                torch.save(state, os.path.join(args.save_dir, 'checkpoint-mse.pth'))
        elif cur_mse < best_mse:
            best_mse = cur_mse
            if args.local_rank == 0:
                state = {
                    'encoder': encoder.state_dict(),
                    'predictor': predictor.state_dict(),
                    'target_encoder': target_encoder.state_dict(),
                    'epoch': i
                }
                log(logger, '----- Save best model - L1: %.4f -----' % cur_mse)
                torch.save(state, os.path.join(args.save_dir, 'checkpoint-mse.pth'))
        if args.distributed:
            dist.barrier()

    if args.distributed:
        dist.barrier()
    test(args, 'checkpoint-mse.pth', test_dataloader)
