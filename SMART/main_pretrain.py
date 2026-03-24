import argparse
import copy
import json
import os
import logging
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


# ── PMAE v1: Variable-level proportional masking ──────────────────────────────

def compute_variable_density(dataloader):
    """One-time scan of training set to compute per-variable observation fraction.
    Returns p_obs: (V,) float tensor on CPU, values in [0, 1].
    """
    total_obs = None
    total_steps = 0
    for batch in dataloader:
        m = batch['mask'].float()          # (B, T, V)
        obs = m.sum(dim=(0, 1)).cpu()      # (V,)
        total_obs = obs if total_obs is None else total_obs + obs
        total_steps += m.shape[0] * m.shape[1]
    return total_obs / (total_steps + 1e-8)


def compute_var_max_ratios(p_obs, ratio_min, ratio_max, temperature=1.0):
    """Maps per-variable observation density to per-variable max masking ratio.
    temperature=1.0: linear  max_ratio_v = ratio_min + (ratio_max - ratio_min) * p_obs_v
    temperature>1:   sigmoid-like, amplifies contrast between high/low density vars.
    Returns var_max_ratios: (V,) in [ratio_min, ratio_max]
    """
    if temperature == 1.0:
        f = p_obs.clamp(0.0, 1.0)
    else:
        raw = torch.sigmoid(temperature * (2.0 * p_obs - 1.0))
        lo = torch.sigmoid(torch.full_like(p_obs, -temperature))
        hi = torch.sigmoid(torch.full_like(p_obs, temperature))
        f = (raw - lo) / (hi - lo + 1e-8)
    return ratio_min + (ratio_max - ratio_min) * f


def proportional_random_masking(x, original_mask, min_mask_ratio, max_mask_ratio,
                                 var_max_ratios):
    """PMAE-style per-variable proportional masking.
    Drop-in replacement for random_masking() when --pretrain-mask-mode=proportional_var.

    Difference from random_masking:
      random_masking:          mask_ratios ~ (B,)   -> same ratio for all variables
      proportional_random_masking: sampled_ratios ~ (B, V) -> each variable sampled independently

    Args:
        x              : (B, T, V) float32
        original_mask  : (B, T, V) bool, 1=observed
        min_mask_ratio : float
        max_mask_ratio : float  (unused directly; upper bound encoded in var_max_ratios)
        var_max_ratios : (V,) float tensor from compute_var_max_ratios()

    Returns same triple as random_masking:
        x_masked, visible_mask, pretrain_mask
    """
    B, T, V = x.shape
    device = x.device
    rand_bv = torch.rand(B, V, device=device)                                    # (B, V)
    sampled_ratios = (min_mask_ratio +
                      (var_max_ratios.to(device) - min_mask_ratio) * rand_bv)   # (B, V)
    rand_btv = torch.rand(B, T, V, device=device)                                # (B, T, V)
    mask = rand_btv < sampled_ratios.unsqueeze(1)                                # (B, T, V)
    x = x * (~mask)
    return x, original_mask * (~mask), original_mask * mask


# ── Variable groupings (dynamically built from feature names) ─────────────────

CARDIOVASCULAR_NAMES = {
    # C12/C19 short names
    'HR', 'SysBP', 'DiaBP', 'MAP', 'HeartRate',
    'NBPsys', 'NBPdia', 'NBPmean', 'MeanBP', 'SystolicBP', 'DiastolicBP',
    'SBP', 'DBP',
    # MIMIC-III YerevaNN benchmark full names
    'Heart Rate', 'Diastolic blood pressure',
    'Mean blood pressure', 'Systolic blood pressure',
}
RESPIRATORY_NAMES = {
    # C12/C19 short names
    'RR', 'SpO2', 'FiO2', 'PaO2', 'PaCO2', 'pH', 'Resp',
    'SaO2', 'EtCO2', 'RespRate', 'O2Sat', 'BE', 'BaseExcess',
    # MIMIC-III YerevaNN benchmark full names
    'Fraction inspired oxygen', 'Oxygen saturation',
    'Respiratory rate', 'pH',
}
RENAL_NAMES = {
    'Creatinine', 'BUN', 'Urine', 'Chloride', 'Sodium', 'Potassium', 'HCO3',
    'Blood urea nitrogen',
}
HEPATIC_NAMES = {
    'TotalBili', 'Albumin', 'ASAT', 'ALAT', 'ALP', 'Bilirubin', 'Bilirubin_direct',
    'Bilirubin_total', 'AST', 'Alkalinephos',
}
HEMATOLOGY_NAMES = {
    'WBC', 'Hct', 'Platelets', 'PTT', 'PT', 'Hgb', 'Fibrinogen',
    'White blood cell count', 'Hematocrit',
}
METABOLIC_NAMES = {
    'Glucose', 'Lactate', 'Magnesium', 'Calcium', 'Troponin', 'TroponinI',
    'Phosphate', 'PaO2FiO2',
}


def build_system_groups(feature_names):
    """Build physiological system groupings from feature names. Returns {} when empty.

    Splits variables into 6 physiological sub-groups (3-8 vars each) instead of
    a monolithic 'lab' group. This prevents system_level_masking from wiping out
    73% of C12 variables at once (which caused extreme val loss spikes).
    """
    if not feature_names:
        return {}
    groups = {
        'cardiovascular': [i for i, f in enumerate(feature_names) if f in CARDIOVASCULAR_NAMES],
        'respiratory':    [i for i, f in enumerate(feature_names) if f in RESPIRATORY_NAMES],
        'renal':          [i for i, f in enumerate(feature_names) if f in RENAL_NAMES],
        'hepatic':        [i for i, f in enumerate(feature_names) if f in HEPATIC_NAMES],
        'hematology':     [i for i, f in enumerate(feature_names) if f in HEMATOLOGY_NAMES],
        'metabolic':      [i for i, f in enumerate(feature_names) if f in METABOLIC_NAMES],
    }
    used = set(idx for g in groups.values() for idx in g)
    other = [i for i in range(len(feature_names)) if i not in used]
    if other:
        groups['other'] = other
    return {k: v for k, v in groups.items() if v}


# Dataset-specific system groups keyed by FEATURE_NAMES_* index order.
# C12 indices based on FEATURE_NAMES_C12 (data/challenge2012.py),
# informed by t4_block.csv block-missingness significance (analysis/results/).
_C12_MASK_GROUPS = {
    # Significant co-missingness (delta > 0.5 in t4_block.csv)
    'Vital_Invasive':  [3, 4, 5],               # SysBP, MAP, DiaBP
    'Blood_Gas':       [8, 20, 21, 22],          # pH, FiO2, PaO2, PaCO2
    'Electrolyte':     [10, 11, 12, 17, 18],     # Sodium, Potassium, HCO3, Magnesium, Calcium
    'Liver_Enzyme':    [13, 14, 23, 24, 25, 28], # TotalBili, Albumin, ASAT, ALAT, ALP, Bilirubin
    'CBC':             [7, 19, 32],              # WBC, Hct, Platelets
    'Oxygenation':     [1, 35, 36],             # SpO2, PaO2FiO2, BE
    'Metabolic':       [15, 16],                # Glucose, Lactate
    # Removed (E-fix): Vital_Basic(delta=-0.035), Renal(p=0.40), Coagulation(delta=-0.13)
    # GCS(27), Weight(29), MechVent(30) not grouped (non-physiological)
    # Troponin(31), PTT(33), PT(34) ungrouped (no co-missingness support)
    # HR(0), Temp(2), Resp(6), Creatinine(9), Urine(26) ungrouped (no co-missingness)
}

# C19: indices based on FEATURE_NAMES_C19 (challenge2019.py) = C19_FEATURES order (verified).
# Groups validated against analysis/results/C19_Sepsis/t4_block.csv.
_C19_MASK_GROUPS = {
    'Vital_Basic':    [0, 1, 2, 3, 4, 5, 6],    # HR, O2Sat, Temp, SBP, MAP, DBP, Resp
    'Resp_Gas':       [7, 8, 9, 10, 11, 12, 13], # EtCO2, BaseExcess, HCO3, FiO2, pH, PaCO2, SaO2
    'Liver_Enzyme':   [14, 16],                  # AST, Alkalinephos
    'Renal':          [15, 18, 19],              # BUN, Chloride, Creatinine
    'Coagulation':    [30, 32, 33],              # PTT, Fibrinogen, Platelets
    'CBC':            [28, 29, 31],              # Hct, Hgb, WBC
    'Bilirubin':      [20, 26],                  # Bilirubin_direct, Bilirubin_total
    # Removed (E-fix): Blood_Gas_Acid (subset of Resp_Gas, 4 overlapping indices)
    # Removed (E-fix): Electrolyte (delta=0.10, p=0.16, not significant)
    # Calcium(17), Lactate(22), TroponinI(27) ungrouped (low block-missingness in t4)
}

# MIMIC-III: indices based on FEATURE_NAMES_MIMIC (mimiciii.py) = MIMIC_FEATURES order (verified).
# Groups validated against analysis/results/MIMIC3_Mortality/t4_block.csv.
_MIMIC_MASK_GROUPS = {
    'Vital_BP':    [1, 10, 13],         # DiasBP, MeanBP, SysBP
    'Vital_Basic': [0, 8, 11, 12, 14],  # CapRefill, HR, O2Sat, RespRate, Temp
    'GCS':         [3, 4, 5, 6],        # GCS eye/motor/total/verbal
    'Anthropo':    [9, 15],             # Height, Weight
    # Removed (E-fix): Resp_Gas (delta=0.005, p=0.27, no co-missingness support)
    #   - also eliminates O2Sat(11) double-membership (was in Vital_Basic + Resp_Gas)
    #   - FiO2(2), pH(16) fall back to random/temporal masking
    # Glucose(7) ungrouped (no natural co-missingness group in MIMIC)
}

_DATASET_MASK_GROUPS = {
    'c12': _C12_MASK_GROUPS,
    'c19': _C19_MASK_GROUPS,
    'mimic_mortality':     _MIMIC_MASK_GROUPS,
    'mimic_phenotyping':   _MIMIC_MASK_GROUPS,
    'mimic_decompensation': _MIMIC_MASK_GROUPS,
    'mimic_lengthofstay':  _MIMIC_MASK_GROUPS,
}


def get_mask_system_groups(dataset, feature_names):
    """Return dataset-specific index-based system groups for system_level_masking.
    Uses hardcoded index groups for known datasets; falls back to name-matching for others.
    """
    if dataset in _DATASET_MASK_GROUPS:
        return _DATASET_MASK_GROUPS[dataset]
    return build_system_groups(feature_names)


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
    rand_prob = torch.rand(1).item()
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


# Dataset-adaptive fixed ratios (Scheme F) with epoch-level stratified sampling (Scheme D)
# Rationale: 25 epochs too short for curriculum warm-up; fixed ratios give full
# strategy exposure from epoch 1. Per-dataset ratios reflect system_groups coverage.

_DATASET_MASK_RATIOS = {
    # C12: 7 groups, 75.7% coverage, 9 ungrouped vars -> more random
    'c12':   (0.30, 0.25, 0.45),
    # C19: 7 groups, 88.2% coverage, strict stat validation -> more system
    'c19':   (0.35, 0.25, 0.40),
    # MIMIC: 4 groups, 82.4% coverage, strong temporal patterns -> more temporal
    'mimic': (0.25, 0.30, 0.45),
}


def _build_stratified_schedule(n_batches, p_sys, p_temp, p_rand, epoch_seed):
    """Build a deterministic per-batch strategy assignment for one epoch.

    Assigns exactly floor(n*p) batches to each strategy (remainder to random),
    then shuffles with a fixed seed for reproducibility across runs.

    Returns list[str] of length n_batches, each element in
    {'system', 'temporal', 'random'}.
    """
    n_sys = int(n_batches * p_sys)
    n_temp = int(n_batches * p_temp)
    n_rand = n_batches - n_sys - n_temp  # remainder goes to random

    schedule = ['system'] * n_sys + ['temporal'] * n_temp + ['random'] * n_rand
    rng = np.random.RandomState(epoch_seed)
    rng.shuffle(schedule)
    return schedule


def get_stratified_masking_fn(strategy, system_groups):
    """Return the masking function for a given strategy name."""
    if strategy == 'system' and system_groups:
        return lambda x, m, mn, mx: system_level_masking(x, m, mn, mx, system_groups)
    elif strategy == 'temporal':
        return temporal_block_masking
    else:
        return random_masking


def apply_mnar_dropout(original_mask, dropout_rate=0.05):
    """
    Lightweight dropout on observed positions. Prevents shortcut learning from
    (original_mask - input_mask) diff signal; mitigates pretrain->finetune shift.
    """
    drop = torch.rand_like(original_mask.float()) < dropout_rate
    return original_mask * (~drop)


def get_mnar_dropout_rate(epoch, total_epochs, max_rate):
    """Linear ramp: near-0 at epoch 1, max_rate at epoch total_epochs.
    Early training stays stable; regularization increases as model matures.
    """
    return max_rate * (epoch / total_epochs)


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
            # samepretrain: always None (no MNAR encoder, same as training)
            if (args.use_mnar or args.use_smile or args.use_smile_film or args.use_smile_v2
                    or args.use_smile_v2_film or args.use_smile_lean or args.use_smile_lean_samepretrain):
                original_mask = None if (args.smile_no_mnar or args.use_smile_lean_samepretrain) else batch['mask'].clone()
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
    parser.add_argument('--min_mask_ratio', type=float, default=0.15)
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
    parser.add_argument('--use-smile-lean', action='store_true', default=False,
                        help='Use SMILELeanEncoder (MNAR cooccur bias + VarAtt FiLM + local obs density)')
    parser.add_argument('--use-smile-lean-samepretrain', action='store_true', default=False,
                        help='Use SMILELeanEncoder with same pretrain as smart (random masking, no MNAR)')
    parser.add_argument('--use-mnar', action='store_true', default=False,
                        help='Use simplified MNAREncoder (no curriculum masking)')
    parser.add_argument('--save-last', action='store_true', default=False,
                        help='Save final epoch checkpoint instead of best val loss. Pass for non-LoS datasets where curriculum masking causes monotonic loss increase.')
    # SMILE ablation control flags
    parser.add_argument('--smile-no-mnar', action='store_true',
                        help='Disable MissingPatternEncoder (curriculum mask only)')
    parser.add_argument('--smile-no-curriculum', action='store_true',
                        help='Disable curriculum masking (MNAR encoder only, random mask)')
    parser.add_argument('--smile-mask-type', choices=['all', 'temporal', 'system'], default='all',
                        help='Which masking types to include in curriculum')
    parser.add_argument('--smile-mnar-dropout', type=float, default=0.05,
                        help='MNAR dropout rate (default 0.05)')
    parser.add_argument('--obs-density-window', type=int, default=5,
                        help='Sliding window size for observation density embedding (must be odd)')
    parser.add_argument('--pretrain-mask-mode', type=str, default='fixed',
                        choices=['fixed', 'proportional_var'],
                        help='fixed=uniform ratio across variables (default); '
                             'proportional_var=PMAE per-variable ratio scaled by obs density')
    parser.add_argument('--ratio-temperature', type=float, default=1.0,
                        help='Steepness of density->ratio mapping; 1.0=linear (recommended)')
    parser.add_argument('--smile-stratified', action='store_true', default=False,
                        help='Use dataset-adaptive fixed-ratio masking with epoch-level '
                             'stratified sampling (Scheme F+D). Replaces curriculum schedule.')
    parser.add_argument('--smile-mask-weights', type=float, nargs=3, default=None,
                        metavar=('P_SYS', 'P_TEMP', 'P_RAND'),
                        help='Custom fixed masking mix (system temporal random), must sum to 1. '
                             'Overrides dataset-adaptive defaults when --smile-stratified is set.')
    args = parser.parse_args()
    if args.use_smile_lean_samepretrain:
        from models.smart import SMILELeanEncoder as Encoder
        model_name = 'smart-smile-lean-samepretrain'
    elif args.use_smile_lean:
        from models.smart import SMILELeanEncoder as Encoder
        model_name = 'smart-smile-lean'
    elif args.use_smile_v2_film:
        from models.smart import SMILEv2FiLMEncoder as Encoder
        model_name = 'smart-smile-v2-film'
    elif args.use_smile_v2:
        from models.smart import SMILEv2Encoder as Encoder
        model_name = 'smart-smile-v2'
    elif args.use_mnar:
        from models.smart import MNAREncoder as Encoder
        model_name = 'smart-mnar'
        if not args.save_last:
            args.save_last = True  # default to save-last for mnar
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
        elif args.smile_stratified:
            model_name = 'smart-smile-stratified'
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
    # Auto-enable save_last for curriculum masking models to avoid
    # monotonic val-loss increase causing epoch-1 checkpoint selection
    _uses_curriculum = (
        not args.use_mnar
        and not args.use_smile_lean_samepretrain
        and not args.smile_no_curriculum
        and not args.smile_stratified
        and (args.use_smile or args.use_smile_film or args.use_smile_lean
             or args.use_smile_v2 or args.use_smile_v2_film)
    )
    if _uses_curriculum and not args.save_last:
        args.save_last = True
    if getattr(args, 'pretrain_mask_mode', 'fixed') == 'proportional_var':
        model_name = model_name + '-pmae'
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
    system_groups = get_mask_system_groups(args.dataset, feature_names)
    log(logger, f'system_groups: { {k: len(v) for k, v in system_groups.items()} }')
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

    # PMAE v1: pre-compute per-variable observation density (one-time scan)
    PROP_MODE = getattr(args, 'pretrain_mask_mode', 'fixed')
    var_max_ratios = None
    if PROP_MODE == 'proportional_var':
        log(logger, '[PMAE] Computing variable-level observation density...')
        p_obs = compute_variable_density(train_dataloader)
        var_max_ratios = compute_var_max_ratios(
            p_obs, args.min_mask_ratio, args.max_mask_ratio,
            temperature=getattr(args, 'ratio_temperature', 1.0)).cuda()
        log(logger, f'[PMAE] p_obs  range: [{p_obs.min():.3f}, {p_obs.max():.3f}]  mean={p_obs.mean():.3f}')
        log(logger, f'[PMAE] var_max_ratios: [{var_max_ratios.min():.3f}, {var_max_ratios.max():.3f}]')

    best_auc = 0
    best_prc = 0
    best_mse = float('inf')
    ema_val = float('inf')
    # Scheme F+D: resolve dataset-adaptive ratios once before training
    if args.smile_stratified:
        if args.smile_mask_weights is not None:
            _p_sys, _p_temp, _p_rand = args.smile_mask_weights
        else:
            _ds_key = 'mimic' if args.dataset.startswith('mimic') else args.dataset
            _p_sys, _p_temp, _p_rand = _DATASET_MASK_RATIOS.get(
                _ds_key, (0.30, 0.25, 0.45))
        log(logger, f'[Stratified] P(sys)={_p_sys:.2f} P(temp)={_p_temp:.2f} '
                     f'P(rand)={_p_rand:.2f}  batches/epoch={len(train_dataloader)}')

    epoch_bar = tqdm(range(1, args.epochs + 1), desc='[Pretrain]', unit='epoch')
    for i in epoch_bar:
        train_loss = 0
        val_loss = 0
        encoder.train()
        predictor.train()
        target_encoder.eval()  # EMA target: eval mode prevents dropout noise in targets
        # Scheme D: build per-epoch stratified schedule (deterministic shuffle)
        if args.smile_stratified:
            _strat_schedule = _build_stratified_schedule(
                len(train_dataloader), _p_sys, _p_temp, _p_rand,
                epoch_seed=args.seed * 10000 + i)
        batch_bar = tqdm(train_dataloader, desc=f'  Ep{i:>3}', leave=False, unit='batch')
        for step, batch in enumerate(batch_bar, 1):
            for key in batch:
                batch[key] = batch[key].cuda()
            # Save original clinical observation pattern with progressive MNAR dropout
            mnar_drop = get_mnar_dropout_rate(i, args.epochs, args.smile_mnar_dropout)
            original_mask = apply_mnar_dropout(batch['mask'].clone(), dropout_rate=mnar_drop)
            # When ablating MNAR encoder, pass None so SMILEEncoder skips it
            enc_original_mask = None if args.smile_no_mnar else original_mask
            # smart-smile-lean-samepretrain: same pretrain as smart (random mask, no MNAR)
            if args.use_smile_lean_samepretrain:
                enc_original_mask = None
            # smart-mnar: always random masking, no curriculum
            if args.use_mnar or args.use_smile_lean_samepretrain:
                masking_fn = random_masking
            elif args.smile_no_curriculum:
                masking_fn = random_masking
            elif args.smile_stratified:
                # Scheme F+D: deterministic strategy from pre-built schedule
                masking_fn = get_stratified_masking_fn(
                    _strat_schedule[step - 1], system_groups)
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
            # PMAE v1: replace random branch with proportional masking; temporal/system unchanged
            if var_max_ratios is not None and masking_fn is random_masking:
                batch['x'], batch['mask'], pretrain_mask = proportional_random_masking(
                    batch['x'], batch['mask'],
                    args.min_mask_ratio, args.max_mask_ratio, var_max_ratios)
            else:
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
                # samepretrain: always None (no MNAR encoder, same as training)
                if (args.use_mnar or args.use_smile or args.use_smile_film or args.use_smile_v2
                        or args.use_smile_v2_film or args.use_smile_lean or args.use_smile_lean_samepretrain):
                    enc_original_mask = None if (args.smile_no_mnar or args.use_smile_lean_samepretrain) else batch['mask'].clone()
                else:
                    enc_original_mask = None
                h = target_encoder(**batch, original_mask=enc_original_mask)
                batch['labels'] = batch['x']
                if var_max_ratios is not None:
                    batch['x'], batch['mask'], pretrain_mask = proportional_random_masking(
                        batch['x'], batch['mask'],
                        args.min_mask_ratio, args.max_mask_ratio, var_max_ratios)
                else:
                    batch['x'], batch['mask'], pretrain_mask = random_masking(
                        batch['x'], batch['mask'], args.min_mask_ratio, args.max_mask_ratio)
                z = encoder(**batch, original_mask=enc_original_mask)
                z = predictor(z)
                val_loss += criterion(z[:, :, 1:], h[:, :, 1:], pretrain_mask.permute(0, 2, 1).unsqueeze(-1).expand_as(z[:, :, 1:])).item() * batch['x'].shape[0]
        t_loss = train_loss / len(train_dataset) * args.world_size
        v_loss = val_loss / len(val_dataset)
        ema_val = v_loss if i == 1 else 0.3 * v_loss + 0.7 * ema_val
        epoch_bar.set_postfix(train=f'{t_loss:.4f}', val=f'{v_loss:.4f}', ema=f'{ema_val:.4f}')
        log(logger, 'Epoch %d: Train Loss %.4f, Valid Loss %.4f, EMA Val %.4f' % (i, t_loss, v_loss, ema_val))
        cur_mse = ema_val
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
