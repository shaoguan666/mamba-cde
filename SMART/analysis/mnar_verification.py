#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MNAR Statistical Verification for C12, C19, MIMIC-III (mortality task)

Tests implemented:
  T1. Little's MCAR test       -- chi-square; rejection => NOT MCAR
  T2. Outcome-missingness      -- chi-square per variable; significance => MNAR signal
  T3. Temporal non-stationarity-- Kruskal-Wallis across early/mid/late thirds
  T4. Block missingness        -- within-system vs cross-system correlation delta
  T5. Missing indicator AUC    -- logistic regression; AUC > 0.55 => predictable missingness

Usage (run from SMART/ directory):
    python analysis/mnar_verification.py [--output-dir OUTPUT] [--no-plot]

Output:
    analysis/results/<dataset>/t1_little.csv
    analysis/results/<dataset>/t2_outcome.csv
    analysis/results/<dataset>/t3_temporal.csv
    analysis/results/<dataset>/t4_block.csv
    analysis/results/<dataset>/t5_indicator.csv
    analysis/results/<dataset>/temporal_heatmap.png
    analysis/results/mnar_summary.csv
"""

import argparse
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from scipy import stats
from scipy.stats import chi2 as chi2_dist
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# Feature metadata
# ============================================================

C19_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

# Clinical system groupings for C19
C19_SYSTEMS = {
    "Vital_Basic":    [0, 1, 2, 3, 4, 5, 6],         # HR..Resp
    "Resp_Gas":       [7, 8, 9, 10, 11, 12, 13],      # EtCO2..SaO2
    "Liver_Enzyme":   [14, 16],                        # AST, Alkalinephos
    "Renal":          [15, 18, 19],                    # BUN, Chloride, Creatinine
    "Blood_Gas_Acid": [8, 9, 11, 12],                  # BaseExcess, HCO3, pH, PaCO2
    "Electrolyte":    [21, 23, 24, 25],                # Glucose, Magnesium, Phosphate, K
    "Coagulation":    [30, 32, 33],                    # PTT, Fibrinogen, Platelets
    "CBC":            [28, 29, 31],                    # Hct, Hgb, WBC
    "Bilirubin":      [20, 26],                        # Bilirubin_direct, Bilirubin_total
}

C12_FEATURES = [
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN",
    "Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose",
    "HCO3", "HCT", "HR", "K", "Lactate", "Mg", "MAP", "MechVent",
    "Na", "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH",
    "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI",
    "TroponinT", "Urine", "WBC", "Weight",
]

C12_SYSTEMS = {
    "Vital_Invasive":   [8, 18, 30],      # DiasABP, MAP, SysABP
    "Vital_NonInvasive":[21, 22, 23],     # NIDiasABP, NIMAP, NISysABP
    "Vital_Basic":      [14, 28, 31],     # HR, RespRate, Temp
    "Blood_Gas":        [9, 24, 25, 26],  # FiO2, PaCO2, PaO2, pH
    "Electrolyte":      [15, 17, 20, 12], # K, Mg, Na, HCO3
    "Liver_Enzyme":     [0, 1, 2, 3, 4],  # Albumin, ALP, ALT, AST, Bilirubin
    "Renal":            [5, 7, 34],       # BUN, Creatinine, Urine
    "CBC":              [13, 27, 35],     # HCT, Platelets, WBC
    "Coagulation":      [32, 33],         # TroponinI, TroponinT
}

MIMIC_FEATURES = [
    "CapRefill", "DiasBP", "FiO2", "GCS_Eye", "GCS_Motor",
    "GCS_Total", "GCS_Verbal", "Glucose", "HR", "Height",
    "MeanBP", "O2Sat", "RespRate", "SysBP", "Temp", "Weight", "pH",
]

MIMIC_SYSTEMS = {
    "Vital_BP":    [1, 10, 13],           # DiasBP, MeanBP, SysBP
    "Vital_Basic": [8, 11, 12, 14],       # HR, O2Sat, RespRate, Temp
    "GCS":         [3, 4, 5, 6],          # GCS subcomponents + total
    "Resp_Gas":    [2, 11, 16],           # FiO2, O2Sat, pH
    "Anthropo":    [9, 15],               # Height, Weight
}


# ============================================================
# Data loading
# ============================================================

def load_pickle_data(pkl_path):
    """Load a SMART-format pickle. Returns (x_list, y_list, mask_list)."""
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    # Formats:
    #   4-elem: (x, y, mask, name)
    #   5-elem: (x, y, static, mask, name) OR (x, y, mask, name, split_sizes)
    if len(payload) == 4:
        x, y, mask, _name = payload
    elif len(payload) == 5:
        # Distinguish by checking whether element[2] is a list of strings (names)
        # or arrays (static/mask). If element[3] is a list of strings, it is
        # (x, y, mask, name, split_sizes); otherwise (x, y, static, mask, name).
        if isinstance(payload[3], list) and len(payload[3]) > 0 and isinstance(payload[3][0], str):
            x, y, mask, _name, _splits = payload
        else:
            x, y, _static, mask, _name = payload
    else:
        raise ValueError("Unexpected pickle format with %d elements" % len(payload))
    return x, y, mask


def build_aggregate_matrix(x_list, mask_list):
    """
    For Little's MCAR test: aggregate variable-length time series to per-patient.
    Returns:
        agg_obs  (N, V)  -- per-patient mean of observed values (NaN if never seen)
        agg_miss (N, V)  -- per-patient missingness rate in [0, 1]
        labels_np -- not returned here; provided separately
    """
    N = len(x_list)
    V = np.array(x_list[0]).shape[1]
    agg_obs = np.full((N, V), np.nan, dtype=np.float64)
    agg_miss = np.zeros((N, V), dtype=np.float64)

    for i in range(N):
        x_i = np.array(x_list[i], dtype=np.float64)   # (T_i, V)
        m_i = np.array(mask_list[i], dtype=np.float64) # (T_i, V), 1=observed

        T_i = m_i.shape[0]
        for j in range(V):
            obs_idx = m_i[:, j] == 1
            if obs_idx.any():
                agg_obs[i, j] = x_i[obs_idx, j].mean()
            agg_miss[i, j] = 1.0 - obs_idx.mean()

    return agg_obs, agg_miss


def build_3d_mask(mask_list, max_T=None):
    """
    Pad variable-length masks to (N, max_T, V) for temporal analysis.
    Padding positions are set to -1 (sentinel, excluded from analysis).
    """
    N = len(mask_list)
    V = np.array(mask_list[0]).shape[1]
    lengths = [np.array(m).shape[0] for m in mask_list]
    if max_T is None:
        max_T = max(lengths)

    arr = np.full((N, max_T, V), -1, dtype=np.float32)
    for i, m in enumerate(mask_list):
        m_arr = np.array(m, dtype=np.float32)
        T_i = min(m_arr.shape[0], max_T)
        arr[i, :T_i, :] = m_arr[:T_i]
    return arr, lengths


# ============================================================
# T1: Little's MCAR test
# ============================================================

def littles_mcar_test(agg_obs):
    """
    Little (1988) MCAR test on (N, V) matrix with NaN for missing.
    Returns (chi2_stat, p_value, df, n_patterns).
    """
    N, V = agg_obs.shape

    # Overall mean (ignoring NaN)
    mu_grand = np.nanmean(agg_obs, axis=0)  # (V,)

    # Pooled covariance from pairwise-complete observations
    sigma = np.zeros((V, V))
    for j in range(V):
        for k in range(j, V):
            both_obs = ~np.isnan(agg_obs[:, j]) & ~np.isnan(agg_obs[:, k])
            if both_obs.sum() > 1:
                cov_jk = np.cov(agg_obs[both_obs, j], agg_obs[both_obs, k])[0, 1]
            else:
                cov_jk = 0.0
            sigma[j, k] = sigma[k, j] = cov_jk
    # Ensure positive definiteness
    sigma += np.eye(V) * 1e-6

    # Identify unique missing patterns
    miss_indicator = np.isnan(agg_obs)  # (N, V), True = missing
    patterns = {}
    for i in range(N):
        key = tuple(miss_indicator[i].tolist())
        if key not in patterns:
            patterns[key] = []
        patterns[key].append(i)

    chi2_sum = 0.0
    df_sum = 0

    for key, idx_list in patterns.items():
        obs_vars = [j for j in range(V) if not key[j]]
        if len(obs_vars) == 0:
            continue

        n_k = len(idx_list)
        sub_data = agg_obs[np.ix_(idx_list, obs_vars)]  # (n_k, n_obs)
        mu_k = np.nanmean(sub_data, axis=0)             # (n_obs,)

        # Skip if any NaN in pattern mean (all missing for some var in this group)
        if np.any(np.isnan(mu_k)):
            continue

        diff = mu_k - mu_grand[obs_vars]                # (n_obs,)
        sigma_k = sigma[np.ix_(obs_vars, obs_vars)]     # (n_obs, n_obs)

        try:
            sigma_inv = np.linalg.pinv(sigma_k)
            d2 = n_k * float(diff @ sigma_inv @ diff)
            if np.isfinite(d2) and d2 >= 0:
                chi2_sum += d2
                df_sum += len(obs_vars)
        except Exception:
            continue

    # df = (sum obs_vars per pattern) - V
    df_total = max(df_sum - V, 1)
    p_value = 1.0 - chi2_dist.cdf(chi2_sum, df_total)

    return chi2_sum, p_value, df_total, len(patterns)


# ============================================================
# T2: Outcome-stratified missingness
# ============================================================

def outcome_missingness_test(agg_miss, labels, feature_names):
    """
    For each variable, test whether missingness rate differs between outcome groups.
    Uses chi-square test on (miss/obs) x (pos/neg) contingency table.
    MNAR signal: significant p-value + high odds-ratio for clinical lab tests.
    """
    N, V = agg_miss.shape
    y = np.array(labels, dtype=np.float64)

    # For multi-label or regression, use median split
    if y.ndim > 1:
        y = y[:, 0]
    if len(np.unique(y)) > 2:
        y = (y > np.median(y)).astype(float)

    pos = y == 1
    neg = y == 0
    n_pos, n_neg = pos.sum(), neg.sum()

    rows = []
    for j in range(V):
        # Binarize: >0.5 missingness = "predominantly missing"
        miss_j = (agg_miss[:, j] > 0.5).astype(int)
        a = miss_j[pos].sum()        # missing & positive
        b = int(n_pos) - a           # observed & positive
        c = miss_j[neg].sum()        # missing & negative
        d = int(n_neg) - c           # observed & negative

        miss_rate_pos = miss_j[pos].mean()
        miss_rate_neg = miss_j[neg].mean()

        if min(a + b, c + d) < 10 or min(a, b, c, d) < 3:
            chi2_stat, p_val = np.nan, np.nan
        else:
            ct = np.array([[a, b], [c, d]])
            chi2_stat, p_val, _, _ = stats.chi2_contingency(ct, correction=False)

        or_val = (a * d) / (b * c + 1e-12) if (b > 0 and c > 0) else np.nan

        rows.append({
            "feature":          feature_names[j],
            "miss_rate_pos":    round(miss_rate_pos, 4),
            "miss_rate_neg":    round(miss_rate_neg, 4),
            "diff(pos-neg)":    round(miss_rate_pos - miss_rate_neg, 4),
            "chi2":             round(chi2_stat, 3) if not np.isnan(chi2_stat) else np.nan,
            "p_value":          round(p_val, 6)     if not np.isnan(p_val)    else np.nan,
            "odds_ratio":       round(or_val, 3)    if not np.isnan(or_val)   else np.nan,
            "significant":      (p_val < 0.05)      if not np.isnan(p_val)    else False,
        })

    df_result = pd.DataFrame(rows)
    return df_result.sort_values("p_value")


# ============================================================
# T3: Temporal missingness non-stationarity
# ============================================================

def temporal_missingness_test(mask_3d, feature_names, n_splits=3):
    """
    Kruskal-Wallis test: is missingness rate constant across early/mid/late period?
    MNAR signal: non-constant missingness (KW p < 0.05).
    mask_3d: (N, T, V), -1=padded, 0=missing, 1=observed
    Returns (df_stats, miss_rate_matrix) where miss_rate_matrix is (T, V).
    """
    N, T, V = mask_3d.shape

    # Compute per-time-step missingness rate (only among non-padded entries)
    miss_rate = np.full((T, V), np.nan)
    n_valid = np.zeros(T, dtype=int)

    for t in range(T):
        valid = mask_3d[:, t, :] != -1  # (N, V)
        n_pat = valid[:, 0].sum()       # number of patients at step t
        n_valid[t] = int(n_pat)
        if n_pat == 0:
            continue
        for j in range(V):
            valid_j = mask_3d[:, t, j] != -1
            if valid_j.sum() == 0:
                continue
            miss_rate[t, j] = (mask_3d[valid_j, t, j] == 0).mean()

    # Only use time steps with >= 20 valid patients
    valid_steps = np.where(n_valid >= 20)[0]
    if len(valid_steps) < n_splits * 3:
        # Fallback: use all steps
        valid_steps = np.arange(T)

    split_size = max(1, len(valid_steps) // n_splits)
    groups_idx = [
        valid_steps[i * split_size : (i + 1) * split_size]
        for i in range(n_splits)
    ]

    rows = []
    for j in range(V):
        groups = []
        for g_idx in groups_idx:
            rates = miss_rate[g_idx, j]
            rates = rates[~np.isnan(rates)]
            groups.append(rates)

        # Need at least 2 non-empty groups with >1 observation
        valid_groups = [g for g in groups if len(g) > 1]
        if len(valid_groups) < 2:
            rows.append({"feature": feature_names[j], "kw_stat": np.nan, "kw_p": np.nan,
                         "mean_miss": np.nanmean(miss_rate[:, j]),
                         "std_miss":  np.nanstd(miss_rate[:, j]),
                         "significant": False})
            continue

        try:
            kw_stat, kw_p = stats.kruskal(*valid_groups)
        except Exception:
            kw_stat, kw_p = np.nan, np.nan

        rows.append({
            "feature":     feature_names[j],
            "kw_stat":     round(kw_stat, 3) if not np.isnan(kw_stat) else np.nan,
            "kw_p":        round(kw_p, 6)    if not np.isnan(kw_p)    else np.nan,
            "mean_miss":   round(np.nanmean(miss_rate[:, j]), 4),
            "std_miss":    round(np.nanstd(miss_rate[:, j]), 4),
            "significant": (kw_p < 0.05) if not np.isnan(kw_p) else False,
        })

    df_result = pd.DataFrame(rows)
    return df_result.sort_values("kw_p"), miss_rate


# ============================================================
# T4: Block / system-level missingness correlation
# ============================================================

def block_missingness_test(agg_miss, feature_names, system_groups):
    """
    Compute per-variable missingness correlation matrix.
    Compare mean within-system vs cross-system correlation.
    MNAR signal: within-system >> cross-system (delta >> 0).
    agg_miss: (N, V) missingness rates
    """
    V = agg_miss.shape[1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = np.corrcoef(agg_miss.T)  # (V, V)

    rows = []
    for sys_name, var_ids in system_groups.items():
        var_ids = [v for v in var_ids if v < V]
        if len(var_ids) < 2:
            continue

        other_vars = [v for s, vs in system_groups.items()
                      if s != sys_name for v in vs if v < V]

        within = [corr[i, j] for idx_i, i in enumerate(var_ids)
                  for j in var_ids[idx_i + 1:]
                  if not np.isnan(corr[i, j])]

        cross = [corr[i, j] for i in var_ids for j in other_vars
                 if not np.isnan(corr[i, j])]

        mean_within = float(np.mean(within)) if within else np.nan
        mean_cross  = float(np.mean(cross))  if cross  else np.nan
        delta = (mean_within - mean_cross) if not (np.isnan(mean_within) or np.isnan(mean_cross)) else np.nan

        # Mann-Whitney U between within and cross distributions
        if within and cross and len(within) >= 3 and len(cross) >= 3:
            u_stat, mw_p = stats.mannwhitneyu(within, cross, alternative="greater")
        else:
            u_stat, mw_p = np.nan, np.nan

        rows.append({
            "system":            sys_name,
            "n_vars":            len(var_ids),
            "n_within_pairs":    len(within),
            "n_cross_pairs":     len(cross),
            "mean_within_corr":  round(mean_within, 4) if not np.isnan(mean_within) else np.nan,
            "mean_cross_corr":   round(mean_cross, 4)  if not np.isnan(mean_cross)  else np.nan,
            "delta":             round(delta, 4)        if not np.isnan(delta)       else np.nan,
            "mw_p":              round(mw_p, 6)         if not np.isnan(mw_p)        else np.nan,
            "significant":       (mw_p < 0.05)          if not np.isnan(mw_p)        else False,
        })

    return pd.DataFrame(rows).sort_values("delta", ascending=False)


# ============================================================
# T5: Missing indicator regression (AUC)
# ============================================================

def missing_indicator_auc(agg_miss, agg_obs, feature_names, sample_frac=1.0, max_iter=300):
    """
    For each variable j, train logistic regression to predict R_j (is j missing?)
    from observed mean values of all other variables.
    AUC > 0.55: missingness is predictable from data -> not MCAR.
    AUC > 0.65: strong MAR/MNAR evidence.

    agg_miss: (N, V) per-patient missingness rate
    agg_obs:  (N, V) per-patient mean observed values (NaN if never observed)
    """
    N, V = agg_miss.shape

    # Subsample for speed
    if sample_frac < 1.0:
        n_use = max(200, int(N * sample_frac))
        idx = np.random.choice(N, n_use, replace=False)
        agg_miss = agg_miss[idx]
        agg_obs  = agg_obs[idx]
        N = n_use

    # Build feature matrix: impute NaN with column mean
    obs_imputed = agg_obs.copy()
    col_means = np.nanmean(obs_imputed, axis=0)
    for j in range(V):
        nan_mask = np.isnan(obs_imputed[:, j])
        obs_imputed[nan_mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0.0

    scaler = StandardScaler()
    obs_scaled = scaler.fit_transform(obs_imputed)

    rows = []
    for j in range(V):
        target = (agg_miss[:, j] > 0.5).astype(int)  # binary: predominantly missing
        miss_rate = float(target.mean())

        # Skip near-constant targets
        if miss_rate < 0.03 or miss_rate > 0.97:
            rows.append({"feature": feature_names[j], "auc": np.nan,
                         "miss_rate": round(miss_rate, 4), "note": "near-constant"})
            continue

        # Features: all variables except j
        feat_idx = [k for k in range(V) if k != j]
        X = obs_scaled[:, feat_idx]

        try:
            lr = LogisticRegression(C=0.5, max_iter=max_iter, random_state=42,
                                    solver="lbfgs", n_jobs=1)
            lr.fit(X, target)
            proba = lr.predict_proba(X)[:, 1]
            auc = float(roc_auc_score(target, proba))
        except Exception as e:
            auc = np.nan

        rows.append({
            "feature":    feature_names[j],
            "auc":        round(auc, 4) if not np.isnan(auc) else np.nan,
            "miss_rate":  round(miss_rate, 4),
            "note":       "mnar_signal" if (not np.isnan(auc) and auc >= 0.60) else "",
        })

    return pd.DataFrame(rows).sort_values("auc", ascending=False)


# ============================================================
# Visualization helpers
# ============================================================

def plot_temporal_heatmap(miss_rate, feature_names, title, out_path):
    """
    Heatmap of missingness rate over time: rows=variables, cols=time steps.
    """
    # Transpose: rows=features, cols=time
    data = miss_rate.T  # (V, T)
    V, T = data.shape

    # Trim trailing all-NaN columns
    last_valid = T
    for t in range(T - 1, -1, -1):
        if not np.all(np.isnan(data[:, t])):
            last_valid = t + 1
            break
    data = data[:, :last_valid]

    fig, ax = plt.subplots(figsize=(min(16, last_valid // 3 + 4), max(4, V // 3 + 2)))
    masked_data = np.ma.masked_invalid(data)
    cmap = plt.cm.RdYlGn_r
    cmap.set_bad(color="lightgrey")
    im = ax.imshow(masked_data, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_yticks(range(V))
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Variable")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label="Missingness rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_outcome_missingness(df_t2, title, out_path, top_n=15):
    """
    Bar chart: missingness rate by outcome for top-N most differentiated variables.
    """
    df = df_t2.dropna(subset=["p_value"]).head(top_n).copy()
    if df.empty:
        return

    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, df["miss_rate_pos"], width, label="Positive outcome",
           color="#e74c3c", alpha=0.8)
    ax.bar(x + width / 2, df["miss_rate_neg"], width, label="Negative outcome",
           color="#2ecc71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["feature"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Missingness rate (>50% threshold)")
    ax.set_title(title, fontsize=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ============================================================
# New visualizations: co-occurrence heatmap, outcome bar, temporal curves
# ============================================================

def _build_system_order(feature_names, system_groups):
    """
    Reorder variable indices so that variables belonging to the same
    physiological system are adjacent.  Variables not in any group are
    appended at the end.
    """
    V = len(feature_names)
    used = set()
    ordered_idx = []
    group_boundaries = []  # (start, end, system_name)

    for sys_name, var_ids in system_groups.items():
        ids = [v for v in var_ids if v < V and v not in used]
        if not ids:
            continue
        start = len(ordered_idx)
        ordered_idx.extend(ids)
        used.update(ids)
        group_boundaries.append((start, len(ordered_idx), sys_name))

    # Append remaining variables
    for v in range(V):
        if v not in used:
            ordered_idx.append(v)

    return ordered_idx, group_boundaries


def plot_cooccurrence_heatmap(agg_miss, feature_names, system_groups,
                              title, out_path):
    """
    T4 visualization: V x V Spearman correlation heatmap of missingness,
    with variables ordered by physiological system so that block structures
    are visually apparent along the diagonal.
    """
    V = agg_miss.shape[1]
    ordered_idx, group_boundaries = _build_system_order(feature_names,
                                                        system_groups)

    # Reorder missingness matrix columns
    miss_ordered = agg_miss[:, ordered_idx]
    ordered_names = [feature_names[i] for i in ordered_idx]

    # Spearman correlation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_mat, _ = spearmanr(miss_ordered)
    if corr_mat.ndim == 0:
        corr_mat = np.array([[corr_mat]])

    fig, ax = plt.subplots(figsize=(max(8, V * 0.32), max(7, V * 0.30)))
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr_mat, cmap=cmap, vmin=-1, vmax=1,
                   interpolation="nearest", aspect="equal")

    # Tick labels
    ax.set_xticks(range(V))
    ax.set_xticklabels(ordered_names, rotation=90, fontsize=6)
    ax.set_yticks(range(V))
    ax.set_yticklabels(ordered_names, fontsize=6)

    # Draw block boundaries
    for start, end, sys_name in group_boundaries:
        rect = plt.Rectangle((start - 0.5, start - 0.5),
                              end - start, end - start,
                              linewidth=2, edgecolor="black",
                              facecolor="none")
        ax.add_patch(rect)
        # Label in margin
        mid = (start + end) / 2
        ax.annotate(sys_name, xy=(V + 0.5, mid), fontsize=5,
                    va="center", ha="left",
                    annotation_clip=False)

    ax.set_title(title, fontsize=11, pad=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.12)
    cbar.set_label("Spearman corr", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_outcome_missrate_comparison(agg_miss, labels, feature_names,
                                      title, out_path, top_n=12):
    """
    T2 visualization: grouped bar chart showing *observation frequency*
    (1 - missingness rate) for Positive vs Negative outcome patients.
    Selects the top_n variables with the largest absolute difference.
    """
    N, V = agg_miss.shape
    y = np.array(labels, dtype=np.float64)
    if y.ndim > 1:
        y = y[:, 0]
    if len(np.unique(y)) > 2:
        y = (y > np.median(y)).astype(float)

    pos_mask = y == 1
    neg_mask = y == 0

    obs_rate_pos = []
    obs_rate_neg = []
    diffs = []
    for j in range(V):
        r_pos = 1.0 - float(agg_miss[pos_mask, j].mean())
        r_neg = 1.0 - float(agg_miss[neg_mask, j].mean())
        obs_rate_pos.append(r_pos)
        obs_rate_neg.append(r_neg)
        diffs.append(abs(r_pos - r_neg))

    # Select top_n by absolute difference
    rank = np.argsort(diffs)[::-1][:top_n]
    sel_names = [feature_names[i] for i in rank]
    sel_pos = [obs_rate_pos[i] for i in rank]
    sel_neg = [obs_rate_neg[i] for i in rank]

    x = np.arange(len(sel_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(sel_names) * 0.8), 5))
    bars_neg = ax.bar(x - width / 2, sel_neg, width,
                      label="Negative (survive)", color="#3498db", alpha=0.85)
    bars_pos = ax.bar(x + width / 2, sel_pos, width,
                      label="Positive (critical/death)", color="#e74c3c",
                      alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(sel_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Observation Frequency (1 - miss rate)")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # Add significance stars
    for idx, ri in enumerate(rank):
        diff_val = obs_rate_pos[ri] - obs_rate_neg[ri]
        if abs(diffs[ri]) > 0.05:
            higher = max(sel_pos[idx], sel_neg[idx])
            ax.annotate("***" if abs(diffs[ri]) > 0.10 else "**",
                        xy=(x[idx], higher + 0.02),
                        ha="center", fontsize=7, color="red")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_temporal_missrate_curves(mask_3d, feature_names, system_groups,
                                  title, out_path, max_vars=8):
    """
    T3 visualization: line plot of average observation frequency over time
    (hours since admission) for selected representative variables.
    Shows admission spike, decay, and possible 24h periodicity.
    """
    N, T, V = mask_3d.shape

    # Compute observation rate per time step
    obs_rate = np.full((T, V), np.nan)
    for t in range(T):
        for j in range(V):
            valid = mask_3d[:, t, j] != -1
            n_valid = valid.sum()
            if n_valid < 10:
                continue
            obs_rate[t, j] = (mask_3d[valid, t, j] == 1).mean()

    # Find last meaningful time step
    last_t = T
    for t in range(T - 1, -1, -1):
        if not np.all(np.isnan(obs_rate[t, :])):
            last_t = t + 1
            break
    obs_rate = obs_rate[:last_t, :]

    # Select representative variables: pick ones with highest temporal variance
    temporal_var = np.nanvar(obs_rate, axis=0)
    # Also pick from different systems for diversity
    selected = set()
    # First: top by variance
    var_rank = np.argsort(temporal_var)[::-1]
    for vi in var_rank:
        if len(selected) >= max_vars:
            break
        if not np.isnan(temporal_var[vi]):
            selected.add(vi)

    selected = sorted(selected)

    # Color map
    cmap = plt.cm.tab10
    colors = [cmap(i / max(1, len(selected) - 1)) for i in range(len(selected))]

    fig, ax = plt.subplots(figsize=(12, 5))
    time_hours = np.arange(last_t)

    for idx, vi in enumerate(selected):
        curve = obs_rate[:, vi]
        valid_t = ~np.isnan(curve)
        ax.plot(time_hours[valid_t], curve[valid_t],
                label=feature_names[vi], color=colors[idx],
                linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Time Step (hours since admission)", fontsize=10)
    ax.set_ylabel("Observation Frequency", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    # Mark 24h boundaries if time range is long enough
    for h in range(24, last_t, 24):
        ax.axvline(x=h, color="grey", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.text(h, 1.02, "%dh" % h, ha="center", fontsize=7, color="grey")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _compute_obs_rate(mask_3d_sub):
    """Compute per-timestep observation rate from a (N, T, V) mask array."""
    N, T, V = mask_3d_sub.shape
    obs_rate = np.full((T, V), np.nan)
    for t in range(T):
        for j in range(V):
            valid = mask_3d_sub[:, t, j] != -1
            n_valid = valid.sum()
            if n_valid < 10:
                continue
            obs_rate[t, j] = (mask_3d_sub[valid, t, j] == 1).mean()
    return obs_rate


def plot_temporal_curves_by_outcome(mask_3d, labels, feature_names,
                                    system_groups, title_prefix, out_path,
                                    max_vars=8):
    """
    Plot observation frequency over time separately for positive and negative
    patients. Produces vertically stacked Positive / Negative panels.
    """
    N, T, V = mask_3d.shape
    y = np.array(labels, dtype=np.float64)
    if y.ndim > 1:
        y = y[:, 0]
    if len(np.unique(y)) > 2:
        y = (y > np.median(y)).astype(float)

    pos_mask = y == 1
    neg_mask = y == 0

    mask_pos = mask_3d[pos_mask]
    mask_neg = mask_3d[neg_mask]

    obs_rate_pos = _compute_obs_rate(mask_pos)
    obs_rate_neg = _compute_obs_rate(mask_neg)

    # Find last meaningful time step (union of both groups)
    def _find_last_t(obs_rate):
        for t in range(obs_rate.shape[0] - 1, -1, -1):
            if not np.all(np.isnan(obs_rate[t, :])):
                return t + 1
        return obs_rate.shape[0]

    last_t = max(_find_last_t(obs_rate_pos), _find_last_t(obs_rate_neg))
    obs_rate_pos = obs_rate_pos[:last_t, :]
    obs_rate_neg = obs_rate_neg[:last_t, :]

    # Select variables: highest temporal variance across both groups
    combined = np.nanmean(
        np.stack([
            np.nan_to_num(obs_rate_pos, nan=0.0),
            np.nan_to_num(obs_rate_neg, nan=0.0),
        ]),
        axis=0,
    )
    temporal_var = np.nanvar(combined, axis=0)
    var_rank = np.argsort(temporal_var)[::-1]
    selected = []
    for vi in var_rank:
        if len(selected) >= max_vars:
            break
        if not np.isnan(temporal_var[vi]):
            selected.append(vi)
    selected = sorted(selected)

    cmap = plt.cm.tab10
    colors = [cmap(i / max(1, len(selected) - 1)) for i in range(len(selected))]
    time_hours = np.arange(last_t)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.6), sharex=True, sharey=True)

    groups = [
        ("Positive", obs_rate_pos, axes[0]),
        ("Negative", obs_rate_neg, axes[1]),
    ]

    for group_name, obs_rate, ax in groups:
        for idx, vi in enumerate(selected):
            curve = obs_rate[:, vi]
            valid_t = ~np.isnan(curve)
            if valid_t.sum() == 0:
                continue
            ax.plot(time_hours[valid_t], curve[valid_t],
                    label=feature_names[vi], color=colors[idx],
                    linewidth=1.5, alpha=0.85)

        ax.set_xlabel("Time Step (hours since admission)", fontsize=10)
        ax.set_ylabel("Observation Frequency", fontsize=10)
        ax.set_title("%s: %s" % (title_prefix, group_name), fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

        for h in range(24, last_t, 24):
            ax.axvline(x=h, color="grey", linestyle="--", alpha=0.4,
                       linewidth=0.8)
            ax.text(h, 1.02, "%dh" % h, ha="center", fontsize=7, color="grey")

    plt.tight_layout(h_pad=0.7)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# T3-V: Temporal volatility quantification (pos vs neg)
# ============================================================

def quantify_temporal_volatility(mask_3d, labels, feature_names,
                                 csv_path, plot_path, title_prefix):
    """
    Quantify observation-frequency temporal volatility for positive vs
    negative patients.  For each variable and each group, compute:
      - temporal_std:  std of per-step observation rate
      - cv:            coefficient of variation (std / mean)
      - range:         max - min observation rate
      - mad1:          mean |delta| of consecutive steps (jitter)
      - levene_p:      Levene's test for equality of variances (pos vs neg)
      - vol_ratio:     temporal_std_pos / temporal_std_neg

    Saves a CSV and a grouped bar chart comparing pos/neg volatility.
    """
    N, T, V = mask_3d.shape
    y = np.array(labels, dtype=np.float64)
    if y.ndim > 1:
        y = y[:, 0]
    if len(np.unique(y)) > 2:
        y = (y > np.median(y)).astype(float)

    pos_mask = y == 1
    neg_mask = y == 0

    obs_pos = _compute_obs_rate(mask_3d[pos_mask])
    obs_neg = _compute_obs_rate(mask_3d[neg_mask])

    # Trim trailing NaN
    def _trim(arr):
        for t in range(arr.shape[0] - 1, -1, -1):
            if not np.all(np.isnan(arr[t, :])):
                return arr[:t + 1, :]
        return arr

    obs_pos = _trim(obs_pos)
    obs_neg = _trim(obs_neg)

    def _vol_stats(curve):
        c = curve[~np.isnan(curve)]
        if len(c) < 3:
            return np.nan, np.nan, np.nan, np.nan
        t_std = float(np.std(c))
        t_mean = float(np.mean(c))
        cv = t_std / t_mean if t_mean > 1e-8 else np.nan
        rng = float(np.max(c) - np.min(c))
        diffs = np.abs(np.diff(c))
        mad1 = float(np.mean(diffs))
        return t_std, cv, rng, mad1

    rows = []
    for j in range(V):
        std_p, cv_p, rng_p, mad1_p = _vol_stats(obs_pos[:, j] if j < obs_pos.shape[1] else np.array([np.nan]))
        std_n, cv_n, rng_n, mad1_n = _vol_stats(obs_neg[:, j] if j < obs_neg.shape[1] else np.array([np.nan]))

        # Levene's test on per-patient per-step observation indicators
        # Use per-timestep observation rates as two samples
        curve_p = obs_pos[:, j] if j < obs_pos.shape[1] else np.array([])
        curve_n = obs_neg[:, j] if j < obs_neg.shape[1] else np.array([])
        cp = curve_p[~np.isnan(curve_p)] if len(curve_p) > 0 else np.array([])
        cn = curve_n[~np.isnan(curve_n)] if len(curve_n) > 0 else np.array([])

        if len(cp) >= 3 and len(cn) >= 3:
            lev_stat, lev_p = stats.levene(cp, cn, center="median")
        else:
            lev_stat, lev_p = np.nan, np.nan

        vol_ratio = std_p / std_n if (not np.isnan(std_n) and std_n > 1e-8) else np.nan

        rows.append({
            "feature":        feature_names[j],
            "std_pos":        round(std_p, 6) if not np.isnan(std_p) else np.nan,
            "std_neg":        round(std_n, 6) if not np.isnan(std_n) else np.nan,
            "cv_pos":         round(cv_p, 4) if not np.isnan(cv_p) else np.nan,
            "cv_neg":         round(cv_n, 4) if not np.isnan(cv_n) else np.nan,
            "range_pos":      round(rng_p, 4) if not np.isnan(rng_p) else np.nan,
            "range_neg":      round(rng_n, 4) if not np.isnan(rng_n) else np.nan,
            "mad1_pos":       round(mad1_p, 6) if not np.isnan(mad1_p) else np.nan,
            "mad1_neg":       round(mad1_n, 6) if not np.isnan(mad1_n) else np.nan,
            "vol_ratio":      round(vol_ratio, 3) if not np.isnan(vol_ratio) else np.nan,
            "levene_stat":    round(lev_stat, 3) if not np.isnan(lev_stat) else np.nan,
            "levene_p":       round(lev_p, 6) if not np.isnan(lev_p) else np.nan,
            "levene_sig":     (lev_p < 0.05) if not np.isnan(lev_p) else False,
        })

    df = pd.DataFrame(rows).sort_values("vol_ratio", ascending=False)
    df.to_csv(csv_path, index=False)

    # --- Grouped bar chart: std_pos vs std_neg ---
    df_plot = df.dropna(subset=["std_pos", "std_neg"]).copy()
    if df_plot.empty:
        return df

    x = np.arange(len(df_plot))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(df_plot) * 0.6), 9))

    # Panel 1: temporal std comparison
    ax = axes[0]
    ax.bar(x - width / 2, df_plot["std_pos"], width,
           label="Positive", color="#e74c3c", alpha=0.85)
    ax.bar(x + width / 2, df_plot["std_neg"], width,
           label="Negative", color="#3498db", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["feature"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Temporal Std of Obs Rate")
    ax.set_title("%s: Temporal Volatility (Std)" % title_prefix, fontsize=11)
    ax.legend(fontsize=8)
    for idx in range(len(df_plot)):
        sig = df_plot.iloc[idx]["levene_sig"]
        if sig:
            higher = max(df_plot.iloc[idx]["std_pos"],
                         df_plot.iloc[idx]["std_neg"])
            ax.annotate("*", xy=(x[idx], higher + 0.002),
                        ha="center", fontsize=9, color="red",
                        fontweight="bold")

    # Panel 2: volatility ratio
    ax2 = axes[1]
    ratios = df_plot["vol_ratio"].values
    colors_bar = ["#e74c3c" if r > 1.0 else "#3498db" for r in ratios]
    ax2.bar(x, ratios, 0.6, color=colors_bar, alpha=0.85)
    ax2.axhline(y=1.0, color="grey", linestyle="--", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_plot["feature"], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Vol Ratio (Pos / Neg)")
    ax2.set_title("%s: Volatility Ratio (>1 = Positive more volatile)"
                  % title_prefix, fontsize=11)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return df


# ============================================================
# Publication-grade integrated MNAR audit dashboard
# ============================================================

def _short_dataset_name(name):
    """Compact labels for dense publication figures."""
    mapping = {
        "C12_Mortality": "C12",
        "C19_Sepsis": "C19",
        "MIMIC3_Mortality": "MIMIC",
        "MIMIC3_Phenotyping": "MIMIC-Pheno.",
        "MIMIC3_Decompensation": "MIMIC-Decomp.",
        "MIMIC3_LengthOfStay": "MIMIC-LOS",
    }
    return mapping.get(name, name.replace("_", "\n"))


def _dashboard_dataset_name(name):
    """Compact multi-line labels for the legacy dashboard."""
    mapping = {
        "C19_Sepsis": "C19\nSepsis",
        "C12_Mortality": "C12\nMortality",
        "MIMIC3_Mortality": "MIMIC-III\nMortality",
        "MIMIC3_Phenotyping": "MIMIC-III\nPhenotyping",
        "MIMIC3_Decompensation": "MIMIC-III\nDecomp.",
        "MIMIC3_LengthOfStay": "MIMIC-III\nLOS",
    }
    return mapping.get(name, name.replace("_", "\n"))


def _safe_read_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _neg_log10_p(p_value, cap=50.0):
    """Stable -log10(p) with a cap for underflowed p-values."""
    try:
        p = float(p_value)
    except Exception:
        return np.nan
    if np.isnan(p):
        return np.nan
    if p <= 0:
        return cap
    return min(-np.log10(p), cap)


def _format_p_value(p_value):
    try:
        p = float(p_value)
    except Exception:
        return "n/a"
    if np.isnan(p):
        return "n/a"
    if p <= 0:
        return "p<1e-300"
    if p < 1e-4:
        return "p=%.0e" % p
    return "p=%.3f" % p


def _style_publication_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(colors="#333333", labelsize=8, width=0.8)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.8, zorder=0)


def _get_representative_dataset(summary_df, output_dir):
    """Prefer C19 for feature-rich panels; otherwise use strongest evidence."""
    if "C19_Sepsis" in set(summary_df["dataset"]):
        return "C19_Sepsis"
    ranked = summary_df.sort_values(
        ["evidence_count", "V", "N"], ascending=[False, False, False]
    )
    return str(ranked.iloc[0]["dataset"])


def plot_mnar_audit_dashboard(output_dir="analysis/results",
                              out_prefix="mnar_audit_dashboard"):
    """
    Build a single publication-grade dashboard summarizing the five MNAR
    statistical audits across all processed datasets.

    Inputs are the CSV files already produced by run_analysis:
      - mnar_summary.csv
      - <dataset>/t2_outcome.csv
      - <dataset>/t3_temporal.csv
      - <dataset>/t4_block.csv
      - <dataset>/t5_indicator.csv

    Outputs:
      - <output_dir>/<out_prefix>.png
      - <output_dir>/<out_prefix>.pdf
    """
    summary_path = os.path.join(output_dir, "mnar_summary.csv")
    summary_df = _safe_read_csv(summary_path)
    if summary_df.empty:
        print("  [WARN] Cannot create dashboard: missing %s" % summary_path)
        return None

    required_cols = ["dataset", "N", "V", "t1_p", "t2_n_significant",
                     "t3_n_nonstationary", "t4_mean_delta", "t5_mean_auc",
                     "evidence_count", "mnar_verdict"]
    missing_cols = [c for c in required_cols if c not in summary_df.columns]
    if missing_cols:
        print("  [WARN] Cannot create dashboard: missing columns %s"
              % ", ".join(missing_cols))
        return None

    datasets = summary_df["dataset"].astype(str).tolist()
    rep_ds = _get_representative_dataset(summary_df, output_dir)
    rep_dir = os.path.join(output_dir, rep_ds)

    # Publication style: clean white canvas, restrained clinical palette.
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    red = "#B2182B"
    red_mid = "#D6604D"
    blue = "#2166AC"
    blue_light = "#67A9CF"
    dark = "#222222"
    panel_bg = "#F7F7F7"

    fig = plt.figure(figsize=(18, 11), constrained_layout=False)
    outer = fig.add_gridspec(
        nrows=3, ncols=6,
        height_ratios=[1.20, 1.05, 1.15],
        width_ratios=[1, 1, 1, 1, 1, 1],
        left=0.055, right=0.985, top=0.90, bottom=0.075,
        wspace=0.55, hspace=0.72,
    )

    fig.text(
        0.055, 0.965,
        "MNAR statistical audit across clinical time-series datasets",
        fontsize=18, fontweight="bold", color=dark, ha="left", va="top",
    )
    fig.text(
        0.055, 0.935,
        "Five complementary tests reject MCAR and expose outcome-, time-, block-, and covariate-dependent missingness.",
        fontsize=10, color="#555555", ha="left", va="top",
    )

    # --------------------------------------------------
    # A. Evidence matrix
    # --------------------------------------------------
    ax_a = fig.add_subplot(outer[0, :])
    V = summary_df["V"].replace(0, np.nan).astype(float)
    t1_score = summary_df["t1_p"].apply(lambda p: _neg_log10_p(p, cap=20.0) / 20.0)
    t2_score = (summary_df["t2_n_significant"].astype(float) / V).clip(0, 1)
    t3_score = (summary_df["t3_n_nonstationary"].astype(float) / V).clip(0, 1)
    t4_score = (summary_df["t4_mean_delta"].astype(float) / 0.50).clip(0, 1)
    t5_score = ((summary_df["t5_mean_auc"].astype(float) - 0.50) / 0.35).clip(0, 1)

    evidence = np.vstack([t1_score, t2_score, t3_score, t4_score, t5_score]).T
    evidence = np.nan_to_num(evidence, nan=0.0)
    col_labels = [
        "T1\nMCAR rejected",
        "T2\nOutcome link",
        "T3\nTemporal drift",
        "T4\nBlock structure",
        "T5\nPredictability",
    ]
    row_labels = [_dashboard_dataset_name(d) for d in datasets]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "mnar_evidence", ["#F2F2F2", "#F4A582", red]
    )
    im = ax_a.imshow(evidence, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax_a.set_xticks(np.arange(len(col_labels)))
    ax_a.set_xticklabels(col_labels, fontsize=9)
    ax_a.set_yticks(np.arange(len(row_labels)))
    ax_a.set_yticklabels(row_labels, fontsize=8)
    ax_a.tick_params(length=0)
    ax_a.set_title("A. Cross-dataset evidence matrix", loc="left",
                   fontweight="bold", color=dark, pad=12)

    for i, row in summary_df.iterrows():
        vals = [
            _format_p_value(row["t1_p"]),
            "%d/%d" % (int(row["t2_n_significant"]), int(row["V"])),
            "%d/%d" % (int(row["t3_n_nonstationary"]), int(row["V"])),
            "Delta=%.2f" % float(row["t4_mean_delta"]),
            "AUC=%.2f" % float(row["t5_mean_auc"]),
        ]
        for j, txt in enumerate(vals):
            color = "white" if evidence[i, j] > 0.62 else dark
            ax_a.text(j, i, txt, ha="center", va="center",
                      fontsize=8, color=color, fontweight="bold")
        ax_a.text(
            len(col_labels) + 0.18, i,
            "%d/5  %s" % (int(row["evidence_count"]), row["mnar_verdict"]),
            ha="left", va="center", fontsize=8.5, color=red,
            fontweight="bold",
        )

    ax_a.set_xlim(-0.5, len(col_labels) + 2.4)
    for x in np.arange(-0.5, len(col_labels), 1):
        ax_a.axvline(x, color="white", linewidth=1.5)
    for y in np.arange(-0.5, len(row_labels), 1):
        ax_a.axhline(y, color="white", linewidth=1.5)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.018, pad=0.015)
    cbar.set_label("Evidence strength", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # --------------------------------------------------
    # B. T1: Little's MCAR
    # --------------------------------------------------
    ax_b = fig.add_subplot(outer[1, 0:2])
    y = np.arange(len(summary_df))
    t1_x = summary_df["t1_p"].apply(lambda p: _neg_log10_p(p, cap=50.0)).values
    ax_b.scatter(t1_x, y, s=80, color=red, edgecolor="white",
                 linewidth=1.0, zorder=3)
    ax_b.axvline(-np.log10(0.05), color="#555555", linestyle="--",
                 linewidth=1.0)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(row_labels)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 52)
    ax_b.set_xlabel("-log10(p), capped at 50")
    ax_b.set_title("B. T1: Little's MCAR test rejects randomness",
                   loc="left", fontweight="bold", color=dark)
    _style_publication_axis(ax_b)

    # --------------------------------------------------
    # C. T2: outcome-dependent missingness
    # --------------------------------------------------
    ax_c = fig.add_subplot(outer[1, 2:4])
    t2_df = _safe_read_csv(os.path.join(rep_dir, "t2_outcome.csv"))
    if not t2_df.empty and "diff(pos-neg)" in t2_df.columns:
        t2_plot = t2_df.copy()
        t2_plot["abs_diff"] = t2_plot["diff(pos-neg)"].abs()
        t2_plot = t2_plot.sort_values("abs_diff", ascending=False).head(10)
        t2_plot = t2_plot.sort_values("diff(pos-neg)")
        colors = [blue if v < 0 else red for v in t2_plot["diff(pos-neg)"]]
        ax_c.barh(np.arange(len(t2_plot)), t2_plot["diff(pos-neg)"],
                  color=colors, alpha=0.92, height=0.68, zorder=3)
        ax_c.axvline(0, color="#333333", linewidth=0.8)
        ax_c.set_yticks(np.arange(len(t2_plot)))
        ax_c.set_yticklabels(t2_plot["feature"], fontsize=8)
        ax_c.set_xlabel("Missing-rate difference, positive - negative")
        ax_c.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    else:
        ax_c.text(0.5, 0.5, "T2 CSV unavailable", transform=ax_c.transAxes,
                  ha="center", va="center", color="#777777")
    ax_c.set_title("C. T2: missingness is outcome-dependent (%s)" % rep_ds,
                   loc="left", fontweight="bold", color=dark)
    _style_publication_axis(ax_c)

    # --------------------------------------------------
    # D. T3: temporal non-stationarity
    # --------------------------------------------------
    ax_d = fig.add_subplot(outer[1, 4:6])
    frac_t3 = summary_df["t3_n_nonstationary"].astype(float) / V
    bars = ax_d.barh(y, frac_t3.values, color=red_mid, alpha=0.94,
                     height=0.62, zorder=3)
    ax_d.axvline(0.30, color="#555555", linestyle="--", linewidth=1.0)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels(row_labels)
    ax_d.invert_yaxis()
    ax_d.set_xlim(0, 1.02)
    ax_d.set_xlabel("Non-stationary variables")
    ax_d.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    for bar, n_sig, v_count in zip(bars, summary_df["t3_n_nonstationary"], summary_df["V"]):
        ax_d.text(bar.get_width() + 0.02,
                  bar.get_y() + bar.get_height() / 2,
                  "%d/%d" % (int(n_sig), int(v_count)),
                  va="center", ha="left", fontsize=8, color=dark)
    ax_d.set_title("D. T3: missingness changes over time",
                   loc="left", fontweight="bold", color=dark)
    _style_publication_axis(ax_d)

    # --------------------------------------------------
    # E. T4: block/system structure
    # --------------------------------------------------
    ax_e = fig.add_subplot(outer[2, 0:3])
    t4_df = _safe_read_csv(os.path.join(rep_dir, "t4_block.csv"))
    if not t4_df.empty and "delta" in t4_df.columns:
        t4_plot = t4_df.dropna(subset=["delta"]).sort_values(
            "delta", ascending=True
        ).tail(10)
        ax_e.barh(np.arange(len(t4_plot)), t4_plot["delta"],
                  color="#7B3294", alpha=0.88, height=0.65, zorder=3)
        ax_e.axvline(0.05, color="#555555", linestyle="--", linewidth=1.0)
        ax_e.set_yticks(np.arange(len(t4_plot)))
        ax_e.set_yticklabels(t4_plot["system"], fontsize=8)
        ax_e.set_xlabel("Within-system minus cross-system correlation")
        for i, (_, row) in enumerate(t4_plot.iterrows()):
            mark = "*" if bool(row.get("significant", False)) else ""
            ax_e.text(float(row["delta"]) + 0.015, i, mark,
                      va="center", ha="left", fontsize=12,
                      color=red, fontweight="bold")
    else:
        ax_e.text(0.5, 0.5, "T4 CSV unavailable", transform=ax_e.transAxes,
                  ha="center", va="center", color="#777777")
    ax_e.set_title("E. T4: missingness clusters by clinical system (%s)" % rep_ds,
                   loc="left", fontweight="bold", color=dark)
    _style_publication_axis(ax_e)

    # --------------------------------------------------
    # F. T5: missing-indicator predictability
    # --------------------------------------------------
    ax_f = fig.add_subplot(outer[2, 3:6])
    auc_data = []
    auc_labels = []
    for ds in datasets:
        t5_df = _safe_read_csv(os.path.join(output_dir, ds, "t5_indicator.csv"))
        if t5_df.empty or "auc" not in t5_df.columns:
            continue
        vals = t5_df["auc"].dropna().astype(float).values
        if len(vals) == 0:
            continue
        auc_data.append(vals)
        auc_labels.append(_dashboard_dataset_name(ds))

    if auc_data:
        positions = np.arange(1, len(auc_data) + 1)
        vp = ax_f.violinplot(auc_data, positions=positions, widths=0.74,
                             showmeans=False, showmedians=False,
                             showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(blue_light)
            body.set_edgecolor("#134B73")
            body.set_alpha(0.55)
            body.set_linewidth(0.8)
        for pos, vals in zip(positions, auc_data):
            jitter = np.linspace(-0.14, 0.14, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax_f.scatter(np.full(len(vals), pos) + jitter, vals,
                         s=18, color=blue, alpha=0.65, edgecolor="white",
                         linewidth=0.3, zorder=3)
            med = np.median(vals)
            ax_f.plot([pos - 0.25, pos + 0.25], [med, med],
                      color=dark, linewidth=1.8, zorder=4)
        ax_f.axhline(0.60, color=red, linestyle="--", linewidth=1.1)
        ax_f.text(0.55, 0.612, "AUC=0.60", fontsize=8, color=red)
        ax_f.set_xticks(positions)
        ax_f.set_xticklabels(auc_labels, fontsize=8)
        ax_f.set_ylim(0.45, 0.92)
        ax_f.set_ylabel("Indicator-prediction AUC")
    else:
        ax_f.text(0.5, 0.5, "T5 CSV unavailable", transform=ax_f.transAxes,
                  ha="center", va="center", color="#777777")
    ax_f.set_title("F. T5: missingness is predictable from observed covariates",
                   loc="left", fontweight="bold", color=dark)
    _style_publication_axis(ax_f)

    # Panel background boxes for a subtle journal-style finish.
    for ax in [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]:
        bbox = ax.get_position()
        rect = plt.Rectangle(
            (bbox.x0 - 0.008, bbox.y0 - 0.012),
            bbox.width + 0.016, bbox.height + 0.030,
            transform=fig.transFigure, facecolor=panel_bg,
            edgecolor="#E1E1E1", linewidth=0.8, zorder=-1,
        )
        fig.patches.append(rect)

    fig.text(
        0.055, 0.028,
        "T1: Little's MCAR test. T2: outcome-stratified missingness. "
        "T3: Kruskal-Wallis temporal non-stationarity. "
        "T4: within-system vs cross-system missingness correlation. "
        "T5: logistic prediction of missing indicators.",
        fontsize=8, color="#666666", ha="left", va="bottom",
    )

    png_path = os.path.join(output_dir, out_prefix + ".png")
    pdf_path = os.path.join(output_dir, out_prefix + ".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


FIGURE2_DATASET_ORDER = [
    "C12_Mortality",
    "C19_Sepsis",
    "MIMIC3_Mortality",
]

FIGURE2_DATASET_CONFIG = {
    "C12_Mortality": {
        "pkl_path": "./data/Challenge2012/data_normalized.pkl",
        "features": C12_FEATURES,
        "systems": C12_SYSTEMS,
        "max_t": 48,
    },
    "C19_Sepsis": {
        "pkl_path": "./data/Challenge2019/data_normalized.pkl",
        "features": C19_FEATURES,
        "systems": C19_SYSTEMS,
        "max_t": 60,
    },
    "MIMIC3_Mortality": {
        "pkl_path": "./data/MIMIC-III/mortality_normalized.pkl",
        "features": MIMIC_FEATURES,
        "systems": MIMIC_SYSTEMS,
        "max_t": 48,
    },
}

FIGURE2_SYSTEM_COLORS = [
    "#92B1D9",
    "#C1D8E9",
    "#DBDDEF",
    "#F6C8B6",
    "#B9D8C2",
    "#D4D4D4",
    "#E9D7B8",
    "#B7CCD4",
    "#D6C6E1",
    "#C8D6A8",
]


def _default_paper_figure_dir():
    """Find smile-paper/figures from common SMART or repo-root launch dirs."""
    candidates = [
        os.path.abspath(os.path.join(os.getcwd(), "..", "smile-paper", "figures")),
        os.path.abspath(os.path.join(os.getcwd(), "smile-paper", "figures")),
    ]
    for candidate in candidates:
        paper_root = os.path.dirname(candidate)
        if os.path.isdir(paper_root):
            return candidate
    return None


def _trim_white_border(img, tol=0.985, pad=6):
    """Remove blank border from a rendered panel image."""
    arr = np.asarray(img)
    rgb = arr[..., :3] if arr.ndim == 3 else arr
    if rgb.dtype.kind in ("u", "i"):
        rgb = rgb.astype(np.float32) / 255.0
    nonwhite = np.any(rgb < tol, axis=-1) if rgb.ndim == 3 else (rgb < tol)
    rows = np.where(nonwhite.any(axis=1))[0]
    cols = np.where(nonwhite.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    y0 = max(int(rows[0]) - pad, 0)
    y1 = min(int(rows[-1]) + pad + 1, arr.shape[0])
    x0 = max(int(cols[0]) - pad, 0)
    x1 = min(int(cols[-1]) + pad + 1, arr.shape[1])
    return arr[y0:y1, x0:x1]


def _split_outcome_curve_panels(curves_img):
    """Split a by-outcome curve image into Positive and Negative panels."""
    height, width = curves_img.shape[:2]
    if width / float(max(height, 1)) > 2.0:
        split = width // 2
        pos_img = curves_img[:, :split]
        neg_img = curves_img[:, split:]
    else:
        split = height // 2
        pos_img = curves_img[:split, :]
        neg_img = curves_img[split:, :]
    return (
        _trim_white_border(pos_img, pad=4),
        _trim_white_border(neg_img, pad=4),
    )


def _flatten_binary_labels(y_list):
    """Flatten labels and median-split non-binary targets for exemplar selection."""
    labels = []
    for y in y_list:
        arr = np.asarray(y).ravel()
        labels.append(float(arr[0]) if arr.size else 0.0)
    labels = np.asarray(labels, dtype=np.float64)
    if len(np.unique(labels)) > 2:
        labels = (labels > np.median(labels)).astype(float)
    return labels.astype(int)


def _patient_mask_stats(mask_list, labels, max_t):
    """Compute compact temporal-mask statistics for selecting exemplars."""
    rows = []
    for idx, mask in enumerate(mask_list):
        m = np.asarray(mask, dtype=np.float32)
        if m.ndim != 2 or m.shape[0] < 24:
            continue
        m_view = m[: min(max_t, m.shape[0])]
        observed_fraction = np.nanmean(m_view, axis=1)
        early_n = min(6, len(observed_fraction))
        late_n = min(6, len(observed_fraction))
        early = float(np.nanmean(observed_fraction[:early_n]))
        late = float(np.nanmean(observed_fraction[-late_n:]))
        overall = float(np.nanmean(m_view))
        rows.append({
            "idx": idx,
            "label": int(labels[idx]),
            "length": int(m.shape[0]),
            "overall": overall,
            "early": early,
            "late": late,
            "range": float(np.nanmax(observed_fraction) -
                           np.nanmin(observed_fraction)),
            "shift": abs(early - late),
        })
    return rows


def _choose_patient_exemplar(rows, label):
    """
    Pick a readable non-degenerate exemplar with strong temporal mask structure.

    This is deterministic and avoids all-empty/all-dense corner cases, so the
    figure remains reproducible while still showing within-record heterogeneity.
    """
    candidates = [
        r for r in rows
        if r["label"] == label
        and 0.03 <= r["overall"] <= 0.75
        and r["length"] >= 24
    ]
    if not candidates:
        candidates = [
            r for r in rows
            if r["label"] == label and r["length"] >= 24
        ]
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda r: (
            (r["shift"] + 0.4 * r["range"])
            * (0.25 + min(r["overall"], 0.5))
        ),
    )


def _build_patient_panel_order(feature_names, system_groups):
    """Order variables by physiological group and return group boundaries."""
    V = len(feature_names)
    used = set()
    ordered_idx = []
    group_boundaries = []
    for sys_name, var_ids in system_groups.items():
        ids = [v for v in var_ids if v < V and v not in used]
        if not ids:
            continue
        start = len(ordered_idx)
        ordered_idx.extend(ids)
        used.update(ids)
        group_boundaries.append((start, len(ordered_idx), sys_name))

    remaining = [v for v in range(V) if v not in used]
    if remaining:
        start = len(ordered_idx)
        ordered_idx.extend(remaining)
        group_boundaries.append((start, len(ordered_idx), "Other"))
    return ordered_idx, group_boundaries


def _draw_patient_mask_panel(fig, outer_spec, mask, ordered_idx,
                             group_boundaries, max_t, show_ylabel=False):
    """Draw one patient-level mask panel with a top observed-fraction trace."""
    sub_gs = outer_spec.subgridspec(
        nrows=2,
        ncols=1,
        height_ratios=[0.28, 1.0],
        hspace=0.03,
    )
    ax_trace = fig.add_subplot(sub_gs[0])
    ax_mask = fig.add_subplot(sub_gs[1], sharex=ax_trace)

    m = np.asarray(mask, dtype=np.float32)
    last_t = min(max_t, m.shape[0])
    m = m[:last_t]
    m_ordered = m[:, ordered_idx].T
    observed_fraction = np.nanmean(m, axis=1)
    time = np.arange(last_t)

    ax_trace.plot(time, observed_fraction, color="#B85B50", lw=1.0)
    ax_trace.fill_between(time, 0, observed_fraction, color="#F6C8B6",
                          alpha=0.36, lw=0)
    ax_trace.set_ylim(0, 1.0)
    ax_trace.set_yticks([0, 1])
    ax_trace.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_trace.tick_params(axis="x", labelbottom=False, length=0)
    ax_trace.tick_params(axis="y", labelsize=5.5, pad=1)
    ax_trace.grid(axis="y", color="#E5E7EB", lw=0.4)
    for spine in ax_trace.spines.values():
        spine.set_visible(False)

    cmap = mcolors.ListedColormap(["#F4F7FA", "#2F5F98"])
    ax_mask.imshow(m_ordered, aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=0, vmax=1)
    ax_mask.set_xlim(-0.5, max_t - 0.5)
    xticks = [x for x in [0, 12, 24, 36, 48, 60] if x <= max_t]
    ax_mask.set_xticks(xticks)
    ax_mask.set_xticklabels(xticks, fontsize=5.8)
    ax_mask.set_yticks([])
    ax_mask.tick_params(axis="x", length=2, pad=1)

    for hour in range(24, max_t + 1, 24):
        ax_trace.axvline(hour - 0.5, color="#9CA3AF", ls="--",
                         lw=0.6, alpha=0.75)
        ax_mask.axvline(hour - 0.5, color="#9CA3AF", ls="--",
                        lw=0.6, alpha=0.75)

    for start, end, _sys_name in group_boundaries:
        ax_mask.axhline(start - 0.5, color="white", lw=0.65)
        ax_mask.axhline(end - 0.5, color="white", lw=0.65)

    strip_x = -2.05
    for k, (start, end, _sys_name) in enumerate(group_boundaries):
        rect = plt.Rectangle(
            (strip_x, start - 0.5),
            0.55,
            end - start,
            facecolor=FIGURE2_SYSTEM_COLORS[k % len(FIGURE2_SYSTEM_COLORS)],
            edgecolor="none",
            clip_on=False,
        )
        ax_mask.add_patch(rect)

    if show_ylabel:
        ax_mask.set_ylabel("Grouped\nvariables", fontsize=5.8)

    for spine in ax_mask.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#344054")

    return ax_trace, ax_mask


def _load_patient_mask_panels(dataset):
    """Load positive/negative patient-mask exemplars for a Figure 2 dataset."""
    cfg = FIGURE2_DATASET_CONFIG[dataset]
    x_list, y_list, mask_list = load_pickle_data(cfg["pkl_path"])
    labels = _flatten_binary_labels(y_list)
    feature_names = cfg["features"][:np.asarray(mask_list[0]).shape[1]]
    ordered_idx, group_boundaries = _build_patient_panel_order(
        feature_names,
        cfg["systems"],
    )
    stats_rows = _patient_mask_stats(mask_list, labels, cfg["max_t"])
    panels = {}
    for label in (1, 0):
        exemplar = _choose_patient_exemplar(stats_rows, label)
        if exemplar is None:
            raise ValueError("No exemplar found for %s label=%d" %
                             (dataset, label))
        panels[label] = {
            "record": exemplar,
            "mask": mask_list[exemplar["idx"]],
            "ordered_idx": ordered_idx,
            "group_boundaries": group_boundaries,
            "max_t": cfg["max_t"],
        }
    return panels


def plot_mnar_cooccurrence_temporal_grid(
    output_dir="analysis/results",
    out_prefix="figure2_mnar_audit_summary",
    paper_figure_dir=None,
):
    """
    Build the paper Figure 2 from co-missingness heatmaps and patient masks.

    Top row: three t4_cooccurrence_heatmap.png panels in a single row.
    Rows 2-3: positive/negative patient-level observation masks selected
    deterministically from the raw SMART pickle files. MIMIC uses the
    mortality task and is labeled simply MIMIC.

    Outputs:
      - <output_dir>/<out_prefix>.png
      - <output_dir>/<out_prefix>.pdf
      - <paper_figure_dir>/<out_prefix>.png/.pdf, when smile-paper exists
    """
    entries = []
    missing = []
    for dataset in FIGURE2_DATASET_ORDER:
        ds_dir = os.path.join(output_dir, dataset)
        heatmap_path = os.path.join(ds_dir, "t4_cooccurrence_heatmap.png")
        cfg = FIGURE2_DATASET_CONFIG.get(dataset)
        pkl_path = cfg["pkl_path"] if cfg is not None else None
        if (os.path.exists(heatmap_path)
                and pkl_path is not None
                and os.path.exists(pkl_path)):
            entries.append((dataset, heatmap_path))
        else:
            if not os.path.exists(heatmap_path):
                missing.append(heatmap_path)
            if pkl_path is None or not os.path.exists(pkl_path):
                missing.append(pkl_path or ("%s pickle config" % dataset))

    if not entries:
        print("  [WARN] Cannot create Figure 2 grid: no T4 heatmaps/pickle data found.")
        return None
    if missing:
        print("  [WARN] Figure 2 grid missing %d panel image(s); using available datasets."
              % len(missing))
        for path in missing[:8]:
            print("         missing: %s" % path)
        if len(missing) > 8:
            print("         ...")

    n_cols = len(entries)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(max(11.0, n_cols * 3.65), 6.95),
                     constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=3, ncols=n_cols,
        height_ratios=[1.08, 0.72, 0.72],
        left=0.018, right=0.992, top=0.985, bottom=0.062,
        wspace=0.05, hspace=0.16,
    )

    selected_records = []
    for col, (dataset, heatmap_path) in enumerate(entries):
        ax_top = fig.add_subplot(gs[0, col])

        heatmap_img = _trim_white_border(plt.imread(heatmap_path), pad=10)
        patient_panels = _load_patient_mask_panels(dataset)

        ax_top.imshow(heatmap_img)
        ax_top.set_anchor("S")
        ax_top.set_title(_short_dataset_name(dataset), fontsize=11,
                         fontweight="bold", pad=2)
        ax_top.set_axis_off()

        for row, label in enumerate((1, 0), start=1):
            panel = patient_panels[label]
            ax_trace, _ax_mask = _draw_patient_mask_panel(
                fig,
                gs[row, col],
                panel["mask"],
                panel["ordered_idx"],
                panel["group_boundaries"],
                panel["max_t"],
                show_ylabel=False,
            )
            rec = panel["record"]
            selected_records.append((dataset, label, rec))
            panel_letter = chr(ord("A") + (row - 1) * n_cols + col)
            ax_trace.set_title(
                "%s. Record %d, y=%d" % (panel_letter, rec["idx"], label),
                loc="left",
                pad=1.0,
                fontsize=7.2,
                fontweight="bold",
            )

    fig.text(0.50, 0.022, "Time step (hours since admission)",
             ha="center", fontsize=8)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#2F5F98",
                      edgecolor="none", label="Observed"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#F4F7FA",
                      edgecolor="#D0D5DD", label="Missing"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#F6C8B6",
                      edgecolor="#B85B50",
                      label="Observed feature fraction"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.992, 0.018),
        ncol=3,
        frameon=False,
        fontsize=6.4,
        handlelength=1.0,
        columnspacing=0.8,
    )

    output_paths = []
    os.makedirs(output_dir, exist_ok=True)
    for ext in (".png", ".pdf"):
        path = os.path.join(output_dir, out_prefix + ext)
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
        output_paths.append(path)

    if paper_figure_dir is None:
        paper_figure_dir = _default_paper_figure_dir()
    if paper_figure_dir is not None:
        os.makedirs(paper_figure_dir, exist_ok=True)
        for ext in (".png", ".pdf"):
            path = os.path.join(paper_figure_dir, out_prefix + ext)
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
            output_paths.append(path)

    plt.close(fig)
    for dataset, label, rec in selected_records:
        print("  Figure 2 exemplar: %s y=%d record=%d "
              "overall=%.3f early=%.3f late=%.3f range=%.3f" %
              (dataset, label, rec["idx"], rec["overall"], rec["early"],
               rec["late"], rec["range"]))
    return tuple(output_paths)

# ============================================================
# Per-dataset analysis runner
# ============================================================

def run_analysis(pkl_path, dataset_name, feature_names, system_groups, out_dir):
    """Run all 5 tests for one dataset. Returns a summary dict."""
    print("\n" + "=" * 60)
    print("Dataset: %s" % dataset_name)
    print("=" * 60)

    os.makedirs(out_dir, exist_ok=True)

    # Load data
    print("  Loading %s ..." % pkl_path)
    x_list, y_list, mask_list = load_pickle_data(pkl_path)
    N = len(x_list)
    V = np.array(x_list[0]).shape[1]
    print("  N=%d, V=%d (features=%d)" % (N, V, len(feature_names)))

    # Truncate feature_names if V < len(feature_names)
    feature_names = feature_names[:V]

    # Labels: handle multi-label and regression
    raw_labels = y_list
    if isinstance(raw_labels[0], (list, np.ndarray)):
        labels_flat = [l[0] if len(l) > 0 else 0 for l in raw_labels]
    else:
        labels_flat = [float(l) for l in raw_labels]
    labels_arr = np.array(labels_flat)

    # Aggregate statistics
    print("  Building aggregate matrices ...")
    agg_obs, agg_miss = build_aggregate_matrix(x_list, mask_list)

    overall_miss = float(np.mean(agg_miss))
    print("  Overall mean missingness rate: %.1f%%" % (overall_miss * 100))

    summary = {
        "dataset": dataset_name,
        "N": N,
        "V": V,
        "overall_miss_rate": round(overall_miss, 4),
    }

    # --------------------------------------------------
    # T1: Little's MCAR test
    # --------------------------------------------------
    print("  [T1] Little's MCAR test ...")
    chi2_stat, p_mcar, df_mcar, n_pat = littles_mcar_test(agg_obs)
    print("       chi2=%.2f, df=%d, p=%.4e, n_patterns=%d"
          % (chi2_stat, df_mcar, p_mcar, n_pat))
    verdict_mcar = "REJECT MCAR" if p_mcar < 0.05 else "cannot reject MCAR"
    print("       => %s" % verdict_mcar)

    t1_df = pd.DataFrame([{
        "chi2_stat": round(chi2_stat, 2),
        "p_value":   p_mcar,
        "df":        df_mcar,
        "n_patterns": n_pat,
        "verdict":   verdict_mcar,
    }])
    t1_df.to_csv(os.path.join(out_dir, "t1_little.csv"), index=False)

    summary.update({"t1_chi2": round(chi2_stat, 2), "t1_p": p_mcar,
                    "t1_verdict": verdict_mcar})

    # --------------------------------------------------
    # T2: Outcome-stratified missingness
    # --------------------------------------------------
    print("  [T2] Outcome-missingness test ...")
    t2_df = outcome_missingness_test(agg_miss, labels_flat, feature_names)
    n_sig_t2 = int(t2_df["significant"].sum())
    print("       %d/%d variables significant (p<0.05)" % (n_sig_t2, V))

    t2_df.to_csv(os.path.join(out_dir, "t2_outcome.csv"), index=False)
    summary["t2_n_significant"] = n_sig_t2

    plot_outcome_missingness(
        t2_df,
        title="%s: Missingness by Outcome (T2, top-15)" % dataset_name,
        out_path=os.path.join(out_dir, "t2_outcome_bar.png"),
    )

    # T2 new: observation frequency comparison (grouped bar)
    plot_outcome_missrate_comparison(
        agg_miss, labels_flat, feature_names,
        title="%s: Observation Freq by Outcome (T2)" % dataset_name,
        out_path=os.path.join(out_dir, "t2_outcome_obs_freq.png"),
    )

    # --------------------------------------------------
    # T3: Temporal non-stationarity
    # --------------------------------------------------
    print("  [T3] Temporal missingness Kruskal-Wallis ...")
    # Use at most max_T=72 for memory efficiency
    max_T = 72
    mask_3d, lengths = build_3d_mask(mask_list, max_T=max_T)
    t3_df, miss_rate_mat = temporal_missingness_test(mask_3d, feature_names)
    n_sig_t3 = int(t3_df["significant"].sum())
    print("       %d/%d variables non-stationary (p<0.05)" % (n_sig_t3, V))

    t3_df.to_csv(os.path.join(out_dir, "t3_temporal.csv"), index=False)
    summary["t3_n_nonstationary"] = n_sig_t3

    plot_temporal_heatmap(
        miss_rate_mat, feature_names,
        title="%s: Missingness Rate Over Time" % dataset_name,
        out_path=os.path.join(out_dir, "t3_temporal_heatmap.png"),
    )

    # T3 new: temporal observation frequency curves
    plot_temporal_missrate_curves(
        mask_3d, feature_names, system_groups,
        title="%s: Observation Freq Over Time (T3)" % dataset_name,
        out_path=os.path.join(out_dir, "t3_temporal_curves.png"),
    )

    # T3 new: temporal curves split by positive / negative outcome
    plot_temporal_curves_by_outcome(
        mask_3d, labels_flat, feature_names, system_groups,
        title_prefix="%s: Obs Freq Over Time (T3)" % dataset_name,
        out_path=os.path.join(out_dir, "t3_temporal_curves_by_outcome.png"),
    )

    # T3-V: quantify temporal volatility difference (pos vs neg)
    print("  [T3-V] Temporal volatility quantification ...")
    vol_df = quantify_temporal_volatility(
        mask_3d, labels_flat, feature_names,
        csv_path=os.path.join(out_dir, "t3v_temporal_volatility.csv"),
        plot_path=os.path.join(out_dir, "t3v_temporal_volatility.png"),
        title_prefix=dataset_name,
    )
    n_more_volatile = int((vol_df["vol_ratio"].dropna() > 1.0).sum())
    n_levene_sig = int(vol_df["levene_sig"].sum())
    print("       %d/%d vars: positive more volatile (ratio>1)" % (n_more_volatile, V))
    print("       %d/%d vars: Levene's test significant (p<0.05)" % (n_levene_sig, V))
    summary["t3v_n_pos_more_volatile"] = n_more_volatile
    summary["t3v_n_levene_sig"] = n_levene_sig
    summary["t3v_mean_vol_ratio"] = round(float(vol_df["vol_ratio"].dropna().mean()), 3) \
        if len(vol_df["vol_ratio"].dropna()) > 0 else np.nan

    # --------------------------------------------------
    # T4: Block missingness correlation
    # --------------------------------------------------
    print("  [T4] Block / system-level missingness ...")
    t4_df = block_missingness_test(agg_miss, feature_names, system_groups)
    if not t4_df.empty and "delta" in t4_df.columns:
        mean_delta = float(t4_df["delta"].dropna().mean())
        n_sig_t4 = int(t4_df["significant"].sum()) if "significant" in t4_df.columns else 0
        print("       mean within-cross delta=%.4f, %d/%d systems significant"
              % (mean_delta, n_sig_t4, len(t4_df)))
    else:
        mean_delta, n_sig_t4 = np.nan, 0
        print("       insufficient data for block test")

    t4_df.to_csv(os.path.join(out_dir, "t4_block.csv"), index=False)
    summary["t4_mean_delta"] = round(mean_delta, 4) if not np.isnan(mean_delta) else np.nan
    summary["t4_n_sig_systems"] = n_sig_t4

    # T4 new: co-occurrence correlation heatmap
    plot_cooccurrence_heatmap(
        agg_miss, feature_names, system_groups,
        title="%s: Missingness Co-occurrence (Spearman, T4)" % dataset_name,
        out_path=os.path.join(out_dir, "t4_cooccurrence_heatmap.png"),
    )

    # --------------------------------------------------
    # T5: Missing indicator AUC
    # --------------------------------------------------
    print("  [T5] Missing indicator regression AUC ...")
    t5_df = missing_indicator_auc(agg_miss, agg_obs, feature_names)
    valid_aucs = t5_df["auc"].dropna()
    mean_auc = float(valid_aucs.mean()) if len(valid_aucs) > 0 else np.nan
    n_mnar_signal = int((valid_aucs >= 0.60).sum())
    print("       mean AUC=%.4f, %d/%d vars with AUC>=0.60"
          % (mean_auc, n_mnar_signal, len(valid_aucs)))

    t5_df.to_csv(os.path.join(out_dir, "t5_indicator.csv"), index=False)
    summary["t5_mean_auc"] = round(mean_auc, 4) if not np.isnan(mean_auc) else np.nan
    summary["t5_n_mnar_signal"] = n_mnar_signal

    # --------------------------------------------------
    # Overall MNAR verdict
    # --------------------------------------------------
    evidence_count = 0
    if p_mcar < 0.05:
        evidence_count += 1          # T1: not MCAR
    if n_sig_t2 >= V * 0.3:
        evidence_count += 1          # T2: >30% vars show outcome correlation
    if n_sig_t3 >= V * 0.3:
        evidence_count += 1          # T3: >30% vars non-stationary
    if not np.isnan(mean_delta) and mean_delta > 0.05:
        evidence_count += 1          # T4: systematic block structure
    if not np.isnan(mean_auc) and mean_auc >= 0.60:
        evidence_count += 1          # T5: missingness predictable

    if evidence_count >= 4:
        mnar_verdict = "STRONG MNAR"
    elif evidence_count >= 3:
        mnar_verdict = "MODERATE MNAR"
    elif evidence_count >= 2:
        mnar_verdict = "WEAK MNAR / MAR"
    else:
        mnar_verdict = "MCAR / insufficient evidence"

    summary["evidence_count"] = evidence_count
    summary["mnar_verdict"] = mnar_verdict

    print("\n  MNAR Verdict: %s (evidence_count=%d/5)" % (mnar_verdict, evidence_count))
    return summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MNAR statistical verification")
    parser.add_argument("--output-dir", default="analysis/results",
                        help="Output directory (default: analysis/results)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset configurations: (pkl_path, name, features, systems)
    datasets = [
        (
            "./data/Challenge2019/data_normalized.pkl",
            "C19_Sepsis",
            C19_FEATURES,
            C19_SYSTEMS,
        ),
        (
            "./data/Challenge2012/data_normalized.pkl",
            "C12_Mortality",
            C12_FEATURES,
            C12_SYSTEMS,
        ),
        (
            "./data/MIMIC-III/mortality_normalized.pkl",
            "MIMIC3_Mortality",
            MIMIC_FEATURES,
            MIMIC_SYSTEMS,
        ),
        (
            "./data/MIMIC-III/phenotyping_normalized.pkl",
            "MIMIC3_Phenotyping",
            MIMIC_FEATURES,
            MIMIC_SYSTEMS,
        ),
        (
            "./data/MIMIC-III/decompensation_normalized.pkl",
            "MIMIC3_Decompensation",
            MIMIC_FEATURES,
            MIMIC_SYSTEMS,
        ),
        (
            "./data/MIMIC-III/lengthofstay_normalized.pkl",
            "MIMIC3_LengthOfStay",
            MIMIC_FEATURES,
            MIMIC_SYSTEMS,
        ),
    ]

    summaries = []
    for pkl_path, ds_name, features, systems in datasets:
        if not os.path.exists(pkl_path):
            print("\n[SKIP] %s not found: %s" % (ds_name, pkl_path))
            continue
        out_dir = os.path.join(args.output_dir, ds_name)
        try:
            summary = run_analysis(pkl_path, ds_name, features, systems, out_dir)
            summaries.append(summary)
        except Exception as e:
            print("  [ERROR] %s: %s" % (ds_name, e))
            import traceback
            traceback.print_exc()

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = os.path.join(args.output_dir, "mnar_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(summary_df.to_string(index=False))
        print("\nSaved to: %s" % summary_path)

        if not args.no_plot:
            dashboard_paths = plot_mnar_audit_dashboard(args.output_dir)
            if dashboard_paths is not None:
                print("Dashboard saved to:")
                print("  %s" % dashboard_paths[0])
                print("  %s" % dashboard_paths[1])
            figure2_paths = plot_mnar_cooccurrence_temporal_grid(args.output_dir)
            if figure2_paths is not None:
                print("Figure 2 replacement saved to:")
                for path in figure2_paths:
                    print("  %s" % path)
    else:
        print("\nNo datasets processed. Check pickle file paths.")
        print("Expected paths (relative to SMART/):")
        for pkl_path, ds_name, _, _ in datasets:
            print("  %s  ->  %s" % (ds_name, pkl_path))


if __name__ == "__main__":
    main()
