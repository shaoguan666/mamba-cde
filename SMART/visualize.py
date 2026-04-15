"""
Visualization toolkit for SMART-FiLM (CCF-B paper figures).

Generates 4 publication-quality plots:
  1. ROC + PR curves (saved to figs/roc_pr.pdf)
  2. FiLM gamma heatmap across time (figs/film_heatmap.pdf)
  3. t-SNE of CLS-token embeddings, colored by label (figs/tsne.pdf)
  4. Performance vs. missing rate (figs/missing_rate.pdf) [optional]

Usage:
    cd SMART/
    python visualize.py --dataset c19 --checkpoint ./export/c19/smart/checkpoint-prc.pth
    python visualize.py --dataset mimic_mortality --checkpoint ./export/mimic_mortality/smart/checkpoint-prc.pth
    # skip missing-rate (slow):
    python visualize.py --dataset c19 --checkpoint ./export/c19/smart/checkpoint-prc.pth --no_missing_rate
"""
import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.manifold import TSNE
from sklearn import metrics as sk_metrics

from data.challenge2012 import load_challenge_2012
from data.challenge2019 import load_challenge_2019
from data.mimiciii import (
    load_mimic_iii_mortality, load_mimic_iii_phenotyping,
    load_mimic_iii_decompensation, load_mimic_iii_lengthofstay,
)
from data.dataloader import collate_fn
from models.smart import Classifier
from utils.utils import set_seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log(msg):
    print(f'[viz] {msg}')


def load_model_and_data(args):
    """Load dataset, encoder and classifier from checkpoint."""
    if args.dataset == 'c12':
        args.input_dim, args.demo_dim = 37, 4
        args.num_class, args.max_len  = 2, 48
        _, _, test_dataset = load_challenge_2012()
        task = 'binary'
    elif args.dataset == 'c19':
        args.input_dim, args.demo_dim = 34, 5
        args.num_class, args.max_len  = 2, 60
        _, _, test_dataset = load_challenge_2019()
        task = 'binary'
    elif args.dataset == 'mimic_mortality':
        args.input_dim, args.demo_dim = 17, 0
        args.num_class, args.max_len  = 2, 48
        _, _, test_dataset = load_mimic_iii_mortality()
        task = 'binary'
    elif args.dataset == 'mimic_phenotyping':
        args.input_dim, args.demo_dim = 17, 0
        args.num_class, args.max_len  = 25, 48
        _, _, test_dataset = load_mimic_iii_phenotyping()
        task = 'multilabel'
    elif args.dataset == 'mimic_decompensation':
        args.input_dim, args.demo_dim = 17, 0
        args.num_class, args.max_len  = 2, 24
        _, _, test_dataset = load_mimic_iii_decompensation()
        task = 'binary'
    elif args.dataset == 'mimic_lengthofstay':
        args.input_dim, args.demo_dim = 17, 0
        args.num_class, args.max_len  = 1, 24
        _, _, test_dataset = load_mimic_iii_lengthofstay()
        task = 'regression'
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    if args.use_smile:
        from models.smart import SMILEEncoder as Encoder
    elif args.use_film:
        from models.smart import TimeFiLMEncoder as Encoder
    else:
        from models.smart import Encoder
    encoder    = Encoder(args).cuda()
    classifier = Classifier(args).cuda()

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location='cuda')
    # strip DDP prefix if present
    def strip_prefix(state):
        return {k.replace('module.', ''): v for k, v in state.items()}
    encoder.load_state_dict(strip_prefix(ckpt['encoder']))
    classifier.load_state_dict(strip_prefix(ckpt['classifier']))
    encoder.eval()
    classifier.eval()
    log(f'Loaded checkpoint from epoch {ckpt["epoch"]}')

    loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        sampler=SequentialSampler(test_dataset),
        collate_fn=collate_fn,
    )
    return encoder, classifier, loader, task


# ---------------------------------------------------------------------------
# 1. ROC + PR curves
# ---------------------------------------------------------------------------
def plot_roc_pr(labels_all, preds_all, out_path, title='SMART-FiLM'):
    """Plot ROC and PR curves side by side."""
    y_true  = np.array(labels_all)
    y_score = preds_all[:, 1] if preds_all.ndim == 2 else preds_all

    fpr, tpr, _  = sk_metrics.roc_curve(y_true, y_score)
    auroc        = sk_metrics.roc_auc_score(y_true, y_score)
    prec, rec, _ = sk_metrics.precision_recall_curve(y_true, y_score)
    auprc        = sk_metrics.auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUROC = {auroc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title} - ROC Curve')
    ax.legend(loc='lower right')

    ax = axes[1]
    baseline = y_true.mean()
    ax.plot(rec, prec, color='darkorange', lw=2, label=f'AUPRC = {auprc:.4f}')
    ax.axhline(baseline, color='k', linestyle='--', lw=1,
               label=f'Baseline = {baseline:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{title} - PR Curve')
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f'Saved: {out_path}  (AUROC={auroc:.4f}, AUPRC={auprc:.4f})')


# ---------------------------------------------------------------------------
# 2. FiLM gamma heatmap
# ---------------------------------------------------------------------------
def plot_film_heatmap(encoder, loader, args, out_path, n_patients=4):
    """
    Collect FiLM gamma from each block's VarAtt and MLP generators.
    Plot as a heatmap: rows = layers, cols = time steps.
    Shows positive and negative patients side by side.
    """
    # Register forward hooks on all film_gen layers
    n_blocks = len(encoder.blocks)
    # Each block has 2 film_gen: var_att_block.film_gen, mlp.film_gen
    film_captures = []   # list of (name, tensor)

    hooks = []
    for i, block in enumerate(encoder.blocks):
        def make_hook(layer_name):
            def hook_fn(module, inp, out):
                # out: (B, T+1, 2*D) -- split gamma/beta
                gamma = out[..., :out.shape[-1] // 2]   # (B, T+1, D)
                # mean over feature dim, keep (B, T+1)
                film_captures.append((layer_name, gamma.mean(-1).detach().cpu()))
            return hook_fn
        hooks.append(block.var_att_block.film_gen.register_forward_hook(
            make_hook(f'Block{i+1}-VarAtt')))
        hooks.append(block.mlp.film_gen.register_forward_hook(
            make_hook(f'Block{i+1}-MLP')))

    # Find a few positive and negative patients
    pos_gammas = []   # each: list of (n_rows, T+1)
    neg_gammas = []

    with torch.no_grad():
        for batch in loader:
            for key in batch:
                batch[key] = batch[key].cuda()
            labels = batch['labels'].cpu().numpy()

            for b_idx in range(batch['x'].shape[0]):
                if len(pos_gammas) >= n_patients and len(neg_gammas) >= n_patients:
                    break
                film_captures.clear()
                single = {k: v[b_idx:b_idx+1] for k, v in batch.items()}
                original_mask = single['mask'] if args.use_film else None
                encoder(**single, original_mask=original_mask)   # triggers hooks

                # Collect: ordered by layer
                gammas = np.stack([c[1][0].numpy() for c in film_captures], axis=0)
                # gammas: (n_layers*2, T+1)
                label_val = int(labels[b_idx]) if labels[b_idx].ndim == 0 \
                            else int(labels[b_idx].max())
                if label_val == 1 and len(pos_gammas) < n_patients:
                    pos_gammas.append(gammas)
                elif label_val == 0 and len(neg_gammas) < n_patients:
                    neg_gammas.append(gammas)

            if len(pos_gammas) >= n_patients and len(neg_gammas) >= n_patients:
                break

    for h in hooks:
        h.remove()

    if not pos_gammas or not neg_gammas:
        log('Not enough patients for FiLM heatmap, skipping.')
        return

    # Average across collected patients
    pos_avg = np.mean([g[:, 1:] for g in pos_gammas], axis=0)  # skip cls t=0
    neg_avg = np.mean([g[:, 1:] for g in neg_gammas], axis=0)

    layer_names = [c[0] for c in film_captures[-2*n_blocks:]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    vmin = min(pos_avg.min(), neg_avg.min())
    vmax = max(pos_avg.max(), neg_avg.max())

    for ax, data, subtitle in zip(axes,
                                   [pos_avg, neg_avg],
                                   ['Positive (label=1)', 'Negative (label=0)']):
        im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_yticks(range(len(layer_names)))
        ax.set_yticklabels(layer_names, fontsize=8)
        ax.set_xlabel('Time step')
        ax.set_title(f'FiLM gamma - {subtitle}')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('FiLM Temporal Modulation (gamma, averaged over features)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# 3. t-SNE of CLS-token embeddings
# ---------------------------------------------------------------------------
def plot_tsne(encoder, loader, out_path, max_samples=2000, use_smile=False):
    """
    Extract CLS-token embedding h[:,0,:] averaged over variables,
    run t-SNE, and plot colored by ground-truth label.
    """
    all_emb    = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            for key in batch:
                batch[key] = batch[key].cuda()
            original_mask = batch['mask'] if use_smile else None
            h      = encoder(**batch, original_mask=original_mask)              # (B, V, T+1, H)
            cls    = h[:, :, 0, :].mean(dim=1)    # (B, H): mean over variables
            labels = batch['labels'].cpu().numpy()
            all_emb.append(cls.cpu().numpy())
            if labels.ndim == 1:
                all_labels.extend(labels.tolist())
            else:
                all_labels.extend(labels.max(-1).tolist())
            if len(all_labels) >= max_samples:
                break

    emb    = np.concatenate(all_emb, axis=0)[:max_samples]
    labels = np.array(all_labels[:max_samples])

    log(f't-SNE: fitting {len(emb)} samples...')
    tsne   = TSNE(n_components=2, random_state=42, perplexity=40,
                  n_iter=1000, init='pca')
    coords = tsne.fit_transform(emb)

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(i)], s=8, alpha=0.6,
                   label=f'label={int(lbl)}' if len(unique_labels) <= 2
                         else f'class {int(lbl)}')
    ax.set_title('t-SNE of CLS-token Embeddings (SMART-FiLM)')
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    if len(unique_labels) <= 10:
        ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# 4. Performance vs. missing rate
# ---------------------------------------------------------------------------
def plot_missing_rate(encoder, classifier, loader, args, out_path,
                      rates=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
    """
    Evaluate AUROC at different simulated missing rates.
    Missing rate is applied by randomly zeroing observed values in the mask.
    """
    use_smile = getattr(args, 'use_smile', False)
    results = []
    for rate in rates:
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                for key in batch:
                    batch[key] = batch[key].cuda()
                # simulate additional missingness
                if rate > 0:
                    drop_mask = torch.bernoulli(
                        torch.full_like(batch['mask'], 1.0 - rate)
                    )
                    batch['x']    = batch['x'] * drop_mask
                    batch['mask'] = batch['mask'] * drop_mask
                original_mask = batch['mask'] if use_smile else None
                h     = encoder(**batch, original_mask=original_mask)
                preds = classifier(h, **batch)
                all_preds.append(preds.softmax(-1).cpu().numpy())
                labels = batch['labels'].cpu().numpy()
                all_labels.extend(labels.tolist())
        preds_arr = np.concatenate(all_preds, axis=0)
        y_score   = preds_arr[:, 1]
        y_true    = np.array(all_labels)
        auroc     = sk_metrics.roc_auc_score(y_true, y_score)
        results.append(auroc)
        log(f'  missing_rate={rate:.0%} -> AUROC={auroc:.4f}')

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r * 100 for r in rates], results,
            marker='o', color='steelblue', lw=2)
    ax.set_xlabel('Additional Missing Rate (%)')
    ax.set_ylabel('AUROC')
    ax.set_title('Robustness to Missing Data (SMART-FiLM)')
    ax.set_ylim(bottom=max(0.5, min(results) - 0.05))
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['c12', 'c19', 'mimic_mortality',
                                 'mimic_phenotyping', 'mimic_decompensation',
                                 'mimic_lengthofstay'])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint-prc.pth from finetune')
    parser.add_argument('--d_model',   type=int, default=32)
    parser.add_argument('--e_layers',  type=int, default=2)
    parser.add_argument('--n_heads',   type=int, default=4)
    parser.add_argument('--time_dim',  type=int, default=16)
    parser.add_argument('--dropout',   type=float, default=0.1)
    parser.add_argument('--batch_size',type=int, default=128)
    parser.add_argument('--seed',      type=int, default=3407)
    parser.add_argument('--out_dir',   type=str, default=None,
                        help='Output dir for figures (default: ./figs/{dataset}/{model}/seed_{seed})')
    parser.add_argument('--use-film', action='store_true', default=False,
                        help='Load TimeFiLMEncoder (SMART-FiLM) instead of base Encoder')
    parser.add_argument('--use-smile', action='store_true', default=False,
                        help='Load SMILEEncoder (SMART-SMILE) instead of base Encoder')
    parser.add_argument('--no_missing_rate', action='store_true',
                        help='Skip the (slow) missing-rate robustness plot')
    args = parser.parse_args()

    if args.out_dir is None:
        if args.use_smile:
            model_name = 'smart-smile'
        elif args.use_film:
            model_name = 'smart-film'
        else:
            model_name = 'smart'
        args.out_dir = os.path.join('./figs', args.dataset, model_name, f'seed_{args.seed}')

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    encoder, classifier, loader, task = load_model_and_data(args)
    tag = args.dataset

    # --- gather predictions once for ROC/PR ---
    if task == 'binary':
        log('Collecting predictions on test set...')
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                for key in batch:
                    batch[key] = batch[key].cuda()
                original_mask = batch['mask'] if args.use_smile else None
                h     = encoder(**batch, original_mask=original_mask)
                preds = classifier(h, **batch).softmax(-1).cpu().numpy()
                all_preds.append(preds)
                all_labels.extend(batch['labels'].cpu().numpy().tolist())

        preds_arr = np.concatenate(all_preds, axis=0)
        labels_arr = np.array(all_labels)

        plot_roc_pr(
            labels_arr, preds_arr,
            out_path=os.path.join(args.out_dir, f'{tag}_roc_pr.pdf'),
            title=f'SMART-FiLM ({args.dataset})'
        )

    # --- FiLM heatmap (only for SMART-FiLM, baseline has no film_gen) ---
    if args.use_film:
        plot_film_heatmap(
            encoder, loader, args,
            out_path=os.path.join(args.out_dir, f'{tag}_film_heatmap.pdf'),
            n_patients=4,
        )

    # --- t-SNE ---
    plot_tsne(
        encoder, loader,
        out_path=os.path.join(args.out_dir, f'{tag}_tsne.pdf'),
        max_samples=2000,
        use_smile=args.use_smile,
    )

    # --- missing rate ---
    if not args.no_missing_rate and task == 'binary':
        plot_missing_rate(
            encoder, classifier, loader, args,
            out_path=os.path.join(args.out_dir, f'{tag}_missing_rate.pdf'),
        )

    log('All figures saved to ' + args.out_dir)
