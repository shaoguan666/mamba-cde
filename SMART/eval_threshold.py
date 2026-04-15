"""
eval_threshold.py - 用 val 集找最优 F1 threshold，再在 test 集上评估
结果追加写入 training.log 并保存到 eval_results.json
"""
import argparse
import json
import logging
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from sklearn import metrics

from data.mimiciii import load_mimic_iii_decompensation
from data.dataloader import collate_fn
from models.smart import Classifier
from utils.utils import set_seed


def load_encoder(args):
    if args.use_mnar:
        from models.smart import MNAREncoder as Encoder
    elif args.use_smile:
        from models.smart import SMILEEncoder as Encoder
    elif args.use_film:
        from models.smart import TimeFiLMEncoder as Encoder
    else:
        from models.smart import Encoder
    return Encoder(args).cuda()


def get_preds(encoder, classifier, dataloader, use_smile_mask):
    encoder.eval()
    classifier.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in dataloader:
            for key in batch:
                batch[key] = batch[key].cuda()
            original_mask = batch['mask'] if use_smile_mask else None
            h = encoder(**batch, original_mask=original_mask)
            preds = classifier(h, **batch)
            preds_all.append(preds.cpu())
            labels_all.append(batch['labels'].cpu())
    return torch.cat(labels_all).numpy(), torch.cat(preds_all).softmax(dim=-1)[:, 1].numpy()


def best_f1_threshold(y_true, probs):
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    idx = np.argmax(f1s)
    return thresholds[idx] if idx < len(thresholds) else 0.5, f1s[idx]


def report(y_true, probs, threshold, label, logger=None):
    preds = (probs >= threshold).astype(int)
    cf = metrics.confusion_matrix(y_true, preds)
    auroc = metrics.roc_auc_score(y_true, probs)
    tp = cf[1][1]
    fp = cf[0][1]
    fn = cf[1][0]
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    auprc = metrics.average_precision_score(y_true, probs)
    lines = [
        f"===== {label} (threshold={threshold:.4f}) =====",
        str(cf),
        f"AUROC  = {auroc:.4f}",
        f"AUPRC  = {auprc:.4f}",
        f"F1     = {f1:.4f}",
        f"Prec   = {prec:.4f}  Rec = {rec:.4f}",
    ]
    for line in lines:
        print(line)
        if logger:
            logger.info(line)
    return {"auroc": float(auroc), "auprc": float(auprc), "f1": float(f1),
            "prec": float(prec), "rec": float(rec), "threshold": float(threshold)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str,
                        default='./export/mimic_decompensation/smart-smile/seed_42')
    parser.add_argument('--checkpoint', type=str, default='checkpoint-prc.pth')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--time_dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use-film', action='store_true', default=False)
    parser.add_argument('--use-smile', action='store_true', default=False)
    parser.add_argument('--use-mnar', action='store_true', default=False)
    parser.add_argument('--smile-no-mnar', action='store_true', default=False)
    parser.add_argument('--smile-no-curriculum', action='store_true', default=False)
    parser.add_argument('--smile-mask-type', default='all')
    parser.add_argument('--smile-mnar-dropout', type=float, default=0.05)
    args = parser.parse_args()

    # 固定 decompensation 参数
    args.dataset = 'mimic_decompensation'
    args.input_dim = 17
    args.demo_dim = 0
    args.num_class = 2
    args.max_len = 24
    args.local_rank = 0
    args.distributed = False

    set_seed(args.seed)

    # 追加写入原 training.log
    log_path = os.path.join(args.save_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='a'),
            logging.StreamHandler(),
        ]
    )
    logger = logging.getLogger()
    logger.info("===== eval_threshold.py START =====")

    train_dataset, val_dataset, test_dataset = load_mimic_iii_decompensation()
    val_loader  = DataLoader(val_dataset,  batch_size=args.batch_size,
                             sampler=SequentialSampler(val_dataset),  collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             sampler=SequentialSampler(test_dataset), collate_fn=collate_fn)

    encoder    = load_encoder(args)
    classifier = Classifier(args).cuda()

    ckpt = torch.load(os.path.join(args.save_dir, args.checkpoint), weights_only=False)
    logger.info(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    encoder.load_state_dict(ckpt['encoder'])
    classifier.load_state_dict(ckpt['classifier'])

    use_smile_mask = args.use_smile or args.use_mnar

    logger.info("Running val set...")
    val_y, val_probs = get_preds(encoder, classifier, val_loader, use_smile_mask)
    opt_thr, val_f1 = best_f1_threshold(val_y, val_probs)
    logger.info(f"Val best F1={val_f1:.4f} at threshold={opt_thr:.4f}")
    report(val_y, val_probs, opt_thr, "Val (optimal threshold)", logger)

    logger.info("Running test set...")
    test_y, test_probs = get_preds(encoder, classifier, test_loader, use_smile_mask)
    r_orig = report(test_y, test_probs, 0.5,     "Test (threshold=0.5, original)",    logger)
    r_opt  = report(test_y, test_probs, opt_thr, "Test (optimal threshold from val)", logger)

    # 保存 eval_results.json 到同目录
    results = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(ckpt['epoch']),
        "val_optimal_threshold": float(opt_thr),
        "val_best_f1": float(val_f1),
        "test_default_threshold": r_orig,
        "test_optimal_threshold": r_opt,
    }
    out_path = os.path.join(args.save_dir, 'eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")
    logger.info("===== eval_threshold.py END =====")


if __name__ == '__main__':
    main()
