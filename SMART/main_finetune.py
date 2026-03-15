import argparse
import json
import os
import logging
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from tqdm import tqdm

from data.challenge2012 import load_challenge_2012
from data.challenge2019 import load_challenge_2019
from data.mimiciii import load_mimic_iii_mortality, load_mimic_iii_phenotyping, load_mimic_iii_decompensation, load_mimic_iii_lengthofstay
from data.dataloader import collate_fn
from models.smart import Classifier
from utils.metrics import print_metrics_binary, print_metrics_multilabel, print_metrics_regression
from utils.utils import set_seed, distributed_init, init_logging
from utils.variable_order import get_variable_order


def compute_class_weights(dataset, num_classes=2):
    """Compute inverse-frequency class weights from dataset labels."""
    from collections import Counter
    counts = Counter()
    for sample in dataset.data:
        label = sample['labels']
        counts[int(label)] += 1
    total = sum(counts.values())
    weights = torch.zeros(num_classes)
    for c in range(num_classes):
        weights[c] = total / (num_classes * max(counts[c], 1))
    return weights


def apply_mnar_dropout(original_mask, dropout_rate=0.05):
    """Apply fixed random dropout to original_mask during training.
    Reduces pretrain->finetune distribution shift by mimicking pretrain's mnar dropout.
    """
    drop = torch.rand_like(original_mask.float()) < dropout_rate
    return original_mask * (~drop)


def test(args, checkpoint_path, test_dataloader):
    checkpoint = torch.load(os.path.join(args.save_dir, checkpoint_path), weights_only=False)
    save_epoch = checkpoint['epoch']
    log(logger, "last saved model is in epoch {}".format(save_epoch))
    encoder.load_state_dict(checkpoint['encoder'])
    classifier.load_state_dict(checkpoint['classifier'])
    encoder.eval()
    classifier.eval()
    test_loss = 0
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for batch in test_dataloader:
            for key in batch:
                batch[key] = batch[key].cuda()
            if args.use_mnar or args.use_smile or args.use_smile_film or args.use_smile_v2 or args.use_smile_v2_film:
                original_mask = batch['mask'].clone()  # no dropout: test uses clean mask
            else:
                original_mask = None
            h = encoder(**batch, original_mask=original_mask)
            preds = classifier(h, **batch)
            test_loss += criterion(preds, batch['labels']).item() * batch['x'].shape[0]
            preds_all.append(preds.cpu())
            labels_all.append(batch['labels'].cpu())
    print_metrics(torch.cat(labels_all), torch.cat(preds_all), args.local_rank == 0)
    log(logger, 'Test Loss %.4f' % (test_loss / len(test_dataset)))


def log(logger, msg):
    if logger is not None:
        logger.info(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mimic_decompensation', choices=['c12', 'c19', 'mimic_mortality', 
                            'mimic_phenotyping', 'mimic_decompensation', 'mimic_lengthofstay'])
    parser.add_argument('--data_dropout', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--freeze_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='./export/')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--time_dim', type=int, default=16)
    parser.add_argument('--use-film', action='store_true', default=False)
    parser.add_argument('--use-smile', action='store_true', default=False)
    parser.add_argument('--use-smile-film', action='store_true', default=False,
                        help='Use SMILEFiLMEncoder (MNAR + FiLM joint modulation)')
    parser.add_argument('--use-mnar', action='store_true', default=False,
                        help='Use simplified MNAREncoder (no curriculum masking)')
    parser.add_argument('--use-smile-v2', action='store_true', default=False,
                        help='Use SMILEv2Encoder (MNAR attn bias + obs density + cross-attn fusion)')
    parser.add_argument('--use-smile-v2-film', action='store_true', default=False,
                        help='Use SMILEv2FiLMEncoder (SMILEv2 + time-conditional FiLM)')
    parser.add_argument('--use-smile-lean', action='store_true', default=False,
                        help='Use SMILELeanEncoder (MNAR cooccur bias + VarAtt FiLM + local obs density)')
    parser.add_argument('--obs-density-window', type=int, default=5,
                        help='Sliding window size for observation density embedding (must be odd)')
    parser.add_argument('--smile-no-mnar', action='store_true', default=False)
    parser.add_argument('--smile-no-curriculum', action='store_true', default=False)
    parser.add_argument('--smile-mask-type', choices=['all', 'temporal', 'system'], default='all')
    parser.add_argument('--smile-mnar-dropout', type=float, default=0.05,
                        help='MNAR dropout rate (base value, default 0.05)')
    parser.add_argument('--smile-mnar-dropout-initial', type=float, default=0.0,
                        help='Initial MNAR dropout for progressive decay schedule. '
                             'When > 0, linearly decays from this value to 0 over all epochs. '
                             'When 0 (default), uses constant smile-mnar-dropout instead.')
    args = parser.parse_args()
    if args.use_smile_lean:
        from models.smart import SMILELeanEncoder as Encoder
        model_name = 'smart-smile-lean'
    elif args.use_mnar:
        from models.smart import MNAREncoder as Encoder
        model_name = 'smart-mnar'
    elif args.use_smile_v2_film:
        from models.smart import SMILEv2FiLMEncoder as Encoder
        model_name = 'smart-smile-v2-film'
    elif args.use_smile_v2:
        from models.smart import SMILEv2Encoder as Encoder
        model_name = 'smart-smile-v2'
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
        train_dataset, val_dataset, test_dataset = load_mimic_iii_lengthofstay()
    else:
        raise Exception("Dataset not exist!")
    if args.data_dropout > 0:
        train_dataset.dropout_data(args.data_dropout)
        val_dataset.dropout_data(args.data_dropout)
        test_dataset.dropout_data(args.data_dropout)
    log(logger, 'Dataset Loaded.')
    distributed_init(args)
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
    classifier = Classifier(args).cuda()
    
    if args.distributed:
        encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[args.gpu], output_device=args.local_rank, find_unused_parameters=True)
        classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu], output_device=args.local_rank, find_unused_parameters=True)
    
    param_groups = [
        {
            'params': encoder.parameters(),
        }, 
        {
            'params': classifier.parameters()
        }
    ]
    optimizer = torch.optim.Adam(param_groups, args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    if args.dataset == 'mimic_phenotyping':
        criterion = torch.nn.BCEWithLogitsLoss()
        print_metrics = print_metrics_multilabel
        save_metric = 'auc_macro'
    elif args.dataset == 'mimic_lengthofstay':
        class_weights = compute_class_weights(train_dataset, num_classes=10).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        log(logger, f'Class weights: {class_weights.tolist()}')
        print_metrics = print_metrics_multilabel
        save_metric = 'auc_macro'
    elif args.dataset in ('mimic_decompensation', 'mimic_mortality'):
        # 与原论文一致，不加权重，极端权重会导致 AUPRC 崩溃
        criterion = torch.nn.CrossEntropyLoss()
        print_metrics = print_metrics_binary
        save_metric = 'auprc'
    else:
        # c12/c19：类别不平衡明显，加权有益
        class_weights = compute_class_weights(train_dataset, num_classes=2).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        log(logger, f'Class weights: {class_weights.tolist()}')
        print_metrics = print_metrics_binary
        save_metric = 'auprc'
    
    checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint-mse.pth'), weights_only=False)
    save_epoch = checkpoint['epoch']
    log(logger, "last saved model is in epoch {}".format(save_epoch))
    encoder.load_state_dict(checkpoint['encoder'])

    best_auc = 0
    best_prc = 0
    best_mse = 100
    # Progressive MNAR dropout schedule: linear decay from initial to 0 over epochs
    _mnar_initial = args.smile_mnar_dropout_initial if args.smile_mnar_dropout_initial > 0 \
        else args.smile_mnar_dropout
    _mnar_progressive = args.smile_mnar_dropout_initial > 0
    epoch_bar = tqdm(range(1, args.epochs + 1), desc='[Finetune]', unit='epoch')
    for i in epoch_bar:
        train_loss = 0
        val_loss = 0
        encoder.train()
        classifier.train()
        # Current MNAR dropout: linear decay if progressive schedule, else constant
        if _mnar_progressive:
            current_mnar_drop = _mnar_initial * max(0.0, 1.0 - (i - 1) / args.epochs)
        else:
            current_mnar_drop = args.smile_mnar_dropout
        batch_bar = tqdm(train_dataloader, desc=f'  Ep{i:>3}', leave=False, unit='batch')
        for step, batch in enumerate(batch_bar, 1):
            for key in batch:
                batch[key] = batch[key].cuda()
            if args.use_mnar or args.use_smile or args.use_smile_film or args.use_smile_v2 or args.use_smile_v2_film:
                original_mask = apply_mnar_dropout(batch['mask'].clone(), current_mnar_drop)
            else:
                original_mask = None
            if i <= args.freeze_epochs:
                with torch.no_grad():
                    h = encoder(**batch, original_mask=original_mask)
            else:
                h = encoder(**batch, original_mask=original_mask)
            preds = classifier(h, **batch)
            loss = criterion(preds, batch['labels'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch['x'].shape[0]
            batch_bar.set_postfix(loss=f'{loss.item():.4f}')

        encoder.eval()
        classifier.eval()
        preds_all = []
        labels_all = []
        with torch.no_grad():
            for batch in val_dataloader:
                for key in batch:
                    batch[key] = batch[key].cuda()
                if args.use_mnar or args.use_smile or args.use_smile_film or args.use_smile_v2 or args.use_smile_v2_film:
                    original_mask = batch['mask'].clone()  # no dropout: val uses clean mask
                else:
                    original_mask = None
                h = encoder(**batch, original_mask=original_mask)
                preds = classifier(h, **batch)
                val_loss += criterion(preds, batch['labels']).item() * batch['x'].shape[0]
                preds_all.append(preds.cpu())
                labels_all.append(batch['labels'].cpu())
        metrics = print_metrics(torch.cat(labels_all), torch.cat(preds_all), args.local_rank == 0)
        t_loss = train_loss / len(train_dataset) * args.world_size
        v_loss = val_loss / len(val_dataset)
        scheduler.step()
        epoch_bar.set_postfix(train=f'{t_loss:.4f}', val=f'{v_loss:.4f}')
        log(logger, 'Epoch %d: Train Loss %.4f, Valid Loss %.4f' % (i, t_loss, v_loss))
        cur_mse = v_loss
        if save_metric != 'mse':
            if metrics[save_metric] > best_prc:
                best_prc = metrics[save_metric]
                if args.local_rank == 0:
                    state = {
                        'encoder': encoder.state_dict(),
                        'classifier': classifier.state_dict(),
                        'epoch': i
                    }
                    log(logger, f'----- Save best model - {save_metric}: %.4f -----' % metrics[save_metric])
                    torch.save(state, os.path.join(args.save_dir, 'checkpoint-prc.pth'))
        else:
            if metrics[save_metric] < best_mse:
                best_mse = metrics[save_metric]
                if args.local_rank == 0:
                    state = {
                        'encoder': encoder.state_dict(),
                        'classifier': classifier.state_dict(),
                        'epoch': i
                    }
                    log(logger, f'----- Save best model - {save_metric}: %.4f -----' % metrics[save_metric])
                    torch.save(state, os.path.join(args.save_dir, 'checkpoint-prc.pth'))
        if args.distributed:
            dist.barrier()

    if args.distributed:
        dist.barrier()
    test(args, 'checkpoint-prc.pth', test_dataloader)
