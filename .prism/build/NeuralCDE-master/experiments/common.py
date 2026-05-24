import copy
import json
import math
import numpy as np
import os
import pathlib
import sklearn.metrics
import torch
import tqdm
import warnings

import models

here = pathlib.Path(__file__).resolve().parent


def _add_weight_regularisation(loss_fn, regularise_parameters, scaling=0.03):
    def new_loss_fn(pred_y, true_y):
        total_loss = loss_fn(pred_y, true_y)
        for parameter in regularise_parameters.parameters():
            if parameter.requires_grad:
                total_loss = total_loss + scaling * parameter.norm()
        return total_loss
    return new_loss_fn


class _SqueezeEnd(torch.nn.Module):
    def __init__(self, model):
        super(_SqueezeEnd, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs).squeeze(-1)


def _count_parameters(model):
    """Counts the number of parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad_)


class _AttrDict(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)


def _evaluate_metrics(dataloader, model, times, loss_fn, num_classes, device, kwargs):
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*unique classes.*", category=UserWarning)
        total_accuracy = 0
        total_confusion = torch.zeros(num_classes, num_classes).numpy()  # occurs all too often
        total_dataset_size = 0
        total_loss = 0
        true_y_cpus = []
        pred_y_cpus = []

        for batch in dataloader:
            batch = tuple(b.to(device) for b in batch)
            *coeffs, true_y, lengths = batch
            batch_size = true_y.size(0)
            pred_y = model(times, coeffs, lengths, **kwargs)

            if num_classes == 2:
                thresholded_y = (pred_y > 0).to(true_y.dtype)
            else:
                thresholded_y = torch.argmax(pred_y, dim=1)
            true_y_cpu = true_y.detach().cpu()
            pred_y_cpu = pred_y.detach().cpu()
            if num_classes == 2:
                # Assume that our datasets aren't so large that this breaks
                true_y_cpus.append(true_y_cpu)
                pred_y_cpus.append(pred_y_cpu)
            thresholded_y_cpu = thresholded_y.detach().cpu()

            total_accuracy += (thresholded_y == true_y).sum().to(pred_y.dtype)
            total_confusion += sklearn.metrics.confusion_matrix(true_y_cpu, thresholded_y_cpu,
                                                                labels=range(num_classes))
            total_dataset_size += batch_size
            total_loss += loss_fn(pred_y, true_y) * batch_size

        total_loss /= total_dataset_size  # assume 'mean' reduction in the loss function
        total_accuracy /= total_dataset_size
        metrics = _AttrDict(accuracy=total_accuracy.item(), confusion=total_confusion, dataset_size=total_dataset_size,
                            loss=total_loss.item())

        if num_classes == 2:
            true_y_cpus = torch.cat(true_y_cpus, dim=0)
            pred_y_cpus = torch.cat(pred_y_cpus, dim=0)
            metrics.auroc = sklearn.metrics.roc_auc_score(true_y_cpus, pred_y_cpus)
            metrics.average_precision = sklearn.metrics.average_precision_score(true_y_cpus, pred_y_cpus)
        return metrics


def _masked_bce_loss(pred_y, true_y, pos_weight=None):
    """Compute BCE loss only at valid time steps (where true_y >= 0).

    Arguments:
        pred_y: [batch, times] logits at each time step.
        true_y: [batch, times] labels, -1 indicates invalid/padding.
        pos_weight: Optional positive class weight for BCEWithLogitsLoss.

    Returns:
        Scalar loss averaged over valid time steps.
    """
    mask = (true_y >= 0)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_y.device, requires_grad=True)
    pred_valid = pred_y[mask]
    true_valid = true_y[mask].float()
    if pos_weight is not None:
        pos_weight = pos_weight.to(pred_y.device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss_fn(pred_valid, true_valid)


def _evaluate_metrics_stream(dataloader, model, times, loss_fn, num_classes, device, kwargs):
    """Per-timestep evaluation for stream mode (decompensation).

    Pools all (patient, hour) predictions and computes global metrics.
    Only evaluates at valid positions (true_y >= 0).
    """
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*unique classes.*", category=UserWarning)
        all_true = []
        all_pred = []
        total_loss = 0.0
        total_valid = 0

        for batch in dataloader:
            batch = tuple(b.to(device) for b in batch)
            *coeffs, true_y_seq, lengths = batch
            # true_y_seq: [batch, max_hours], pred_y: [batch, max_hours]
            pred_y = model(times, coeffs, lengths, **kwargs)

            mask = (true_y_seq >= 0)
            n_valid = mask.sum().item()
            if n_valid > 0:
                all_true.append(true_y_seq[mask].cpu())
                all_pred.append(pred_y[mask].cpu())
                total_loss += loss_fn(pred_y, true_y_seq).item() * n_valid
                total_valid += n_valid

        if total_valid == 0:
            return _AttrDict(accuracy=0.0, loss=0.0, auroc=0.0,
                            average_precision=0.0, dataset_size=0,
                            confusion=torch.zeros(2, 2).numpy())

        all_true = torch.cat(all_true, dim=0)
        all_pred = torch.cat(all_pred, dim=0)

        total_loss /= total_valid
        thresholded = (all_pred > 0).to(all_true.dtype)
        accuracy = (thresholded == all_true).float().mean().item()
        confusion = sklearn.metrics.confusion_matrix(
            all_true.numpy(), thresholded.numpy(), labels=[0, 1])

        auroc = sklearn.metrics.roc_auc_score(all_true.numpy(), all_pred.numpy())
        auprc = sklearn.metrics.average_precision_score(
            all_true.numpy(), all_pred.numpy())

        return _AttrDict(accuracy=accuracy, loss=total_loss,
                         auroc=auroc, average_precision=auprc,
                         dataset_size=total_valid,
                         confusion=confusion)


class _SuppressAssertions:
    def __init__(self, tqdm_range):
        self.tqdm_range = tqdm_range

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is AssertionError:
            self.tqdm_range.write('Caught AssertionError: ' + str(exc_val))
            return True


def _train_loop(train_dataloader, val_dataloader, model, times, optimizer, loss_fn, max_epochs, num_classes, device,
                kwargs, step_mode, stream_mode=False, grad_clip_norm=1.0, scheduler_patience=2):
    model.train()
    best_model = model
    best_train_loss = math.inf
    best_train_accuracy = 0
    best_val_accuracy = 0
    best_val_auroc = 0
    best_train_accuracy_epoch = 0
    best_train_loss_epoch = 0
    history = []
    breaking = False

    if step_mode:
        epoch_per_metric = 10
        plateau_terminate = 100
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience)
    else:
        epoch_per_metric = 10
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, mode='max')

    total_batches = len(train_dataloader)
    tqdm_range = tqdm.tqdm(range(max_epochs), desc='Epochs', position=0)
    tqdm_range.write('Starting training for model:\n\n' + str(model) + '\n\n')
    first_batch = True
    for epoch in tqdm_range:
        if breaking:
            break

        # SSM Warm-up: Freeze alpha for first 50 epochs to let DeepFiLM converge first
        # This prevents untrained low-rank perturbations from destabilizing ODE trajectories
        # Search for alpha parameter in nested model structure
        alpha_param = None
        for name, param in model.named_parameters():
            if 'alpha' in name and param.numel() == 1:  # SSM alpha is a scalar
                alpha_param = param
                break

        if alpha_param is not None:
            if epoch < 50:
                if alpha_param.requires_grad:
                    alpha_param.requires_grad = False
                    if epoch == 0:
                        tqdm_range.write('[SSM Warm-up] Freezing alpha for first 50 epochs...')
            else:
                if not alpha_param.requires_grad:
                    alpha_param.requires_grad = True
                    tqdm_range.write('[SSM Warm-up] Unfreezing alpha at epoch 50. SSM now active!')

        batch_bar = tqdm.tqdm(train_dataloader, desc=f'  Epoch {epoch}', position=1, leave=False,
                              total=total_batches)
        epoch_loss = 0.0
        batch_count = 0
        for batch in batch_bar:
            batch = tuple(b.to(device) for b in batch)
            if breaking:
                break
            with _SuppressAssertions(tqdm_range):
                if first_batch:
                    tqdm_range.write('[INFO] Processing first batch (may take 5-15 minutes for Mamba CUDA kernel compilation)...')
                *train_coeffs, train_y, lengths = batch
                pred_y = model(times, train_coeffs, lengths, **kwargs)
                if first_batch:
                    tqdm_range.write('[INFO] First forward pass completed! Training will be much faster now.')
                    first_batch = False
                loss = loss_fn(pred_y, train_y)
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                # stream_mode generates larger gradient norms (168 steps vs 1), so clip more loosely
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                batch_count += 1
                batch_bar.set_postfix(loss=f'{loss.item():.4f}')
        batch_bar.close()
        avg_loss = epoch_loss / max(batch_count, 1)
        tqdm_range.set_postfix(loss=f'{avg_loss:.4f}')

        if epoch % epoch_per_metric == 0 or epoch == max_epochs - 1:
            model.eval()
            evaluate_fn = _evaluate_metrics_stream if stream_mode else _evaluate_metrics
            train_metrics = evaluate_fn(train_dataloader, model, times, loss_fn, num_classes, device, kwargs)
            val_metrics = evaluate_fn(val_dataloader, model, times, loss_fn, num_classes, device, kwargs)
            model.train()

            if train_metrics.loss * 1.0001 < best_train_loss:
                best_train_loss = train_metrics.loss
                best_train_loss_epoch = epoch

            if train_metrics.accuracy > best_train_accuracy * 1.001:
                best_train_accuracy = train_metrics.accuracy
                best_train_accuracy_epoch = epoch

            if hasattr(val_metrics, 'auroc') and val_metrics.auroc > best_val_auroc:
                best_val_auroc = val_metrics.auroc
                del best_model  # so that we don't have three copies of a model simultaneously
                best_model = copy.deepcopy(model)
            elif not hasattr(val_metrics, 'auroc') and val_metrics.accuracy > best_val_accuracy:
                best_val_accuracy = val_metrics.accuracy
                del best_model
                best_model = copy.deepcopy(model)

            auroc_str = '  Val AUROC: {:.4f}'.format(val_metrics.auroc) if hasattr(val_metrics, 'auroc') else ''
            best_str = '  (best)' if hasattr(val_metrics, 'auroc') and val_metrics.auroc >= best_val_auroc else ''
            tqdm_range.write('Epoch: {}  Train loss: {:.3}  Train accuracy: {:.3}  Val loss: {:.3}  '
                             'Val accuracy: {:.3}{}{}'
                             ''.format(epoch, train_metrics.loss, train_metrics.accuracy, val_metrics.loss,
                                       val_metrics.accuracy, auroc_str, best_str))
            # Update epoch progress bar with latest metrics
            postfix = {'t_loss': f'{train_metrics.loss:.3f}', 'v_acc': f'{val_metrics.accuracy:.3f}'}
            if hasattr(val_metrics, 'auroc'):
                postfix['v_auroc'] = f'{val_metrics.auroc:.4f}'
            tqdm_range.set_postfix(postfix)
            if step_mode:
                scheduler.step(train_metrics.loss)
            else:
                scheduler.step(val_metrics.accuracy)
            history.append(_AttrDict(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics))

            if epoch > best_train_loss_epoch + plateau_terminate:
                tqdm_range.write('Breaking because of no improvement in training loss for {} epochs.'
                                 ''.format(plateau_terminate))
                breaking = True
            if epoch > best_train_accuracy_epoch + plateau_terminate:
                tqdm_range.write('Breaking because of no improvement in training accuracy for {} epochs.'
                                 ''.format(plateau_terminate))
                breaking = True

    for parameter, best_parameter in zip(model.parameters(), best_model.parameters()):
        parameter.data = best_parameter.data
    return history


class _TensorEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()
        else:
            super(_TensorEncoder, self).default(o)


def _save_results(name, result):
    loc = here / 'results' / name
    if not os.path.exists(loc):
        os.mkdir(loc)
    num = -1
    for filename in os.listdir(loc):
        try:
            num = max(num, int(filename))
        except ValueError:
            pass
    result_to_save = result.copy()
    del result_to_save['train_dataloader']
    del result_to_save['val_dataloader']
    del result_to_save['test_dataloader']
    result_to_save['model'] = str(result_to_save['model'])
    if 'vector_field' in result_to_save:
        result_to_save['vector_field'] = str(result_to_save['vector_field'])

    num += 1
    with open(loc / str(num), 'w') as f:
        json.dump(result_to_save, f, cls=_TensorEncoder)


def main(name, times, train_dataloader, val_dataloader, test_dataloader, device, make_model, num_classes, max_epochs,
         lr, kwargs, step_mode, pos_weight=torch.tensor(1), stream_mode=False,
         regularisation_scaling=0.03, grad_clip_norm=1.0, scheduler_patience=2):
    times = times.to(device)
    if device != 'cpu':
        torch.cuda.reset_max_memory_allocated(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None

    model, regularise_parameters = make_model()
    if stream_mode:
        # Stream mode: per-timestep prediction with masked loss
        # Model output is [batch, times, 1], squeezed to [batch, times]
        # Do NOT wrap with _SqueezeEnd (model handles squeezing internally)
        loss_fn = lambda pred_y, true_y: _masked_bce_loss(pred_y, true_y, pos_weight=pos_weight)
    elif num_classes == 2:
        model = _SqueezeEnd(model)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = torch.nn.functional.cross_entropy
    loss_fn = _add_weight_regularisation(loss_fn, regularise_parameters, scaling=regularisation_scaling)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = _train_loop(train_dataloader, val_dataloader, model, times, optimizer, loss_fn, max_epochs,
                          num_classes, device, kwargs, step_mode, stream_mode=stream_mode,
                          grad_clip_norm=grad_clip_norm, scheduler_patience=scheduler_patience)

    model.eval()
    evaluate_fn = _evaluate_metrics_stream if stream_mode else _evaluate_metrics
    train_metrics = evaluate_fn(train_dataloader, model, times, loss_fn, num_classes, device, kwargs)
    val_metrics = evaluate_fn(val_dataloader, model, times, loss_fn, num_classes, device, kwargs)
    test_metrics = evaluate_fn(test_dataloader, model, times, loss_fn, num_classes, device, kwargs)

    if device != 'cpu':
        memory_usage = torch.cuda.max_memory_allocated(device) - baseline_memory
    else:
        memory_usage = None

    result = _AttrDict(times=times,
                       memory_usage=memory_usage,
                       baseline_memory=baseline_memory,
                       num_classes=num_classes,
                       train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       test_dataloader=test_dataloader,
                       model=model.to('cpu'),
                       vector_field=regularise_parameters,  # Store vector_field reference for logging
                       parameters=_count_parameters(model),
                       history=history,
                       train_metrics=train_metrics,
                       val_metrics=val_metrics,
                       test_metrics=test_metrics)
    if name is not None:
        _save_results(name, result)
    return result


def make_model(name, input_channels, output_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers,
               use_intensity, initial):
    if name == 'ncde':
        def make_model():
            vector_field = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'ncde-film':
        # Time-modulated Neural CDE with FiLM mechanism (requires time_aware=True in kwargs)
        def make_model():
            vector_field = models.ModulatedSingleHiddenLayer(input_channels=input_channels,
                                                             hidden_channels=hidden_channels,
                                                             time_dim=32,
                                                             hidden_hidden_channels=hidden_hidden_channels,
                                                             num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'ncde-spectral':
        # Spectral-FiLM hybrid Neural CDE with frequency-domain filtering (requires time_aware=True in kwargs)
        def make_model():
            vector_field = models.SpectralModulatedVectorField(input_channels=input_channels,
                                                               hidden_channels=hidden_channels,
                                                               time_dim=32,
                                                               spectral_sigma=2.0,
                                                               hidden_hidden_channels=hidden_hidden_channels,
                                                               num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'ncde-spectral-v2':
        # Enhanced Spectral-FiLM with dynamic fusion and multi-scale frequency features (requires time_aware=True)
        def make_model():
            vector_field = models.EnhancedSpectralModulatedVectorField(input_channels=input_channels,
                                                                       hidden_channels=hidden_channels,
                                                                       time_dim=32,
                                                                       spectral_sigma=2.0,
                                                                       hidden_hidden_channels=hidden_hidden_channels,
                                                                       num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'ncde-mamba':
        # Mamba-NCDE: Mamba state space model for global dynamics (requires time_aware=True in kwargs)
        def make_model():
            if not models.MAMBA_AVAILABLE:
                raise ImportError("Mamba-NCDE requires mamba-ssm. Install: pip install mamba-ssm")
            vector_field = models.MambaModulatedVectorField(input_channels=input_channels,
                                                            hidden_channels=hidden_channels,
                                                            time_dim=32,
                                                            hidden_hidden_channels=hidden_hidden_channels,
                                                            num_hidden_layers=num_hidden_layers,
                                                            mamba_d_model=64,
                                                            mamba_n_layer=2,
                                                            fusion_mode='learned')
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'ncde-deepfilm':
        # DeepFiLM: Pure time-branch with deep FiLM modulation and strong regularization
        # Removes ineffective global branches (Spectral/Mamba) for better generalization
        def make_model():
            vector_field = models.DeepFiLMVectorField(input_channels=input_channels,
                                                     hidden_channels=hidden_channels,
                                                     time_dim=32,
                                                     hidden_hidden_channels=hidden_hidden_channels,
                                                     num_hidden_layers=num_hidden_layers,
                                                     dropout=0.2)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'ncde-deepmlp':
        # Ablation: Deep MLP without FiLM or Dropout (stateless, time_aware=False)
        def make_model():
            vector_field = models.DeepMLPVectorField(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels,
                                     hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'ncde-deepmlp-dropout':
        # Ablation: Deep MLP with Dropout but without FiLM (stateless, time_aware=False)
        def make_model():
            vector_field = models.DeepMLPDropoutVectorField(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
                dropout=0.2)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels,
                                     hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'ncde-lowrankode-film':
        # LowRankODE-FiLM: SSM-structured dynamics + DeepFiLM (Mode-Mamba-TS inspired)
        def make_model():
            vector_field = models.LowRankODE_FiLM_VectorField(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                time_dim=32,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
                dropout=0.2,
                r_rank=4)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels,
                                     hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'gruode':
        def make_model():
            vector_field = models.GRU_ODE(input_channels=input_channels, hidden_channels=hidden_channels)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels,
                                     hidden_channels=hidden_channels, output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'dt':
        def make_model():
            model = models.GRU_dt(input_channels=input_channels, hidden_channels=hidden_channels,
                                  output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'decay':
        def make_model():
            model = models.GRU_D(input_channels=input_channels, hidden_channels=hidden_channels,
                                 output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'odernn':
        def make_model():
            model = models.ODERNN(input_channels=input_channels, hidden_channels=hidden_channels,
                                  hidden_hidden_channels=hidden_hidden_channels, num_hidden_layers=num_hidden_layers,
                                  output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    else:
        raise ValueError("Unrecognised model name {}. Valid names are 'ncde', 'ncde-film', 'ncde-spectral', 'ncde-mamba', 'ncde-deepfilm', 'ncde-lowrankode-film', 'ncde-deepmlp', 'ncde-deepmlp-dropout', 'gruode', 'dt', 'decay' and 'odernn'."
                         "".format(name))
    return make_model
