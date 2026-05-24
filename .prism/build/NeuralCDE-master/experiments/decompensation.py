import pathlib
import sys
import torch

import common
import datasets.decompensation

here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(here / '..'))
import controldiffeq


class InitialValueNetwork(torch.nn.Module):
    """Generate initial hidden state z0 from the first time step features.

    Unlike Sepsis (which uses separate static features), Decompensation
    derives z0 from the spline evaluation at t=0.
    """
    def __init__(self, input_channels, hidden_channels, model):
        super(InitialValueNetwork, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.linear1 = torch.nn.Linear(input_channels, 256)
        self.linear2 = torch.nn.Linear(256, hidden_channels)
        self.model = model

    def forward(self, times, coeffs, final_index, **kwargs):
        # Build spline from coefficients and evaluate at t=0
        cubic_spline = controldiffeq.NaturalCubicSpline(times, coeffs)
        x0 = cubic_spline.evaluate(times[0])  # [batch, input_channels]
        z0 = self.linear1(x0)
        z0 = z0.relu()
        z0 = self.linear2(z0)
        # Use stream=True to get predictions at all time points
        pred = self.model(times, coeffs, final_index, z0=z0, stream=True, **kwargs)
        # pred shape: [batch, times, 1] -> squeeze to [batch, times]
        return pred.squeeze(-1)


def main(device='cuda', max_epochs=200, pos_weight=55, *,
         model_name, hidden_channels, hidden_hidden_channels, num_hidden_layers,
         data_dir=None, dry_run=False, max_hours=72, **kwargs):
    """Run decompensation experiment.

    Arguments:
        device: 'cuda' or 'cpu'.
        max_epochs: Maximum training epochs.
        pos_weight: Positive class weight for BCE loss (decompensation is imbalanced).
        model_name: Model type ('ncde', 'ncde-deepfilm', etc.).
        hidden_channels: Hidden state dimension.
        hidden_hidden_channels: Hidden layer width in vector field.
        num_hidden_layers: Number of hidden layers in vector field.
        data_dir: Path to mimic3-benchmarks decompensation data directory.
        dry_run: If True, do not save results.
        **kwargs: Additional kwargs passed to cdeint.
    """
    if data_dir is None:
        # Default data directory
        data_dir = here / 'datasets' / 'data' / 'decompensation'

    batch_size = 512
    lr = 0.0001 * (batch_size / 32)

    # Relax ODE solver tolerances for the 72-step decompensation integration.
    # torchdiffeq defaults (rtol=1e-7, atol=1e-9) are extremely strict,
    # causing NFE~32K and ~4.5h training. Error threshold = atol + rtol*|z|.
    # With tanh-bounded hidden state |z|<=1, loosening to rtol=1e-3, atol=1e-4
    # raises the threshold ~10000x, expected NFE reduction: 32K -> ~5K (6x faster).
    kwargs.setdefault('rtol', 1e-3)
    kwargs.setdefault('atol', 1e-4)

    # These models use intensity for evolution
    time_intensity = model_name in ('odernn', 'dt', 'decay')

    times, train_dataloader, val_dataloader, test_dataloader = \
        datasets.decompensation.get_data(data_dir, time_intensity, batch_size,
                                         max_hours=max_hours)

    # input_channels: 1 (time) + 17 MIMIC-III benchmark features = 18
    # (mimic3-benchmarks uses 17 standard clinical variables, not 76)
    # Instead of hardcoding, we infer it from the data
    example_batch = next(iter(train_dataloader))
    input_channels = example_batch[0].size(-1)

    # output_channels=1 for binary classification
    make_model = common.make_model(model_name, input_channels, 1, hidden_channels,
                                   hidden_hidden_channels, num_hidden_layers,
                                   use_intensity=False, initial=False)

    def new_make_model():
        model, regularise = make_model()
        # Boost gradients on final linear layer (same trick as sepsis/speech)
        model.linear.weight.register_hook(lambda grad: 100 * grad)
        model.linear.bias.register_hook(lambda grad: 100 * grad)
        return InitialValueNetwork(input_channels, hidden_channels, model), regularise

    # Set time_aware=True for time-modulated models
    if model_name in ('ncde-film', 'ncde-spectral', 'ncde-spectral-v2',
                       'ncde-mamba', 'ncde-deepfilm', 'ncde-lowrankode-film'):
        kwargs['time_aware'] = True

    if dry_run:
        name = None
    else:
        name = 'decompensation'

    num_classes = 2
    # stream_mode uses 72-step ODE: gradient norms are ~10-20x larger than single-step tasks.
    # Use looser gradient clipping (10.0 vs default 1.0) to avoid crippling effective LR.
    # Regularization scaling reduced (0.003 vs 0.03) because stream-mode BCE loss is already
    # averaged over 72 steps, making the per-sample loss smaller relative to regularization.
    return common.main(name, times, train_dataloader, val_dataloader,
                       test_dataloader, device, new_make_model, num_classes,
                       max_epochs, lr, kwargs,
                       pos_weight=torch.tensor(pos_weight),
                       step_mode=True, stream_mode=True,
                       regularisation_scaling=0.003, grad_clip_norm=10.0,
                       scheduler_patience=4)


def run_all(device, model_names=('ncde', 'ncde-deepfilm', 'odernn', 'gruode')):
    model_kwargs = {
        'ncde': dict(hidden_channels=64, hidden_hidden_channels=49, num_hidden_layers=4),
        'ncde-deepfilm': dict(hidden_channels=64, hidden_hidden_channels=49, num_hidden_layers=4),
        'odernn': dict(hidden_channels=128, hidden_hidden_channels=128, num_hidden_layers=4),
        'gruode': dict(hidden_channels=187, hidden_hidden_channels=None, num_hidden_layers=None),
    }
    for model_name in model_names:
        for _ in range(5):
            main(device, model_name=model_name, **model_kwargs[model_name])
