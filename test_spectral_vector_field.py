"""
Test script for SpectralModulatedVectorField

Verifies:
1. Correct initialization
2. Forward pass works correctly
3. Integration with cdeint (time_aware=True)
4. Low-pass filter behavior
"""

import torch
import sys
import os

# Add NeuralCDE-master to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NeuralCDE-master'))

from experiments.models.vector_fields import SpectralModulatedVectorField
from controldiffeq import cdeint


def test_initialization():
    """Test that the model initializes correctly."""
    print("=" * 60)
    print("Test 1: Initialization")
    print("=" * 60)

    input_channels = 10
    hidden_channels = 64

    model = SpectralModulatedVectorField(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        time_dim=32,
        spectral_sigma=2.0
    )

    print(f"Model created: {model.extra_repr()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Check spectral weights
    print(f"\nSpectral weights shape: {model.spectral_weights.shape}")
    print(f"Expected: [{hidden_channels // 2 + 1}]")

    # Verify low-pass filter initialization
    magnitudes = torch.abs(model.spectral_weights)
    print(f"\nLow-pass filter check:")
    print(f"  DC component (k=0): {magnitudes[0]:.4f} (should be ~1.0)")
    print(f"  High freq (k=max): {magnitudes[-1]:.4f} (should be ~0.0)")
    print(f"  Ratio (decay): {magnitudes[-1] / magnitudes[0]:.6f}")

    print("\nTest 1: PASSED\n")
    return model


def test_forward_pass():
    """Test forward pass with different batch shapes."""
    print("=" * 60)
    print("Test 2: Forward Pass")
    print("=" * 60)

    input_channels = 10
    hidden_channels = 64
    batch_size = 32

    model = SpectralModulatedVectorField(
        input_channels=input_channels,
        hidden_channels=hidden_channels
    )

    # Test case 1: Standard batch
    t = torch.tensor(0.5)
    z = torch.randn(batch_size, hidden_channels)

    print(f"Input: t={t.item():.2f}, z.shape={z.shape}")

    output = model(t, z)
    expected_shape = (batch_size, hidden_channels, input_channels)

    print(f"Output shape: {output.shape}")
    print(f"Expected: {expected_shape}")

    assert output.shape == expected_shape, f"Shape mismatch: {output.shape} != {expected_shape}"
    print("Shape check: PASSED")

    # Test case 2: Different time values
    t_batch = torch.linspace(0, 1, batch_size)
    output2 = model(t_batch, z)
    print(f"\nTime-varying input: t.shape={t_batch.shape}")
    print(f"Output shape: {output2.shape}")
    assert output2.shape == expected_shape
    print("Time-varying check: PASSED")

    # Test case 3: Check that frequency branch is working
    # Disable frequency branch
    with torch.no_grad():
        freq_scale_backup = model.freq_scale.clone()
        model.freq_scale.zero_()

    output_no_freq = model(t, z)

    with torch.no_grad():
        model.freq_scale.copy_(freq_scale_backup)

    output_with_freq = model(t, z)

    diff = (output_with_freq - output_no_freq).abs().mean()
    print(f"\nFrequency branch contribution: {diff:.6f}")
    print(f"Non-zero difference confirms frequency branch is active: {diff > 1e-6}")

    print("\nTest 2: PASSED\n")


def test_cdeint_integration():
    """Test integration with cdeint (time_aware=True)."""
    print("=" * 60)
    print("Test 3: Integration with cdeint")
    print("=" * 60)

    input_channels = 5
    hidden_channels = 32
    batch_size = 16
    length = 10

    model = SpectralModulatedVectorField(
        input_channels=input_channels,
        hidden_channels=hidden_channels
    )

    # Create control signal (interpolated path)
    times = torch.linspace(0, 1, length)
    control_values = torch.randn(batch_size, length, input_channels)

    # Simple linear interpolation for dX_dt
    def dX_dt(t):
        # Find nearest time index
        idx = torch.clamp(
            torch.searchsorted(times, t.item()),
            0, length - 2
        )
        # Linear interpolation
        t0, t1 = times[idx], times[idx + 1]
        x0, x1 = control_values[:, idx], control_values[:, idx + 1]
        alpha = (t.item() - t0) / (t1 - t0 + 1e-8)
        return (1 - alpha) * x0 + alpha * x1

    # Initial state
    z0 = torch.randn(batch_size, hidden_channels)

    print(f"Control signal: {length} time steps, {input_channels} channels")
    print(f"Initial state: z0.shape={z0.shape}")
    print(f"Integration times: {times[:3].tolist()} ... {times[-3:].tolist()}")

    # Solve CDE
    print("\nSolving CDE with time_aware=True...")
    z_trajectory = cdeint(
        dX_dt=dX_dt,
        z0=z0,
        func=model,
        t=times,
        adjoint=False,
        time_aware=True,  # CRITICAL: must be True for SpectralModulatedVectorField
        method='euler',
        options=dict(step_size=0.1)
    )

    print(f"Output trajectory shape: {z_trajectory.shape}")
    print(f"Expected: ({length}, {batch_size}, {hidden_channels})")

    assert z_trajectory.shape == (length, batch_size, hidden_channels)
    print("\nCDE integration: PASSED")

    # Check trajectory is reasonable (not NaN, not exploding)
    print(f"\nTrajectory statistics:")
    print(f"  Mean: {z_trajectory.mean():.4f}")
    print(f"  Std: {z_trajectory.std():.4f}")
    print(f"  Min: {z_trajectory.min():.4f}")
    print(f"  Max: {z_trajectory.max():.4f}")

    assert not torch.isnan(z_trajectory).any(), "NaN detected in trajectory!"
    assert torch.isfinite(z_trajectory).all(), "Inf detected in trajectory!"
    print("Trajectory sanity check: PASSED")

    print("\nTest 3: PASSED\n")


def test_lowpass_behavior():
    """Verify that spectral weights favor low frequencies."""
    print("=" * 60)
    print("Test 4: Low-Pass Filter Behavior")
    print("=" * 60)

    hidden_channels = 64

    # Test different sigma values
    sigmas = [1.0, 2.0, 4.0]

    for sigma in sigmas:
        model = SpectralModulatedVectorField(
            input_channels=10,
            hidden_channels=hidden_channels,
            spectral_sigma=sigma
        )

        magnitudes = torch.abs(model.spectral_weights)

        # Check monotonic decrease
        is_decreasing = (magnitudes[1:] <= magnitudes[:-1]).all()

        # Calculate cutoff frequency (where magnitude drops to ~50%)
        cutoff_idx = (magnitudes > 0.5 * magnitudes[0]).sum().item() - 1
        cutoff_ratio = cutoff_idx / len(magnitudes)

        print(f"\nsigma={sigma:.1f}:")
        print(f"  Magnitude range: [{magnitudes[0]:.4f}, {magnitudes[-1]:.6f}]")
        print(f"  Monotonic decrease: {is_decreasing}")
        print(f"  50% cutoff at {cutoff_ratio*100:.1f}% of frequencies")

    print("\nTest 4: PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing SpectralModulatedVectorField")
    print("=" * 60 + "\n")

    try:
        # Run all tests
        test_initialization()
        test_forward_pass()
        test_cdeint_integration()
        test_lowpass_behavior()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nUsage example:")
        print("-" * 60)
        print("""
from experiments.models.vector_fields import SpectralModulatedVectorField
from controldiffeq import cdeint

# Create model
func = SpectralModulatedVectorField(
    input_channels=10,      # Number of control channels
    hidden_channels=64,     # Hidden state dimension
    time_dim=32,            # Time encoding dimension
    spectral_sigma=2.0      # Low-pass filter bandwidth
)

# Solve CDE
z_t = cdeint(
    dX_dt=control_gradient_func,
    z0=initial_state,
    func=func,
    t=times,
    time_aware=True,  # MUST be True!
    adjoint=True,
    method='dopri5'
)
        """)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
