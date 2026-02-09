"""
Quick integration test for new models (ncde-film, ncde-spectral)

Tests:
1. Model creation through common.make_model()
2. Forward pass with dummy data
3. Parameter counting
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NeuralCDE-master', 'experiments'))

import torch
import common
import models


def test_model_creation(model_name):
    """Test that model can be created successfully."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print('='*60)

    input_channels = 10
    output_channels = 3
    hidden_channels = 32

    make_model_fn = common.make_model(
        name=model_name,
        input_channels=input_channels,
        output_channels=output_channels,
        hidden_channels=hidden_channels,
        hidden_hidden_channels=32,
        num_hidden_layers=3,
        use_intensity=False,
        initial=True
    )

    model, vector_field = make_model_fn()

    print(f"Model type: {type(model).__name__}")
    print(f"Vector field type: {type(vector_field).__name__}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")

    # Print vector field details
    print(f"\nVector field: {vector_field}")

    return model, vector_field


def test_forward_pass(model, model_name):
    """Test forward pass with dummy data."""
    print(f"\n{'-'*60}")
    print("Testing forward pass...")
    print('-'*60)

    batch_size = 8
    seq_length = 10
    input_channels = 10

    # Create dummy cubic spline coefficients
    times = torch.linspace(0, 1, seq_length)
    coeffs_values = torch.randn(batch_size, seq_length, input_channels)

    # Compute natural cubic spline coefficients
    import controldiffeq
    coeffs = controldiffeq.natural_cubic_spline_coeffs(times, coeffs_values)

    # Final index (all sequences end at last time)
    final_index = torch.full((batch_size,), seq_length - 1, dtype=torch.long)

    # Forward pass
    kwargs = {}
    if model_name in ('ncde-film', 'ncde-spectral'):
        kwargs['time_aware'] = True
        print(f"Using time_aware=True for {model_name}")

    try:
        output = model(times, coeffs, final_index, **kwargs)
        print(f"Output shape: {output.shape}")
        print(f"Output stats: mean={output.mean():.4f}, std={output.std():.4f}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        print("\nForward pass: PASSED ✓")
        return True
    except Exception as e:
        print(f"\nForward pass: FAILED ✗")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("Model Integration Test")
    print("="*60)

    models_to_test = ['ncde', 'ncde-film', 'ncde-spectral']
    results = {}

    for model_name in models_to_test:
        try:
            model, vector_field = test_model_creation(model_name)
            success = test_forward_pass(model, model_name)
            results[model_name] = 'PASSED' if success else 'FAILED'
        except Exception as e:
            print(f"\nModel creation FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = 'FAILED'

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model_name, status in results.items():
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"{model_name:<20} {status:<10} {symbol}")

    print("="*60)

    all_passed = all(status == 'PASSED' for status in results.values())
    if all_passed:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print("\n❌ Some tests FAILED")
        return 1


if __name__ == '__main__':
    exit(main())
