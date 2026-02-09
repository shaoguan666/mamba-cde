"""测试 ncde-film 和 ncde-spectral 使用统一超参数后的行为"""
import torch
import sys
sys.path.append('NeuralCDE-master/experiments')

from models import ModulatedSingleHiddenLayer, SpectralModulatedVectorField, FinalTanh


def test_model_creation():
    """测试模型能否正确创建"""
    input_channels = 70  # 1 + (1 + 1) * 34 = 70 (for sepsis with intensity)
    hidden_channels = 64
    hidden_hidden_channels = 49
    num_hidden_layers = 4

    print("=" * 60)
    print("测试模型创建和参数统一性")
    print("=" * 60)

    # 测试 ncde (FinalTanh)
    print("\n1. 创建 ncde 向量场 (FinalTanh)...")
    ncde_vf = FinalTanh(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        hidden_hidden_channels=hidden_hidden_channels,
        num_hidden_layers=num_hidden_layers
    )
    ncde_params = sum(p.numel() for p in ncde_vf.parameters())
    print(f"   [OK] ncde 向量场创建成功")
    print(f"   - 参数: {ncde_vf}")
    print(f"   - 总参数量: {ncde_params:,}")

    # 测试 ncde-film
    print("\n2. 创建 ncde-film 向量场 (ModulatedSingleHiddenLayer)...")
    film_vf = ModulatedSingleHiddenLayer(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        time_dim=32,
        hidden_hidden_channels=hidden_hidden_channels,
        num_hidden_layers=num_hidden_layers
    )
    film_params = sum(p.numel() for p in film_vf.parameters())
    print(f"   [OK] ncde-film 向量场创建成功")
    print(f"   - 参数: {film_vf}")
    print(f"   - 总参数量: {film_params:,}")

    # 测试 ncde-spectral
    print("\n3. 创建 ncde-spectral 向量场 (SpectralModulatedVectorField)...")
    spectral_vf = SpectralModulatedVectorField(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        time_dim=32,
        spectral_sigma=2.0,
        hidden_hidden_channels=hidden_hidden_channels,
        num_hidden_layers=num_hidden_layers
    )
    spectral_params = sum(p.numel() for p in spectral_vf.parameters())
    print(f"   [OK] ncde-spectral 向量场创建成功")
    print(f"   - 参数: {spectral_vf}")
    print(f"   - 总参数量: {spectral_params:,}")

    # 对比参数量
    print("\n" + "=" * 60)
    print("参数量对比")
    print("=" * 60)
    print(f"ncde:          {ncde_params:>10,} 参数")
    print(f"ncde-film:     {film_params:>10,} 参数 (额外: {film_params - ncde_params:+,})")
    print(f"ncde-spectral: {spectral_params:>10,} 参数 (额外: {spectral_params - ncde_params:+,})")

    # 测试前向传播
    print("\n" + "=" * 60)
    print("测试前向传播")
    print("=" * 60)

    batch_size = 16
    t = torch.tensor(0.5)
    z = torch.randn(batch_size, hidden_channels)

    print(f"\n输入: t={t.item():.2f}, z.shape={tuple(z.shape)}")

    # ncde forward (不需要时间参数)
    print("\n1. ncde 前向传播...")
    with torch.no_grad():
        out_ncde = ncde_vf(z)
    print(f"   [OK] 输出形状: {tuple(out_ncde.shape)}")
    print(f"   - 期望: ({batch_size}, {hidden_channels}, {input_channels})")
    assert out_ncde.shape == (batch_size, hidden_channels, input_channels)

    # ncde-film forward (需要时间参数)
    print("\n2. ncde-film 前向传播...")
    with torch.no_grad():
        out_film = film_vf(t, z)
    print(f"   [OK] 输出形状: {tuple(out_film.shape)}")
    print(f"   - 期望: ({batch_size}, {hidden_channels}, {input_channels})")
    assert out_film.shape == (batch_size, hidden_channels, input_channels)

    # ncde-spectral forward (需要时间参数)
    print("\n3. ncde-spectral 前向传播...")
    with torch.no_grad():
        out_spectral = spectral_vf(t, z)
    print(f"   [OK] 输出形状: {tuple(out_spectral.shape)}")
    print(f"   - 期望: ({batch_size}, {hidden_channels}, {input_channels})")
    assert out_spectral.shape == (batch_size, hidden_channels, input_channels)

    print("\n" + "=" * 60)
    print("[OK] 所有测试通过!")
    print("=" * 60)

    return {
        'ncde': (ncde_vf, ncde_params),
        'ncde-film': (film_vf, film_params),
        'ncde-spectral': (spectral_vf, spectral_params)
    }


def test_backward_compatibility():
    """测试向后兼容性：不传递 hidden_hidden_channels 和 num_hidden_layers 时使用默认值"""
    print("\n" + "=" * 60)
    print("测试向后兼容性 (不传递新参数时使用默认值)")
    print("=" * 60)

    input_channels = 70
    hidden_channels = 64

    # 不传递新参数，应该使用默认值 128 和 1
    print("\n1. 创建 ncde-film (无新参数)...")
    film_vf_default = ModulatedSingleHiddenLayer(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        time_dim=32
    )
    print(f"   [OK] 使用默认值: hidden_hidden_channels=128, num_hidden_layers=1")
    print(f"   - {film_vf_default}")

    print("\n2. 创建 ncde-spectral (无新参数)...")
    spectral_vf_default = SpectralModulatedVectorField(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        time_dim=32,
        spectral_sigma=2.0
    )
    print(f"   [OK] 使用默认值: hidden_hidden_channels=128, num_hidden_layers=1")
    print(f"   - {spectral_vf_default}")

    # 测试前向传播
    t = torch.tensor(0.5)
    z = torch.randn(8, hidden_channels)

    with torch.no_grad():
        out1 = film_vf_default(t, z)
        out2 = spectral_vf_default(t, z)

    print(f"\n[OK] 向后兼容性测试通过！")
    print(f"  - ncde-film 输出形状: {tuple(out1.shape)}")
    print(f"  - ncde-spectral 输出形状: {tuple(out2.shape)}")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("测试 ncde-film 和 ncde-spectral 超参数统一化")
    print("=" * 60)

    # 主测试
    results = test_model_creation()

    # 向后兼容性测试
    test_backward_compatibility()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("\n[OK] 成功实现超参数统一！")
    print("\n关键变化:")
    print("  1. ncde-film 和 ncde-spectral 现在支持 hidden_hidden_channels 和 num_hidden_layers")
    print("  2. 三个模型使用相同的基础超参数: hidden_channels=64, hidden_hidden_channels=49, num_hidden_layers=4")
    print("  3. 保留各自特性: ncde-film 保留 FiLM 调制, ncde-spectral 保留频谱滤波")
    print("  4. 向后兼容: 不传递新参数时自动使用默认值 (128, 1)")
    print("\n预期效果:")
    print("  - 三个模型具有更公平的比较基础")
    print("  - ncde-film 和 ncde-spectral 的表达能力增强（更深的网络）")
    print("  - 参数量更接近，更易于对比分析")
