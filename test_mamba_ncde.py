"""
Mamba-NCDE Quick Test Script

Verifies:
1. mamba-ssm installation
2. MambaModulatedVectorField creation and forward pass
3. NCDE integration

Usage: python test_mamba_ncde.py
"""

import torch
import torch.nn as nn
import sys
import os

# Fix Windows encoding issue
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NeuralCDE-master', 'experiments'))


def test_mamba_installation():
    """测试1: 检查mamba-ssm是否安装"""
    print("\n" + "="*60)
    print("测试1: 检查mamba-ssm安装和CUDA")
    print("="*60)

    try:
        from mamba_ssm.modules.mamba_simple import Mamba
        print("✓ mamba-ssm已安装")

        # 检查CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA版本: {torch.version.cuda}")
            return True
        else:
            print("✗ CUDA不可用")
            print("  警告: Mamba需要CUDA支持，测试将失败")
            print("  请在有GPU的环境中运行测试")
            return False

    except ImportError as e:
        print(f"✗ mamba-ssm未安装: {e}")
        print("\n请运行以下命令安装:")
        print("  pip install mamba-ssm")
        print("  或")
        print("  pip install causal-conv1d>=1.2.0 && pip install mamba-ssm --no-cache-dir")
        return False


def test_mamba_branch():
    """测试2: 测试MambaBranch"""
    print("\n" + "="*60)
    print("测试2: 测试MambaBranch")
    print("="*60)

    if not torch.cuda.is_available():
        print("✗ 跳过测试：CUDA不可用")
        return False

    try:
        from models.mamba_vector_field import MambaBranch

        # 创建MambaBranch并移到GPU
        hidden_channels = 64
        d_model = 64
        n_layer = 2
        device = 'cuda'

        branch = MambaBranch(
            hidden_channels=hidden_channels,
            d_model=d_model,
            n_layer=n_layer
        ).to(device)

        print(f"✓ MambaBranch创建成功")
        print(f"  - hidden_channels: {hidden_channels}")
        print(f"  - d_model: {d_model}")
        print(f"  - n_layer: {n_layer}")
        print(f"  - device: {device}")

        # 测试前向传播 - 在GPU上创建张量
        batch_size = 32
        z = torch.randn(batch_size, hidden_channels, device=device)
        t = torch.tensor(1.0, device=device)

        with torch.no_grad():
            output = branch(z, t)

        print(f"✓ 前向传播成功")
        print(f"  - 输入形状: {z.shape}")
        print(f"  - 输出形状: {output.shape}")
        print(f"  - 输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")

        assert output.shape == z.shape, "输出形状应该与输入相同"
        print(f"✓ 形状检查通过")

        return True

    except Exception as e:
        print(f"✗ MambaBranch测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mamba_vector_field():
    """测试3: 测试MambaModulatedVectorField"""
    print("\n" + "="*60)
    print("测试3: 测试MambaModulatedVectorField")
    print("="*60)

    if not torch.cuda.is_available():
        print("✗ 跳过测试：CUDA不可用")
        return False

    try:
        from models.mamba_vector_field import MambaModulatedVectorField

        # 参数
        input_channels = 69  # Sepsis数据集
        hidden_channels = 64
        time_dim = 32
        hidden_hidden_channels = 49
        num_hidden_layers = 4
        mamba_d_model = 64
        mamba_n_layer = 2
        device = 'cuda'

        # 创建vector field并移到GPU
        vector_field = MambaModulatedVectorField(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            time_dim=time_dim,
            hidden_hidden_channels=hidden_hidden_channels,
            num_hidden_layers=num_hidden_layers,
            mamba_d_model=mamba_d_model,
            mamba_n_layer=mamba_n_layer,
            fusion_mode='learned'
        ).to(device)

        print(f"✓ MambaModulatedVectorField创建成功")
        print(f"  - input_channels: {input_channels}")
        print(f"  - hidden_channels: {hidden_channels}")
        print(f"  - mamba_d_model: {mamba_d_model}")
        print(f"  - mamba_n_layer: {mamba_n_layer}")
        print(f"  - fusion_mode: learned")
        print(f"  - device: {device}")

        # 计算参数量
        total_params = sum(p.numel() for p in vector_field.parameters())
        trainable_params = sum(p.numel() for p in vector_field.parameters() if p.requires_grad)
        print(f"  - 总参数量: {total_params:,}")
        print(f"  - 可训练参数: {trainable_params:,}")

        # 测试前向传播 - 在GPU上创建张量
        batch_size = 32
        z = torch.randn(batch_size, hidden_channels, device=device)
        t = torch.tensor(1.0, device=device)

        with torch.no_grad():
            output = vector_field(t, z)

        expected_shape = (batch_size, hidden_channels, input_channels)
        print(f"✓ 前向传播成功")
        print(f"  - 输入z形状: {z.shape}")
        print(f"  - 输出形状: {output.shape}")
        print(f"  - 期望形状: {expected_shape}")
        print(f"  - 输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")

        assert output.shape == expected_shape, f"输出形状错误: {output.shape} vs {expected_shape}"
        print(f"✓ 形状检查通过")

        # 检查输出是否bounded（应该在tanh范围内）
        assert output.min() >= -1.1 and output.max() <= 1.1, "输出应该被tanh约束在[-1, 1]附近"
        print(f"✓ 输出bounding检查通过")

        return True

    except Exception as e:
        print(f"✗ MambaModulatedVectorField测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_branch_contributions():
    """测试4: 测试分支贡献分析功能"""
    print("\n" + "="*60)
    print("测试4: 测试分支贡献分析")
    print("="*60)

    if not torch.cuda.is_available():
        print("✗ 跳过测试：CUDA不可用")
        return False

    try:
        from models.mamba_vector_field import MambaModulatedVectorField

        device = 'cuda'

        # 创建vector field并移到GPU
        vector_field = MambaModulatedVectorField(
            input_channels=69,
            hidden_channels=64,
            fusion_mode='learned'
        ).to(device)

        # 测试数据 - 在GPU上创建
        batch_size = 8
        z = torch.randn(batch_size, 64, device=device)
        t = torch.tensor(1.0, device=device)

        # 获取分支贡献
        with torch.no_grad():
            contrib = vector_field.get_branch_contributions(t, z)

        print(f"✓ 分支贡献分析成功")
        print(f"  - time_branch形状: {contrib['time_branch'].shape}")
        print(f"  - mamba_branch形状: {contrib['mamba_branch'].shape}")
        print(f"  - final_output形状: {contrib['final_output'].shape}")

        # 计算L2范数
        time_norm = torch.norm(contrib['time_branch']).item()
        mamba_norm = torch.norm(contrib['mamba_branch']).item()
        final_norm = torch.norm(contrib['final_output']).item()

        print(f"  - Time分支L2范数: {time_norm:.4f}")
        print(f"  - Mamba分支L2范数: {mamba_norm:.4f}")
        print(f"  - 最终输出L2范数: {final_norm:.4f}")

        if contrib['fusion_alpha'] is not None:
            alpha = contrib['fusion_alpha'].mean().item()
            print(f"  - 融合权重alpha: {alpha:.4f}")
            print(f"    (alpha接近1表示Time分支主导，接近0表示Mamba分支主导)")

        print(f"✓ 分支贡献分析检查通过")

        return True

    except Exception as e:
        print(f"✗ 分支贡献分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_fusion_modes():
    """测试5: 测试不同融合模式"""
    print("\n" + "="*60)
    print("测试5: 测试不同融合模式")
    print("="*60)

    if not torch.cuda.is_available():
        print("✗ 跳过测试：CUDA不可用")
        return False

    fusion_modes = ['learned', 'fixed', 'add']

    try:
        from models.mamba_vector_field import MambaModulatedVectorField

        device = 'cuda'

        for mode in fusion_modes:
            print(f"\n测试融合模式: {mode}")

            vf = MambaModulatedVectorField(
                input_channels=69,
                hidden_channels=64,
                fusion_mode=mode
            ).to(device)

            # 测试前向传播 - 在GPU上创建张量
            z = torch.randn(16, 64, device=device)
            t = torch.tensor(1.0, device=device)

            with torch.no_grad():
                output = vf(t, z)

            print(f"  ✓ {mode}模式工作正常，输出形状: {output.shape}")

        print(f"\n✓ 所有融合模式测试通过")
        return True

    except Exception as e:
        print(f"✗ 融合模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """测试6: 测试梯度流"""
    print("\n" + "="*60)
    print("测试6: 测试梯度流")
    print("="*60)

    if not torch.cuda.is_available():
        print("✗ 跳过测试：CUDA不可用")
        return False

    try:
        from models.mamba_vector_field import MambaModulatedVectorField

        device = 'cuda'

        vector_field = MambaModulatedVectorField(
            input_channels=69,
            hidden_channels=64,
            fusion_mode='learned'
        ).to(device)

        # 创建需要梯度的输入 - 在GPU上
        z = torch.randn(16, 64, requires_grad=True, device=device)
        t = torch.tensor(1.0, device=device)

        # 前向传播
        output = vector_field(t, z)

        # 计算损失（简单的L2损失）
        loss = output.pow(2).mean()

        # 反向传播
        loss.backward()

        print(f"✓ 梯度计算成功")
        print(f"  - 损失值: {loss.item():.6f}")

        # 检查梯度
        has_grad = z.grad is not None
        print(f"  - 输入z有梯度: {has_grad}")

        # 检查模型参数梯度
        params_with_grad = sum(1 for p in vector_field.parameters() if p.grad is not None)
        total_params = sum(1 for p in vector_field.parameters())
        print(f"  - 有梯度的参数: {params_with_grad}/{total_params}")

        # 计算梯度范数
        total_norm = 0
        for p in vector_field.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        print(f"  - 总梯度范数: {total_norm:.6f}")

        assert has_grad, "输入应该有梯度"
        assert params_with_grad > 0, "至少应该有一些参数有梯度"
        print(f"✓ 梯度流检查通过")

        return True

    except Exception as e:
        print(f"✗ 梯度流测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Mamba-NCDE 快速测试")
    print("="*60)

    tests = [
        ("mamba-ssm安装", test_mamba_installation),
        ("MambaBranch", test_mamba_branch),
        ("MambaModulatedVectorField", test_mamba_vector_field),
        ("分支贡献分析", test_branch_contributions),
        ("融合模式", test_different_fusion_modes),
        ("梯度流", test_gradient_flow),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ 测试 '{name}' 遇到未预期的错误: {e}")
            results[name] = False

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {name}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！Mamba-NCDE已准备就绪。")
        print("\n下一步:")
        print("  1. 查看 MAMBA_NCDE_INTEGRATION_PLAN.md 了解架构设计")
        print("  2. 查看 MAMBA_NCDE_USAGE_GUIDE.md 学习如何使用")
        print("  3. 运行实际的Sepsis实验")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息并修复。")

    print("="*60 + "\n")

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
