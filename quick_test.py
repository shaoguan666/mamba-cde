"""
Mamba-NCDE 快速测试

验证：
1. mamba-ssm是否安装
2. 模型是否可以创建
3. 前向传播是否正常
"""

import sys
import os
import torch

# 设置路径
experiments_path = os.path.join(os.path.dirname(__file__), 'NeuralCDE-master', 'experiments')
sys.path.insert(0, experiments_path)
os.chdir(experiments_path)

print("="*60)
print("Mamba-NCDE 快速测试")
print("="*60)

# 测试1: 检查mamba-ssm
print("\n[1/4] 检查mamba-ssm安装...")
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    print("✓ mamba-ssm已安装")
except ImportError as e:
    print(f"✗ mamba-ssm未安装: {e}")
    print("\n请运行: pip install mamba-ssm")
    sys.exit(1)

# 测试2: 导入模型
print("\n[2/4] 导入Mamba-NCDE模型...")
try:
    import models
    from models import MambaModulatedVectorField, MAMBA_AVAILABLE

    if not MAMBA_AVAILABLE:
        print("✗ Mamba不可用")
        sys.exit(1)

    print("✓ MambaModulatedVectorField导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 创建模型
print("\n[3/4] 创建Mamba-NCDE vector field...")
try:
    vector_field = MambaModulatedVectorField(
        input_channels=69,
        hidden_channels=64,
        time_dim=32,
        hidden_hidden_channels=49,
        num_hidden_layers=4,
        mamba_d_model=64,
        mamba_n_layer=2,
        fusion_mode='learned'
    )

    total_params = sum(p.numel() for p in vector_field.parameters())
    print(f"✓ 模型创建成功")
    print(f"  参数量: {total_params:,}")
except Exception as e:
    print(f"✗ 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 前向传播
print("\n[4/4] 测试前向传播...")
try:
    batch_size = 32
    z = torch.randn(batch_size, 64)
    t = torch.tensor(1.0)

    with torch.no_grad():
        output = vector_field(t, z)

    expected_shape = (batch_size, 64, 69)
    assert output.shape == expected_shape, f"形状错误: {output.shape} vs {expected_shape}"

    print(f"✓ 前向传播成功")
    print(f"  输入形状: {z.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
except Exception as e:
    print(f"✗ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("所有测试通过！ ✓")
print("="*60)
print("\n可以开始训练了:")
print("  python run_mamba_ncde.py")
print("  或")
print("  cd NeuralCDE-master/experiments")
print("  python run_experiment.py --dataset sepsis --model ncde-mamba")
print()
