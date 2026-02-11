# Mamba-NCDE 使用指南

## 快速开始

### 1. 环境配置

```bash
# 激活你的环境
conda activate your_env

# 安装mamba-ssm（需要CUDA）
pip install mamba-ssm

# 如果遇到问题，尝试从源码安装
pip install causal-conv1d>=1.2.0
pip install mamba-ssm --no-cache-dir

# 验证安装
python -c "from mamba_ssm.modules.mamba_simple import Mamba; print('Mamba安装成功!')"
```

### 2. 集成到现有代码

#### 2.1 修改 `experiments/models/vector_fields.py`

在文件末尾添加导入：

```python
# 在vector_fields.py最后添加
try:
    from .mamba_vector_field import (
        MambaModulatedVectorField,
        create_mamba_modulated_vector_field
    )
except ImportError as e:
    print(f"警告: Mamba vector field未加载: {e}")
    MambaModulatedVectorField = None
```

#### 2.2 修改 `experiments/common.py`

在`get_vector_field`函数中添加Mamba选项：

```python
def get_vector_field(name, input_channels, hidden_channels, **kwargs):
    """获取vector field"""
    if name == 'final_tanh':
        from .models.vector_fields import FinalTanh
        return FinalTanh(input_channels, hidden_channels,
                        hidden_hidden_channels=kwargs.get('hidden_hidden_channels'),
                        num_hidden_layers=kwargs.get('num_hidden_layers'))

    elif name == 'spectral_modulated':
        from .models.vector_fields import SpectralModulatedVectorField
        return SpectralModulatedVectorField(
            input_channels, hidden_channels,
            time_dim=kwargs.get('time_dim', 32),
            spectral_sigma=kwargs.get('spectral_sigma', 2.0),
            hidden_hidden_channels=kwargs.get('hidden_hidden_channels'),
            num_hidden_layers=kwargs.get('num_hidden_layers')
        )

    # ====== 新增：Mamba-NCDE ======
    elif name == 'mamba_modulated':
        from .models.vector_fields import MambaModulatedVectorField
        if MambaModulatedVectorField is None:
            raise ImportError("Mamba vector field未安装，请先安装mamba-ssm")
        return MambaModulatedVectorField(
            input_channels, hidden_channels,
            time_dim=kwargs.get('time_dim', 32),
            hidden_hidden_channels=kwargs.get('hidden_hidden_channels', 49),
            num_hidden_layers=kwargs.get('num_hidden_layers', 4),
            mamba_d_model=kwargs.get('mamba_d_model', 64),
            mamba_n_layer=kwargs.get('mamba_n_layer', 2),
            fusion_mode=kwargs.get('fusion_mode', 'learned'),
            mamba_dropout=kwargs.get('mamba_dropout', 0.1)
        )

    else:
        raise ValueError(f"未知的vector field: {name}")
```

#### 2.3 创建实验配置文件

创建 `experiments/sepsis_mamba.py`:

```python
"""
Sepsis预测实验 - Mamba-NCDE版本
"""
import common
import numpy as np

def main():
    # 基础配置
    intensity = True
    device = 'cuda'

    # 超参数配置
    config = {
        # 模型配置
        'model': 'ncde',
        'vector_field': 'mamba_modulated',  # 使用Mamba-NCDE

        # NCDE配置
        'hidden_channels': 64,
        'hidden_hidden_channels': 49,
        'num_hidden_layers': 4,

        # Mamba配置
        'mamba_d_model': 64,     # Mamba内部维度
        'mamba_n_layer': 2,      # Mamba Block层数
        'mamba_dropout': 0.1,    # Mamba dropout

        # FiLM配置
        'time_dim': 32,

        # Fusion配置
        'fusion_mode': 'learned',  # 'learned', 'fixed', 'add'

        # 训练配置
        'batch_size': 1024,
        'lr': 0.002,             # 学习率（比spectral略小）
        'epochs': 200,

        # 正则化
        'weight_decay': 1e-5,
        'gradient_clip': 1.0,    # 梯度裁剪（重要！）

        # 类别不平衡
        'pos_weight': 15,        # 正类权重
    }

    # 加载数据
    times, train_dataloader, val_dataloader, test_dataloader = common.get_data(
        dataset_name='sepsis',
        missing_rate=0.5,
        device=device,
        intensity=intensity,
        batch_size=config['batch_size']
    )

    # 获取数据维度
    data_size = 0
    for batch in train_dataloader:
        batch = tuple(b.to(device) for b in batch)
        *coeffs, _, true_y = batch
        data_size = coeffs[0].shape[-1] - 1  # 减去时间维度
        break

    # 创建模型
    input_channels = data_size
    model_maker = common.make_model_mamba_ncde  # 需要创建这个函数

    model, vector_field = model_maker(
        input_channels=input_channels,
        config=config,
        device=device
    )

    print(f"\n{'='*60}")
    print(f"Mamba-NCDE 实验配置")
    print(f"{'='*60}")
    print(f"数据集: Sepsis (intensity={intensity})")
    print(f"输入通道: {input_channels}")
    print(f"隐藏通道: {config['hidden_channels']}")
    print(f"Mamba维度: {config['mamba_d_model']}")
    print(f"Mamba层数: {config['mamba_n_layer']}")
    print(f"Fusion模式: {config['fusion_mode']}")
    print(f"学习率: {config['lr']}")
    print(f"批量大小: {config['batch_size']}")
    print(f"{'='*60}\n")

    # 训练
    result = common.train(
        model=model,
        vector_field=vector_field,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        times=times,
        config=config,
        device=device,
        num_classes=2,
        task='classification'
    )

    # 保存结果
    result_path = f'results/sepsis_mamba_ncde_{config["fusion_mode"]}.pt'
    import torch
    torch.save({
        'config': config,
        'result': result,
        'model_state': model.state_dict(),
        'vector_field': vector_field,
    }, result_path)

    print(f"\n结果已保存到: {result_path}")

    return result


if __name__ == '__main__':
    main()
```

#### 2.4 修改 `experiments/common.py` 添加模型创建函数

```python
def make_model_mamba_ncde(input_channels, config, device):
    """
    创建Mamba-NCDE模型

    Args:
        input_channels: 输入通道数
        config: 配置字典
        device: 设备

    Returns:
        (model, vector_field): 模型和vector field
    """
    from .models import metamodel
    from .models.vector_fields import MambaModulatedVectorField

    # 创建vector field
    vector_field = MambaModulatedVectorField(
        input_channels=input_channels,
        hidden_channels=config['hidden_channels'],
        time_dim=config.get('time_dim', 32),
        hidden_hidden_channels=config.get('hidden_hidden_channels', 49),
        num_hidden_layers=config.get('num_hidden_layers', 4),
        mamba_d_model=config.get('mamba_d_model', 64),
        mamba_n_layer=config.get('mamba_n_layer', 2),
        fusion_mode=config.get('fusion_mode', 'learned'),
        mamba_dropout=config.get('mamba_dropout', 0.1)
    ).to(device)

    # 创建NCDE模型
    model = metamodel.NeuralCDE(
        func=vector_field,
        input_channels=input_channels,
        hidden_channels=config['hidden_channels'],
        output_channels=1,  # 二分类
        initial=False
    )

    # 包装为完整模型（包含初始值网络等）
    model = create_full_model(
        model=model,
        input_channels=input_channels,
        hidden_channels=config['hidden_channels'],
        num_classes=2
    ).to(device)

    return model, vector_field


def create_full_model(model, input_channels, hidden_channels, num_classes):
    """
    创建完整的模型（包括初始值网络和分类头）

    这个函数应该已经存在于common.py中，如果没有则添加
    """
    import torch.nn as nn

    class InitialValueNetwork(nn.Module):
        def __init__(self, input_channels, hidden_channels):
            super().__init__()
            self.linear1 = nn.Linear(input_channels, 256)
            self.linear2 = nn.Linear(256, hidden_channels)

        def forward(self, x):
            x = torch.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    class FullModel(nn.Module):
        def __init__(self, initial_network, ncde, num_classes):
            super().__init__()
            self.initial_network = initial_network
            self.ncde = ncde
            self.linear = nn.Linear(ncde.hidden_channels, num_classes)

        def forward(self, coeffs, final_index, **kwargs):
            # 初始值
            z0 = self.initial_network(coeffs[0][:, 0, 1:])

            # NCDE求解
            z_T = self.ncde(z0, coeffs, final_index, **kwargs)

            # 分类
            pred = self.linear(z_T)
            return pred.squeeze(-1) if pred.shape[-1] == 1 else pred

    initial_network = InitialValueNetwork(input_channels, hidden_channels)
    full_model = FullModel(initial_network, model, num_classes)

    return full_model
```

### 3. 运行实验

```bash
# 进入实验目录
cd NeuralCDE-master/experiments

# 运行Mamba-NCDE实验
python sepsis_mamba.py

# 可选：使用不同的fusion模式
python sepsis_mamba.py --fusion-mode fixed
python sepsis_mamba.py --fusion-mode add
```

### 4. 分析结果

创建 `analyze_mamba_ncde.py`:

```python
"""
分析Mamba-NCDE的训练结果
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

def analyze_result(result_path):
    """分析训练结果"""

    # 加载结果
    result = torch.load(result_path)
    config = result['config']
    history = result['result'].history

    # 提取指标
    train_auroc = [h['train_metrics']['auroc'] for h in history]
    val_auroc = [h['val_metrics']['auroc'] for h in history]
    train_loss = [h['train_metrics']['loss'] for h in history]
    val_loss = [h['val_metrics']['loss'] for h in history]

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # AUROC曲线
    axes[0].plot(train_auroc, label='Train', linewidth=2)
    axes[0].plot(val_auroc, label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('AUROC')
    axes[0].set_title('AUROC曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss曲线
    axes[1].plot(train_loss, label='Train', linewidth=2)
    axes[1].plot(val_loss, label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mamba_ncde_training.pdf', dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: mamba_ncde_training.pdf")

    # 打印最终指标
    final_test = result['result'].test_metrics
    print(f"\n{'='*50}")
    print(f"最终测试集指标:")
    print(f"{'='*50}")
    print(f"AUROC: {final_test['auroc']:.4f}")
    print(f"AP: {final_test['average_precision']:.4f}")
    print(f"Accuracy: {final_test['accuracy']:.4f}")
    print(f"{'='*50}\n")

    return result


def compare_with_baseline(mamba_result_path, baseline_result_path):
    """对比Mamba-NCDE和baseline"""

    mamba = torch.load(mamba_result_path)
    baseline = torch.load(baseline_result_path)

    print(f"\n{'='*60}")
    print(f"Mamba-NCDE vs Baseline对比")
    print(f"{'='*60}")

    metrics = ['auroc', 'average_precision', 'accuracy']
    for metric in metrics:
        mamba_val = mamba['result'].test_metrics[metric]
        baseline_val = baseline['result'].test_metrics[metric]
        improvement = (mamba_val - baseline_val) / baseline_val * 100

        print(f"{metric.upper():<20}: Mamba={mamba_val:.4f}, "
              f"Baseline={baseline_val:.4f}, "
              f"提升={improvement:+.2f}%")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    # 分析Mamba-NCDE结果
    result = analyze_result('results/sepsis_mamba_ncde_learned.pt')

    # 对比baseline（如果有）
    # compare_with_baseline(
    #     'results/sepsis_mamba_ncde_learned.pt',
    #     'results/sepsis_intensity/0'  # baseline结果
    # )
```

## 高级用法

### 1. 可视化分支贡献

```python
"""可视化Time Branch和Mamba Branch的贡献"""
import torch
import matplotlib.pyplot as plt

def visualize_branch_contributions(model, vector_field, test_dataloader, times, device):
    """
    可视化两个分支的贡献度随时间的变化
    """
    model.eval()
    vector_field.eval()

    # 收集数据
    alphas = []
    time_norms = []
    mamba_norms = []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(b.to(device) for b in batch)
            *coeffs, final_index, true_y = batch

            # 获取初始状态
            z0 = model.initial_network(coeffs[0][:, 0, 1:])

            # 在不同时间点评估
            for t in times[::5]:  # 每5个时间步采样一次
                t_tensor = torch.tensor(t, dtype=torch.float32, device=device)

                # 获取分支贡献
                contrib = vector_field.get_branch_contributions(t_tensor, z0)

                # 计算L2范数
                time_norm = torch.norm(contrib['time_branch'], dim=(-2, -1)).mean().item()
                mamba_norm = torch.norm(contrib['mamba_branch'], dim=(-2, -1)).mean().item()

                time_norms.append(time_norm)
                mamba_norms.append(mamba_norm)

                if contrib['fusion_alpha'] is not None:
                    alphas.append(contrib['fusion_alpha'].mean().item())

            break  # 只用第一个batch演示

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 分支贡献L2范数
    ax1.plot(times[::5], time_norms, label='Time Branch', linewidth=2)
    ax1.plot(times[::5], mamba_norms, label='Mamba Branch', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('L2 Norm')
    ax1.set_title('Branch Contribution Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Fusion权重
    if alphas:
        ax2.plot(times[::5], alphas, label='Alpha (Time Branch weight)',
                color='purple', linewidth=2)
        ax2.axhline(y=0.5, color='gray', linestyle='--', label='Equal weight')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Fusion Weight Alpha')
        ax2.set_title('Dynamic Fusion Weight')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mamba_ncde_branch_analysis.pdf', dpi=300, bbox_inches='tight')
    print("分支分析图已保存到: mamba_ncde_branch_analysis.pdf")

# 使用示例
# visualize_branch_contributions(model, vector_field, test_dataloader, times, device)
```

### 2. 超参数搜索

```python
"""超参数搜索脚本"""
import itertools

def hyperparameter_search():
    """网格搜索最佳超参数"""

    # 搜索空间
    param_grid = {
        'mamba_d_model': [32, 64, 128],
        'mamba_n_layer': [1, 2, 3],
        'fusion_mode': ['learned', 'fixed'],
        'lr': [0.001, 0.002, 0.003],
        'pos_weight': [10, 15, 20],
    }

    # 生成所有组合
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))

    results = []
    for combo in combinations:
        config = dict(zip(keys, combo))
        print(f"\n测试配置: {config}")

        # 运行实验（这里简化，实际应该调用完整的训练流程）
        result = run_experiment(config)

        results.append({
            'config': config,
            'auroc': result['auroc'],
            'ap': result['ap'],
        })

    # 找到最佳配置
    best = max(results, key=lambda x: x['auroc'])
    print(f"\n最佳配置:")
    print(f"  {best['config']}")
    print(f"  AUROC: {best['auroc']:.4f}")
    print(f"  AP: {best['ap']:.4f}")

    return results
```

## 常见问题

### Q1: mamba-ssm安装失败
**A**: 确保安装了CUDA和对应的PyTorch版本：
```bash
# 检查CUDA版本
nvcc --version

# 安装对应的PyTorch
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 再安装mamba
pip install mamba-ssm
```

### Q2: 训练时显存不足
**A**: 尝试以下方法：
1. 减小batch_size
2. 减小mamba_d_model
3. 减少mamba_n_layer
4. 使用梯度检查点（需要修改代码）

### Q3: Mamba分支贡献仍然很低
**A**: 尝试：
1. 使用'fixed'融合模式，手动设置alpha=0.3给Mamba更多权重
2. 增大mamba_d_model
3. 为Mamba分支添加auxiliary loss
4. 调整Mamba的初始化

### Q4: 如何调试模型？
**A**: 使用`get_branch_contributions`方法：
```python
# 在训练循环中
if epoch % 10 == 0:
    with torch.no_grad():
        t = torch.tensor(times[len(times)//2], device=device)
        z = model.get_hidden_state(...)  # 获取某个隐状态
        contrib = vector_field.get_branch_contributions(t, z)

        print(f"Epoch {epoch}:")
        print(f"  Time branch norm: {torch.norm(contrib['time_branch']).item():.4f}")
        print(f"  Mamba branch norm: {torch.norm(contrib['mamba_branch']).item():.4f}")
        print(f"  Fusion alpha: {contrib['fusion_alpha'].mean().item():.4f}")
```

## 下一步

1. **运行实验**：按照上述步骤运行Mamba-NCDE
2. **对比baseline**：验证性能提升
3. **调优超参数**：找到最佳配置
4. **深入分析**：可视化分支贡献、融合权重等
5. **扩展应用**：尝试其他数据集

祝实验成功！ 🚀
