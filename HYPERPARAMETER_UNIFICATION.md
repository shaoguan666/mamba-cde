# 超参数统一化说明

## 修改概述

成功将 `ncde-film` 和 `ncde-spectral` 的超参数设置与 `ncde` 统一，同时保留了各自的核心特性。

## 修改内容

### 1. 向量场架构更新

#### ModulatedSingleHiddenLayer (ncde-film)
**修改前:**
- 硬编码 `hidden_hidden = 128`
- 单层结构

**修改后:**
- 支持可配置的 `hidden_hidden_channels` 参数
- 支持可配置的 `num_hidden_layers` 参数
- 多层 MLP 架构（与 FinalTanh 一致）
- 保留 FiLM 时间调制机制

#### SpectralModulatedVectorField (ncde-spectral)
**修改前:**
- 硬编码 `hidden_hidden = 128`
- 单层结构

**修改后:**
- 支持可配置的 `hidden_hidden_channels` 参数
- 支持可配置的 `num_hidden_layers` 参数
- 多层 MLP 架构（与 FinalTanh 一致）
- 保留频谱滤波 + FiLM 混合机制

### 2. 超参数配置更新

#### sepsis.py 中的配置

**修改前:**
```python
model_kwargs = dict(
    ncde=dict(hidden_channels=64, hidden_hidden_channels=49, num_hidden_layers=4),
    **{'ncde-film': dict(hidden_channels=64, hidden_hidden_channels=None, num_hidden_layers=None)},
    **{'ncde-spectral': dict(hidden_channels=64, hidden_hidden_channels=None, num_hidden_layers=None)},
    ...
)
```

**修改后:**
```python
model_kwargs = dict(
    ncde=dict(hidden_channels=64, hidden_hidden_channels=49, num_hidden_layers=4),
    **{'ncde-film': dict(hidden_channels=64, hidden_hidden_channels=49, num_hidden_layers=4)},
    **{'ncde-spectral': dict(hidden_channels=64, hidden_hidden_channels=49, num_hidden_layers=4)},
    ...
)
```

### 3. 修改的文件列表

1. **NeuralCDE-master/experiments/models/vector_fields.py**
   - 更新 `ModulatedSingleHiddenLayer` 类
   - 更新 `SpectralModulatedVectorField` 类

2. **NeuralCDE-master/experiments/common.py**
   - 在 `make_model` 函数中传递 `hidden_hidden_channels` 和 `num_hidden_layers` 参数

3. **NeuralCDE-master/experiments/sepsis.py**
   - 更新 `run_all` 函数中的 `model_kwargs` 配置

## 统一后的超参数对比

| 模型 | hidden_channels | hidden_hidden_channels | num_hidden_layers | 特有参数 |
|------|-----------------|------------------------|-------------------|----------|
| **ncde** | 64 | 49 | 4 | - |
| **ncde-film** | 64 | 49 | 4 | time_dim=32 |
| **ncde-spectral** | 64 | 49 | 4 | time_dim=32, spectral_sigma=2.0 |

## 参数量对比

| 模型 | 参数量 | 相比 ncde |
|------|--------|-----------|
| ncde | 234,535 | 基准 |
| ncde-film | 237,785 | +3,250 (+1.4%) |
| ncde-spectral | 241,004 | +6,469 (+2.8%) |

**说明:**
- ncde-film 额外参数来自：时间编码器 + FiLM 生成器
- ncde-spectral 额外参数来自：时间编码器 + FiLM 生成器 + 频域投影 + 频谱权重

## 核心特性保留

### ncde (标准 Neural CDE)
- ✓ 多层 MLP 向量场 (FinalTanh)
- ✓ 固定的非线性变换

### ncde-film (FiLM 调制)
- ✓ 多层 MLP 向量场
- ✓ FiLM 时间调制机制
- ✓ 时间编码 (time_dim=32)
- ✓ `time_aware=True` 模式

### ncde-spectral (频谱-FiLM 混合)
- ✓ 多层 MLP 向量场
- ✓ FiLM 时间调制机制
- ✓ 频域滤波 (spectral_sigma=2.0)
- ✓ 时间-频率特征融合
- ✓ `time_aware=True` 模式

## 向后兼容性

两个模块都保持了向后兼容性：

```python
# 不传递新参数时，使用默认值 (128, 1)
film_vf = ModulatedSingleHiddenLayer(
    input_channels=70,
    hidden_channels=64,
    time_dim=32
)
# hidden_hidden_channels=128 (默认), num_hidden_layers=1 (默认)

# 传递新参数时，使用指定值
film_vf = ModulatedSingleHiddenLayer(
    input_channels=70,
    hidden_channels=64,
    time_dim=32,
    hidden_hidden_channels=49,  # 与 ncde 一致
    num_hidden_layers=4          # 与 ncde 一致
)
```

## 预期效果

### 1. 更公平的模型对比
- 三个模型现在使用相同的基础架构（4层，隐藏维度49）
- 差异仅来自各自的核心特性（标准/FiLM/频谱）
- 更容易评估各种机制的独立贡献

### 2. 增强的表达能力
- ncde-film 和 ncde-spectral 从单层升级到4层
- 更强的非线性建模能力
- 可能带来性能提升

### 3. 参数量更接近
- 三个模型的参数量差异在 ±3% 以内
- 更易于对比分析和性能归因

### 4. 统一的实验设置
- 所有模型使用相同的超参数搜索空间
- 减少超参数调优的工作量
- 更可靠的实验结论

## 测试验证

运行以下脚本验证修改：

```bash
python test_unified_hyperparams.py
```

测试包括：
1. ✓ 模型创建测试
2. ✓ 参数量对比
3. ✓ 前向传播测试
4. ✓ 向后兼容性测试

## 使用示例

### 训练 Sepsis 模型

```python
from NeuralCDE-master.run_sepsis import run_sepsis

# 训练单个模型
run_sepsis(model_name='ncde-film', intensity=True, device='cuda')

# 训练所有模型（使用统一的超参数）
from NeuralCDE-master.experiments.sepsis import run_all
run_all(intensity=True, device='cuda',
        model_names=('ncde', 'ncde-film', 'ncde-spectral'))
```

### 单独使用向量场

```python
import torch
from NeuralCDE-master.experiments.models import ModulatedSingleHiddenLayer

# 创建 ncde-film 向量场（与 ncde 一致的超参数）
vector_field = ModulatedSingleHiddenLayer(
    input_channels=70,
    hidden_channels=64,
    time_dim=32,
    hidden_hidden_channels=49,
    num_hidden_layers=4
)

# 前向传播
t = torch.tensor(0.5)
z = torch.randn(16, 64)
output = vector_field(t, z)  # 输出形状: (16, 64, 70)
```

## 注意事项

1. **训练时间**: 由于网络变深，训练时间可能略有增加
2. **内存使用**: 更深的网络需要更多显存，注意 batch size 调整
3. **收敛性**: 可能需要重新调整学习率或训练 epoch 数
4. **对比基准**: 与之前的实验结果对比时，需要说明超参数变化

## 未来改进方向

1. **超参数搜索**: 统一超参数后，可以进行联合超参数优化
2. **消融研究**: 容易进行深度、宽度对模型性能的影响分析
3. **模型融合**: 相似的架构更容易进行集成学习
4. **迁移学习**: 统一架构便于跨任务的模型迁移
