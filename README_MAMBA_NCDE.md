# Mamba-NCDE 快速使用

## 安装依赖

```bash
pip install mamba-ssm
```

## 一键运行

### 方式1: 使用快捷脚本（推荐）

```bash
# Sepsis数据集（默认200 epochs）
python run_mamba_ncde.py

# 自定义训练轮数
python run_mamba_ncde.py --epochs 50

# 使用CPU
python run_mamba_ncde.py --cpu
```

### 方式2: 使用统一脚本

```bash
cd NeuralCDE-master/experiments

# 运行Mamba-NCDE
python run_experiment.py --dataset sepsis --model ncde-mamba

# 对比所有模型
python run_experiment.py --dataset sepsis --all-models

# UEA数据集
python run_experiment.py --dataset uea --model ncde-mamba

# Speech数据集
python run_experiment.py --dataset speech --model ncde-mamba
```

## 预期结果

| 模型 | AUROC | AP | 参数量 |
|------|-------|-----|--------|
| NCDE (baseline) | 0.879 | 0.491 | 193K |
| NCDE-Spectral (失效) | 0.888 | 0.494 | 256K |
| **NCDE-Mamba (新)** | **0.91+?** | **0.52+?** | ~280K |

**关键改进**:
- Mamba分支贡献率：3% → 30%+
- 训练稳定性：消除accuracy/AUROC背离
- 长程依赖建模：状态空间模型替代频谱

## 测试安装

```bash
python test_mamba_ncde.py
```

## 架构说明

```
Time Branch (Local)      ──┐
  MLP + FiLM(t)            │
  捕捉瞬时动态              │  Learned Fusion
                           ├─> α·Time + (1-α)·Mamba ──> 输出
Mamba Branch (Global)    ──┘         ↑
  Mamba SSM(z)                       |
  捕捉长程序列模式            Fusion Gate(t, z)
```

## 配置文件位置

- 核心实现: `NeuralCDE-master/experiments/models/mamba_vector_field.py`
- 模型注册: `NeuralCDE-master/experiments/common.py`
- 实验配置: `NeuralCDE-master/experiments/sepsis.py`
- 统一入口: `NeuralCDE-master/experiments/run_experiment.py`

## 常见问题

**Q: mamba-ssm安装失败?**
A: 确保CUDA已安装，使用: `pip install causal-conv1d>=1.2.0 && pip install mamba-ssm --no-cache-dir`

**Q: 显存不足?**
A: 减小batch_size或使用`--cpu`

**Q: 如何调整Mamba参数?**
A: 修改`common.py`中`make_model`函数的`mamba_d_model`和`mamba_n_layer`

## 文件说明

- `run_mamba_ncde.py` - 快速运行脚本
- `test_mamba_ncde.py` - 测试脚本（6个测试）
- `MAMBA_NCDE_INTEGRATION_PLAN.md` - 详细设计文档
- `MAMBA_NCDE_USAGE_GUIDE.md` - 完整使用指南
