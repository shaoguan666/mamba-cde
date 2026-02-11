# Mamba-NCDE 集成完成 ✓

## 已完成的集成

### 修改的文件:
1. ✓ `NeuralCDE-master/experiments/models/vector_fields.py` - 添加Mamba导入
2. ✓ `NeuralCDE-master/experiments/models/__init__.py` - 导出MambaModulatedVectorField
3. ✓ `NeuralCDE-master/experiments/models/mamba_vector_field.py` - **核心实现**
4. ✓ `NeuralCDE-master/experiments/common.py` - 添加'ncde-mamba'模型
5. ✓ `NeuralCDE-master/experiments/sepsis.py` - 添加mamba配置
6. ✓ `NeuralCDE-master/experiments/run_experiment.py` - 添加命令行支持

### 新增的文件:
- `run_mamba_ncde.py` - 一键运行脚本
- `quick_test.py` - 快速测试脚本（4个测试）
- `README_MAMBA_NCDE.md` - 使用说明

## 快速开始（3步）

### 1. 安装依赖
```bash
pip install mamba-ssm
```

### 2. 测试安装
```bash
python quick_test.py
```

### 3. 运行实验
```bash
# 方式A: 使用快捷脚本（推荐）
python run_mamba_ncde.py

# 方式B: 使用统一脚本
cd NeuralCDE-master/experiments
python run_experiment.py --dataset sepsis --model ncde-mamba
```

## 命令行选项

```bash
# 基础用法
python run_mamba_ncde.py

# 自定义训练轮数
python run_mamba_ncde.py --epochs 50

# 使用CPU
python run_mamba_ncde.py --cpu

# 重复5次实验
python run_mamba_ncde.py --repeats 5

# 对比所有模型（包括mamba）
cd NeuralCDE-master/experiments
python run_experiment.py --dataset sepsis --all-models
```

## 预期改进

基于分析，Mamba-NCDE应该解决以下问题：

| 问题 | Spectral（旧） | Mamba（新） |
|------|---------------|------------|
| 分支贡献率 | 3% | 30%+ |
| AUROC | 0.888 | 0.91+ |
| 训练稳定性 | 波动大 | 稳定 |
| 长程依赖 | 无法捕捉 | SSM建模 |

## 架构对比

```
【失效的Spectral】
Time Branch (97%) ──┐
                    ├─> Add ──> 输出
Spectral (3%) ──────┘

【新的Mamba-NCDE】
Time Branch ────────┐
  (瞬时动态)        │  Learned Fusion
                    ├─> α·Time + (1-α)·Mamba ──> 输出
Mamba Branch ───────┘         ↑
  (长程序列)            Fusion Gate(t,z)
```

## 如果遇到问题

### ImportError: mamba-ssm未安装
```bash
pip install causal-conv1d>=1.2.0
pip install mamba-ssm --no-cache-dir
```

### CUDA显存不足
- 减小batch_size（默认1024）
- 使用`--cpu`（慢但可以跑通）

### 找不到模块
确保在正确的目录运行：
```bash
# 应该在mamba-cde目录下
cd d:\实验室\mamba-cde
python run_mamba_ncde.py
```

## 调整超参数

编辑 `NeuralCDE-master/experiments/common.py` 第306-315行:

```python
elif name == 'ncde-mamba':
    def make_model():
        vector_field = models.MambaModulatedVectorField(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            time_dim=32,
            hidden_hidden_channels=hidden_hidden_channels,
            num_hidden_layers=num_hidden_layers,
            mamba_d_model=64,        # ← 调整Mamba维度
            mamba_n_layer=2,         # ← 调整Mamba层数
            fusion_mode='learned'    # ← 'learned', 'fixed', 'add'
        )
        ...
```

## 查看结果

训练完成后，结果保存在:
```
NeuralCDE-master/experiments/results/sepsis_intensity/
```

输出包含:
- 训练历史（每10 epoch）
- 最终测试指标（AUROC, AP, Accuracy）
- 混淆矩阵
- 参数量

## 下一步

1. **运行单次实验**验证代码工作正常
2. **对比baseline**（ncde vs ncde-mamba）
3. **超参数搜索**找到最佳配置
4. **多次运行**获取统计结果

开始实验吧！ 🚀
