# Mamba-NCDE集成架构设计方案

## 问题诊断总结
当前SpectralModulatedVectorField存在严重问题：
- **Spectral分支贡献率仅2.9%**（最大12%）
- 性能提升微乎其微（AUROC 0.888 vs baseline 0.879）
- 参数增加32%但效果不明显
- 训练不稳定（accuracy和AUROC背离）

## 集成方案：Mamba-Modulated Vector Field

### 核心思想
用**Mamba状态空间模型**替代失效的**Spectral频谱分支**，形成双分支架构：
- **Local Branch（时间分支）**：原有的time-dependent MLP，捕捉瞬时动态
- **Global Branch（Mamba分支）**：Mamba Block，捕捉长程时序依赖

### 架构对比

```
当前架构（失效）：
┌─────────────────────────────────────────┐
│  SpectralModulatedVectorField           │
├─────────────────────────────────────────┤
│  Time Branch:  MLP + FiLM(t)            │
│  Spectral Branch: FFT + Gaussian RBF    │ ← 贡献仅3%
│  Fusion: Element-wise Addition          │
└─────────────────────────────────────────┘

新架构（Mamba-NCDE）：
┌─────────────────────────────────────────┐
│  MambaModulatedVectorField              │
├─────────────────────────────────────────┤
│  Time Branch:  MLP + FiLM(t)            │
│  Mamba Branch: Mamba Block(z, t)        │ ← 全局序列建模
│  Fusion: Learnable weighted sum         │
└─────────────────────────────────────────┘
```

## 详细设计

### 1. Mamba Branch设计

```python
class MambaBranch(nn.Module):
    """
    用Mamba捕捉隐状态z的长程时序依赖
    """
    def __init__(self, hidden_channels, d_model, n_layer=2):
        """
        Args:
            hidden_channels: NCDE隐状态维度
            d_model: Mamba内部维度
            n_layer: Mamba层数
        """
        super().__init__()

        # 投影到Mamba维度
        self.proj_in = nn.Linear(hidden_channels, d_model)

        # Mamba核心Block
        self.mamba_blocks = nn.ModuleList([
            create_block(d_model, layer_idx=i)
            for i in range(n_layer)
        ])

        # 投影回NCDE维度
        self.proj_out = nn.Linear(d_model, hidden_channels)

    def forward(self, z, t):
        """
        Args:
            z: [..., hidden_channels] - NCDE隐状态
            t: scalar or [...] - 当前时间
        Returns:
            [..., hidden_channels] - Mamba处理后的特征
        """
        # z: [..., hidden_channels] -> [..., d_model]
        h = self.proj_in(z)

        # Mamba处理（需要sequence维度）
        # 如果z是[batch, hidden_channels]，需要添加seq维度
        if h.dim() == 2:
            h = h.unsqueeze(1)  # [batch, 1, d_model]
            squeeze_seq = True
        else:
            squeeze_seq = False

        # 通过Mamba Blocks
        residual = None
        for block in self.mamba_blocks:
            h, residual = block(h, residual)

        if squeeze_seq:
            h = h.squeeze(1)  # [batch, d_model]

        # 投影回原维度
        return self.proj_out(h)  # [..., hidden_channels]
```

### 2. Mamba-Modulated Vector Field

```python
class MambaModulatedVectorField(nn.Module):
    """
    双分支Vector Field：Time Branch + Mamba Branch
    """
    def __init__(self, input_channels, hidden_channels,
                 time_dim=32, hidden_hidden_channels=49,
                 num_hidden_layers=4, mamba_d_model=64,
                 mamba_n_layer=2, fusion_mode='learned'):
        super().__init__()

        # Time Encoder
        self.time_encoder = TimeEncoder(time_dim)

        # === Time Branch (Local) ===
        self.film_generator = nn.Linear(time_dim, 2 * hidden_hidden_channels)
        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = nn.ModuleList([
            nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
            for _ in range(num_hidden_layers - 1)
        ])
        self.linear_out = nn.Linear(hidden_hidden_channels,
                                     input_channels * hidden_channels)

        # === Mamba Branch (Global) ===
        self.mamba_branch = MambaBranch(
            hidden_channels, mamba_d_model, mamba_n_layer
        )
        self.mamba_out = nn.Linear(hidden_channels,
                                     input_channels * hidden_channels)

        # === Fusion ===
        self.fusion_mode = fusion_mode
        if fusion_mode == 'learned':
            # 可学习的融合权重
            self.fusion_gate = nn.Sequential(
                nn.Linear(time_dim + hidden_channels, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        # FiLM初始化为恒等变换
        nn.init.zeros_(self.film_generator.weight)
        nn.init.zeros_(self.film_generator.bias)

        # Fusion gate初始化为0.5（平衡两个分支）
        if self.fusion_mode == 'learned':
            with torch.no_grad():
                self.fusion_gate[-2].bias.fill_(0.0)  # sigmoid(0) = 0.5

    def forward(self, t, z):
        """
        Args:
            t: 时间
            z: [..., hidden_channels] - NCDE隐状态
        Returns:
            [..., hidden_channels, input_channels] - vector field
        """
        batch_shape = z.shape[:-1]

        # Time encoding
        if t.dim() == 0:
            t_expanded = t.expand(z.shape[0] if z.dim() > 1 else 1)
        else:
            t_expanded = t
        time_enc = self.time_encoder(t_expanded)

        # === Time Branch ===
        film_params = self.film_generator(time_enc)
        gamma = film_params[..., :self.hidden_hidden_channels]
        beta = film_params[..., self.hidden_hidden_channels:]

        # MLP with FiLM
        h_time = self.linear_in(z)
        h_time = (1 + gamma) * h_time + beta
        h_time = torch.relu(h_time)

        for linear in self.linears:
            h_time = torch.relu(linear(h_time))

        out_time = self.linear_out(h_time)  # [..., input_channels * hidden_channels]
        out_time = torch.tanh(out_time)  # 重要：输出bounding

        # === Mamba Branch ===
        h_mamba = self.mamba_branch(z, t)  # [..., hidden_channels]
        out_mamba = self.mamba_out(h_mamba)  # [..., input_channels * hidden_channels]
        out_mamba = torch.tanh(out_mamba)  # 重要：输出bounding

        # === Fusion ===
        if self.fusion_mode == 'learned':
            # 动态融合权重，基于时间和状态
            gate_input = torch.cat([time_enc, z], dim=-1)
            alpha = self.fusion_gate(gate_input)  # [..., 1]
            out = alpha * out_time + (1 - alpha) * out_mamba
        elif self.fusion_mode == 'fixed':
            # 固定权重（可调参）
            out = 0.7 * out_time + 0.3 * out_mamba
        else:  # 'add'
            out = out_time + out_mamba

        # Reshape
        out = out.view(*batch_shape, self.hidden_channels, self.input_channels)
        return out
```

### 3. 训练配置

```python
# experiments/sepsis.py 中的修改
def get_mamba_ncde_config():
    return {
        'model': 'ncde',
        'vector_field': 'mamba_modulated',  # 新增

        # NCDE配置
        'hidden_channels': 64,
        'hidden_hidden_channels': 49,
        'num_hidden_layers': 4,

        # Mamba配置
        'mamba_d_model': 64,  # Mamba内部维度
        'mamba_n_layer': 2,   # Mamba层数（建议2-4）

        # FiLM配置
        'time_dim': 32,

        # Fusion配置
        'fusion_mode': 'learned',  # 'learned', 'fixed', 'add'

        # 训练配置
        'batch_size': 1024,
        'lr': 0.002,  # 比之前略小
        'pos_weight': 15,  # 增加正类权重

        # 正则化
        'gradient_clip': 1.0,  # 新增梯度裁剪
        'weight_decay': 1e-5,
    }
```

## 优势分析

### 1. 解决Spectral分支失效问题
- **Mamba天然适合序列建模**：状态空间模型专为长序列设计
- **动态依赖捕捉**：Mamba通过选择性状态传播捕捉重要模式
- **比频谱更适合医疗数据**：不依赖傅里叶假设

### 2. 保留NCDE优势
- **连续时间建模**：ODE求解器保持连续动态
- **不规则采样处理**：通过插值处理缺失
- **可解释性**：Vector field仍可可视化

### 3. 架构协同
```
时间尺度互补：
- Time Branch: 局部瞬时动态（dt级别）
- Mamba Branch: 全局序列模式（整个trajectory）

特征互补：
- Time Branch: 时间调制特征
- Mamba Branch: 上下文感知特征
```

### 4. 计算效率
- **Mamba高效**：O(L)复杂度 vs Transformer的O(L²)
- **轻量级集成**：只在vector field中添加Mamba
- **可并行**：两个分支可独立前向传播

## 实施路线图

### Phase 1: 核心实现（1-2天）
- [ ] 实现`MambaBranch`
- [ ] 实现`MambaModulatedVectorField`
- [ ] 在`vector_fields.py`中注册
- [ ] 在`common.py`中添加模型创建逻辑

### Phase 2: 基础实验（1-2天）
- [ ] Sepsis数据集单次训练验证
- [ ] 与baseline对比（FinalTanh）
- [ ] 与SpectralModulated对比
- [ ] 检查训练稳定性

### Phase 3: 调优（2-3天）
- [ ] 超参数搜索（mamba_d_model, mamba_n_layer）
- [ ] Fusion策略对比（learned vs fixed vs add）
- [ ] 学习率和正则化调优
- [ ] 处理类别不平衡（focal loss等）

### Phase 4: 分析与可视化（1-2天）
- [ ] Contribution ratio分析（Time vs Mamba）
- [ ] Fusion gate可视化
- [ ] 消融实验（只用Time/只用Mamba）
- [ ] 特征重要性分析

### Phase 5: 扩展实验（可选）
- [ ] 其他数据集验证
- [ ] 不同Mamba变体（Mamba2等）
- [ ] 多任务学习
- [ ] 实时推理性能测试

## 预期改进

基于当前问题分析，预期改进：

1. **性能提升**：
   - AUROC：0.888 → 0.91+ （目标提升2-3%）
   - AP：0.494 → 0.52+ （类别不平衡改善）
   - Contribution ratio：3% → 30%+ （Mamba分支有效）

2. **训练稳定性**：
   - 消除accuracy/AUROC背离
   - 减少epoch间波动
   - 梯度裁剪防止爆炸

3. **可解释性**：
   - Fusion gate权重变化反映时间动态
   - Mamba状态可追踪序列模式
   - 双分支贡献可独立分析

## 风险与缓解

| 风险 | 影响 | 缓解策略 |
|------|------|----------|
| Mamba依赖mamba-ssm库 | 环境配置 | 提供docker/conda环境配置 |
| 内存占用增加 | 训练成本 | 使用梯度检查点、减小batch size |
| 融合策略不当 | 性能不佳 | 多种fusion mode可选 |
| 过拟合 | 泛化性差 | 添加dropout、weight decay |

## 依赖安装

```bash
# 安装mamba-ssm（需要CUDA）
pip install mamba-ssm

# 或从源码安装
pip install causal-conv1d>=1.2.0
pip install mamba-ssm --no-cache-dir

# 验证安装
python -c "from mamba_ssm.modules.mamba_simple import Mamba; print('Mamba installed!')"
```

## 参考文献

1. **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
2. **NeuralCDE**: Kidger et al., "Neural Controlled Differential Equations for Irregular Time Series", 2020
3. **ApricotM**: Medical time series with Mamba（你的参考实现）
4. **FiLM**: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", 2018

## 后续优化方向

1. **Mamba2集成**：使用最新的Mamba2架构
2. **多分辨率建模**：不同Mamba层捕捉不同时间尺度
3. **注意力增强**：在fusion处添加cross-attention
4. **预训练**：在大规模医疗数据上预训练Mamba分支
5. **联邦学习**：分布式训练保护隐私

---

**总结**：Mamba-NCDE集成是解决当前Spectral分支失效的最佳方案，兼顾性能、效率和可解释性。
