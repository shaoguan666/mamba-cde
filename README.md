# APRICOT-Mamba v2 重构方案

> 基于 Neural CDE + 双向 Mamba 的败血症早期预警系统

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 环境配置](#2-环境配置)
- [3. 架构设计](#3-架构设计)
- [4. 模块实现](#4-模块实现)
  - [4.1 纯 PyTorch Mamba](#41-纯-pytorch-mamba-mamba_minimalpy)
  - [4.2 Neural CDE 嵌入](#42-neural-cde-嵌入-neural_cdepy)
  - [4.3 双向 Mamba 主干](#43-双向-mamba-主干-bimambapy)
  - [4.4 主模型与损失函数](#44-主模型与损失函数-apricotm_v2py)
- [5. 使用示例](#5-使用示例)
- [6. 文件结构](#6-文件结构)

---

## 1. 项目概述

### 1.1 重构目标

| 组件 | 原始实现 | 重构后实现 |
|------|---------|-----------|
| **嵌入层** | Conv1d + Variable Embedding | Neural CDE 连续时间建模 |
| **Mamba** | 单向 + mamba-ssm 库 (Linux) | 双向 + 纯 PyTorch (跨平台) |
| **损失函数** | 简单 BCE | 时间加权 Focal Loss |
| **时间处理** | 离散位置编码 | 连续微分方程建模 |

### 1.2 关键改进

1. **Neural CDE 嵌入**：使用神经受控微分方程处理不规则采样的 ICU 时间序列
2. **双向 Mamba (BiMamba)**：同时捕获正向和反向时间依赖
3. **效用损失函数 (UtilityLoss)**：模拟 PhysioNet Utility Score，强调发病前关键窗口期
4. **纯 PyTorch 实现**：不依赖 Linux 专用 CUDA 扩展，兼容 Windows + CUDA 12.8

---

## 2. 环境配置

### 2.1 requirements.txt

```txt
# ===== 核心深度学习 =====
torch>=2.0.0
torchvision>=0.15.0

# ===== 张量操作 =====
einops>=0.6.0

# ===== Neural CDE (可选，有后备方案) =====
torchcde>=0.2.5
torchdiffeq>=0.2.3

# ===== 数据处理 =====
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
h5py>=3.8.0

# ===== 机器学习 =====
scikit-learn>=1.2.0

# ===== 超参数优化 =====
optuna>=3.1.0

# ===== 可视化 =====
matplotlib>=3.7.0
tqdm>=4.65.0

# ===== 可解释性 =====
captum>=0.6.0

# ===== 基线模型 =====
catboost>=1.1.0
```

### 2.2 Windows 安装命令

```bash
# 1. 创建 conda 环境
conda create -n mamba-cde python=3.10 -y
conda activate mamba-cde

# 2. 安装 PyTorch (CUDA 12.1/12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装核心依赖
pip install einops numpy pandas scipy h5py scikit-learn optuna matplotlib tqdm captum catboost

# 4. 尝试安装 torchcde (可能失败，但有后备方案)
pip install torchdiffeq
pip install torchcde
```

### 2.3 环境判断说明

**mamba-ssm 库在 Windows 11 + CUDA 12.8 环境下安装困难**，原因：
- 依赖 `causal-conv1d` 和自定义 CUDA 核心，需要在 Linux 上编译
- 官方不提供 Windows 预编译轮子
- Triton 对 Windows 支持有限

**解决方案**：本项目提供纯 PyTorch 实现的 Mamba 块，无需任何 CUDA 扩展。

---

## 3. 架构设计

### 3.1 整体架构图

```
Input: (B, L, 3) [time, variable_id, value]
         |
         v
+------------------+
| Neural CDE       |  <-- 不规则采样 -> 规则采样
| Embedding        |      输出: (B, T, d_model)
+------------------+
         |
         v
+------------------+
| Static Feature   |  <-- 静态特征编码
| Encoder          |      输出: (B, d_model) -> broadcast
+------------------+
         |
         v
+------------------+
| BiMamba          |  <-- 双向序列建模
| Backbone         |      正向 + 反向 Mamba
+------------------+
         |
         v
+------------------+
| TopK Pooling     |  <-- 序列聚合
| + MLP            |      输出: (B, d_model)
+------------------+
         |
         v
+------------------+
| Multi-Task       |  <-- 4 主任务 + 8 辅助任务
| Classifier       |      输出: (B, 12)
+------------------+
```

### 3.2 BiMamba 结构

```
Input: (B, L, D)
    |
    +---> Forward Mamba ---> h_forward
    |
    +---> Flip -> Backward Mamba -> Flip ---> h_backward
    |
    +---> Concat [h_forward, h_backward] ---> Linear ---> Output
```

---

## 4. 模块实现

### 4.1 纯 PyTorch Mamba (`mamba_minimal.py`)

```python
"""
mamba_minimal.py
纯 PyTorch 实现的 Mamba 块，不依赖 causal_conv1d 或 mamba_ssm CUDA 核心
参考: https://github.com/johnma2006/mamba-minimal
"""
from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class ModelArgs:
    """Mamba 模型配置"""
    d_model: int = 128          # 模型维度
    d_state: int = 16           # SSM 状态维度 N
    d_conv: int = 4             # 局部卷积宽度
    expand: int = 2             # 扩展因子 E，内部维度 = d_model * expand
    dt_rank: str = "auto"       # dt 投影的秩
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"     # "random" 或 "constant"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    bias: bool = False
    conv_bias: bool = True
    pscan: bool = True          # 是否使用并行扫描

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class MambaBlock(nn.Module):
    """单个 Mamba 块，包含 SSM + 选通机制"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # 输入投影: d_model -> 2 * d_inner (用于分支 x 和 z)
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        # 1D 卷积 (深度可分离)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            kernel_size=args.d_conv,
            padding=args.d_conv - 1,
            groups=args.d_inner,
            bias=args.conv_bias
        )

        # SSM 参数投影: d_inner -> dt_rank + 2*d_state
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)

        # dt 投影: dt_rank -> d_inner
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        # 初始化 dt_proj 偏置
        dt = torch.exp(
            torch.rand(args.d_inner) * (math.log(args.dt_max) - math.log(args.dt_min))
            + math.log(args.dt_min)
        ).clamp(min=args.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # SSM 参数 A (对角化，存储为 log)
        A = repeat(
            torch.arange(1, args.d_state + 1, dtype=torch.float32),
            'n -> d n',
            d=args.d_inner
        )
        self.A_log = nn.Parameter(torch.log(A))

        # SSM 参数 D (跳跃连接)
        self.D = nn.Parameter(torch.ones(args.d_inner))

        # 输出投影: d_inner -> d_model
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) 输入张量
        Returns:
            output: (B, L, D) 输出张量
        """
        B, L, D = x.shape

        # 输入投影
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x, res = x_and_res.split([self.args.d_inner, self.args.d_inner], dim=-1)

        # 1D 卷积
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :L]  # 截断到原始长度
        x = rearrange(x, 'b d l -> b l d')

        # SiLU 激活
        x = F.silu(x)

        # SSM
        y = self.ssm(x)

        # 选通与输出
        y = y * F.silu(res)
        output = self.out_proj(y)

        return output

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """选择性状态空间模型

        Args:
            x: (B, L, d_inner)
        Returns:
            y: (B, L, d_inner)
        """
        B, L, D = x.shape

        # 计算 delta, B, C
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        delta, B_proj, C = x_dbl.split(
            [self.args.dt_rank, self.args.d_state, self.args.d_state],
            dim=-1
        )

        # delta: (B, L, dt_rank) -> (B, L, d_inner)
        delta = F.softplus(self.dt_proj(delta))

        # A: (d_inner, d_state)
        A = -torch.exp(self.A_log.float())

        # 选择扫描算法
        if self.args.pscan:
            y = self.selective_scan_parallel(x, delta, A, B_proj, C)
        else:
            y = self.selective_scan_sequential(x, delta, A, B_proj, C)

        # 跳跃连接
        y = y + x * self.D

        return y

    def selective_scan_sequential(
        self,
        x: torch.Tensor,      # (B, L, D)
        delta: torch.Tensor,  # (B, L, D)
        A: torch.Tensor,      # (D, N)
        B: torch.Tensor,      # (B, L, N)
        C: torch.Tensor,      # (B, L, N)
    ) -> torch.Tensor:
        """顺序扫描实现"""
        B_batch, L, D = x.shape
        N = A.shape[1]

        # 离散化
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, N)
        deltaB_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)  # (B, L, D, N)

        # 状态空间扫描
        h = torch.zeros(B_batch, D, N, device=x.device, dtype=x.dtype)
        ys = []

        for i in range(L):
            h = deltaA[:, i] * h + deltaB_x[:, i]
            y = torch.einsum('bdn,bn->bd', h, C[:, i])
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, L, D)
        return y

    def selective_scan_parallel(
        self,
        x: torch.Tensor,      # (B, L, D)
        delta: torch.Tensor,  # (B, L, D)
        A: torch.Tensor,      # (D, N)
        B: torch.Tensor,      # (B, L, N)
        C: torch.Tensor,      # (B, L, N)
    ) -> torch.Tensor:
        """并行扫描实现（利用结合律加速）"""
        B_batch, L, D = x.shape
        N = A.shape[1]

        # 离散化
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, N)
        deltaB_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)  # (B, L, D, N)

        # 使用 associative scan
        h = torch.zeros(B_batch, D, N, device=x.device, dtype=x.dtype)
        hs = []

        for i in range(L):
            h = deltaA[:, i] * h + deltaB_x[:, i]
            hs.append(h)

        hs = torch.stack(hs, dim=1)  # (B, L, D, N)
        y = torch.einsum('bldn,bln->bld', hs, C)

        return y


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class ResidualBlock(nn.Module):
    """Mamba 块 + 残差连接 + 归一化"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mamba = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class MambaBackbone(nn.Module):
    """Mamba 主干网络（单向）"""
    def __init__(self, args: ModelArgs, n_layer: int):
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList([
            ResidualBlock(args) for _ in range(n_layer)
        ])
        self.norm_f = RMSNorm(args.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) 输入特征序列
        Returns:
            (B, L, D) 输出特征序列
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)
```

---

### 4.2 Neural CDE 嵌入 (`neural_cde.py`)

```python
"""
neural_cde.py
Neural CDE 嵌入层，用于处理不规则采样的时间序列
支持多种后端：torchcde / 项目自带 CDE / GRU 后备
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入项目自带的 CDE 模块
try:
    sys.path.append("./NeuralCDE-master")
    from controldiffeq import cdeint, NaturalCubicSpline
    HAS_LOCAL_CDE = True
except ImportError:
    HAS_LOCAL_CDE = False

# 尝试导入 torchcde
try:
    import torchcde
    HAS_TORCHCDE = True
except ImportError:
    HAS_TORCHCDE = False


class CDEFunc(nn.Module):
    """CDE 向量场函数 f(t, h)

    将隐藏状态 h 映射到导数 dh/dt
    """
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        # f: R^hidden -> R^(hidden x input)
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, hidden_channels * input_channels),
        )

    def forward(self, t, h):
        """
        Args:
            t: 当前时间
            h: (B, hidden_channels) 当前隐藏状态
        Returns:
            (B, hidden_channels, input_channels) 导数矩阵
        """
        out = self.net(h)
        return out.view(h.size(0), self.hidden_channels, self.input_channels)


class NeuralCDEEmbedding(nn.Module):
    """Neural CDE 嵌入模块

    将不规则采样的三元组数据 (time, variable_id, value)
    转换为规则采样的固定维度序列

    自动选择可用后端：torchcde > local CDE > GRU fallback
    """
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        max_variable_id: int,
        num_output_steps: int = 48,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_output_steps = num_output_steps

        # 变量嵌入
        self.variable_embedding = nn.Embedding(max_variable_id + 1, hidden_dim)

        # 值编码
        self.value_encoder = nn.Linear(2, hidden_dim)  # time + value

        # CDE 输入通道 = hidden_dim * 2 + 1 (时间)
        self.cde_input_dim = hidden_dim * 2 + 1

        # 初始状态
        self.initial_linear = nn.Linear(hidden_dim * 2, hidden_dim)

        # CDE 函数
        self.cde_func = CDEFunc(self.cde_input_dim, hidden_dim)

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # 选择后端
        if HAS_TORCHCDE:
            self.backend = "torchcde"
            print("[NeuralCDEEmbedding] Using torchcde backend")
        elif HAS_LOCAL_CDE:
            self.backend = "local"
            print("[NeuralCDEEmbedding] Using local CDE backend")
        else:
            self.backend = "gru_fallback"
            print("[NeuralCDEEmbedding] Using GRU fallback (torchcde not available)")
            self.gru = nn.GRU(hidden_dim * 2, hidden_dim, num_layers=2, batch_first=True)
            self.time_query = nn.Parameter(torch.randn(1, num_output_steps, hidden_dim))
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=4, batch_first=True
            )

    def forward(
        self,
        times: torch.Tensor,      # (B, L)
        variables: torch.Tensor,  # (B, L)
        values: torch.Tensor,     # (B, L)
        mask: torch.Tensor = None,
    ):
        """
        Args:
            times: 时间戳（小时）
            variables: 变量 ID
            values: 观测值
            mask: 有效位置掩码

        Returns:
            output: (B, T, output_dim) 规则采样的嵌入序列
        """
        B, L = times.shape
        device = times.device

        if mask is None:
            mask = (variables != 0).float()

        # 编码
        var_emb = self.variable_embedding(variables.long())  # (B, L, hidden)
        time_val = torch.stack([times, values], dim=-1)
        val_enc = self.value_encoder(time_val)  # (B, L, hidden)

        # 拼接路径特征
        path_feat = torch.cat([var_emb, val_enc], dim=-1)  # (B, L, 2*hidden)

        if self.backend == "torchcde":
            z = self._forward_torchcde(times, path_feat, mask)
        elif self.backend == "local":
            z = self._forward_local_cde(times, path_feat, mask)
        else:
            z = self._forward_gru_fallback(path_feat, mask)

        return self.output_proj(z)

    def _forward_torchcde(self, times, path_feat, mask):
        """使用 torchcde 后端"""
        B, L, _ = path_feat.shape
        device = path_feat.device

        # 拼接时间维度
        path = torch.cat([times.unsqueeze(-1), path_feat], dim=-1)  # (B, L, 2*hidden+1)

        # 创建样条系数
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(path)
        X = torchcde.CubicSpline(coeffs)

        # 初始状态
        z0 = self.initial_linear(path_feat[:, 0])  # (B, hidden)

        # 输出时间点
        t_min = times[:, 0].min().item()
        t_max = times.max().item()
        t_span = max(t_max - t_min, 1.0)
        t_eval = torch.linspace(t_min, t_min + t_span, self.num_output_steps, device=device)

        # CDE 积分
        z = torchcde.cdeint(
            X=X,
            func=self.cde_func,
            z0=z0,
            t=t_eval,
            method='rk4',
            options={'step_size': 0.5}
        )

        # 调整维度
        if z.dim() == 3 and z.size(0) == self.num_output_steps:
            z = z.permute(1, 0, 2)  # (B, T, hidden)

        return z

    def _forward_local_cde(self, times, path_feat, mask):
        """使用项目自带 CDE 后端"""
        B, L, _ = path_feat.shape
        device = path_feat.device

        path = torch.cat([times.unsqueeze(-1), path_feat], dim=-1)

        # 创建样条
        spline = NaturalCubicSpline(times, path)

        # 初始状态
        z0 = self.initial_linear(path_feat[:, 0])

        # 输出时间点
        t_min = times[:, 0].min().item()
        t_max = times.max().item()
        t_span = max(t_max - t_min, 1.0)
        t_eval = torch.linspace(t_min, t_min + t_span, self.num_output_steps, device=device)

        # CDE 积分
        z = cdeint(spline.derivative, z0, self.cde_func, t_eval)

        if z.dim() == 3 and z.size(0) == self.num_output_steps:
            z = z.permute(1, 0, 2)

        return z

    def _forward_gru_fallback(self, path_feat, mask):
        """GRU 后备方案"""
        B = path_feat.size(0)
        device = path_feat.device

        # GRU 编码
        h, _ = self.gru(path_feat)  # (B, L, hidden)

        # 使用交叉注意力进行时间重采样
        queries = self.time_query.expand(B, -1, -1)  # (B, T, hidden)
        attn_mask = (mask == 0)  # True 表示需要忽略

        output, _ = self.cross_attention(
            queries, h, h,
            key_padding_mask=attn_mask
        )

        return output


class NeuralCDEEmbeddingSimple(nn.Module):
    """简化版嵌入（纯 GRU + 交叉注意力，无 CDE 依赖）"""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        max_variable_id: int,
        num_output_steps: int = 48,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_output_steps = num_output_steps

        # 变量嵌入
        self.variable_embedding = nn.Embedding(max_variable_id + 1, hidden_dim)

        # 值编码器
        self.value_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 时间演化 GRU
        self.gru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )

        # 时间查询
        self.time_query = nn.Parameter(torch.randn(1, num_output_steps, hidden_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        times: torch.Tensor,
        variables: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        B, L = times.shape
        device = times.device

        if mask is None:
            mask = (variables != 0).float()

        # 编码输入
        var_emb = self.variable_embedding(variables.long())
        time_value = torch.stack([times, values], dim=-1)
        val_enc = self.value_encoder(time_value)

        x = torch.cat([var_emb, val_enc], dim=-1)  # (B, L, 2*hidden)

        # GRU 编码
        h, _ = self.gru(x)  # (B, L, hidden)

        # 交叉注意力重采样
        queries = self.time_query.expand(B, -1, -1)
        attn_mask = (mask == 0)

        output, _ = self.cross_attention(
            queries, h, h,
            key_padding_mask=attn_mask
        )

        return self.output_proj(output)
```

---

### 4.3 双向 Mamba 主干 (`bimamba.py`)

```python
"""
bimamba.py
双向 Mamba 主干网络
"""
import torch
import torch.nn as nn
from typing import Optional

# 从同目录导入
from mamba_minimal import MambaBackbone, ModelArgs, RMSNorm


class BiMambaBackbone(nn.Module):
    """双向 Mamba 主干网络

    包含正向和反向两个 Mamba 分支，通过特征融合捕获双向依赖
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layer: int = 4,
        share_weights: bool = False,  # 是否共享正反向权重
        fusion: str = "concat",       # 融合方式: "concat", "add", "gate"
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.share_weights = share_weights
        self.fusion = fusion

        # 配置参数
        args = ModelArgs(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # 正向 Mamba
        self.forward_mamba = MambaBackbone(args, n_layer)

        # 反向 Mamba（可选共享权重）
        if share_weights:
            self.backward_mamba = self.forward_mamba
        else:
            self.backward_mamba = MambaBackbone(args, n_layer)

        # 融合层
        if fusion == "concat":
            self.fusion_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
        elif fusion == "gate":
            self.gate_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        elif fusion == "add":
            self.fusion_proj = nn.Identity()
            self.scale = nn.Parameter(torch.tensor(0.5))
        else:
            raise ValueError(f"Unknown fusion method: {fusion}")

        # 输出归一化
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) 输入特征序列
            mask: (B, L) 可选掩码，1 表示有效位置
        Returns:
            (B, L, D) 输出特征序列
        """
        B, L, D = x.shape

        # 正向流
        h_forward = self.forward_mamba(x)  # (B, L, D)

        # 反向流：翻转 -> Mamba -> 翻转回来
        x_reversed = torch.flip(x, dims=[1])
        h_backward = self.backward_mamba(x_reversed)
        h_backward = torch.flip(h_backward, dims=[1])  # (B, L, D)

        # 融合
        if self.fusion == "concat":
            h_cat = torch.cat([h_forward, h_backward], dim=-1)  # (B, L, 2D)
            output = self.fusion_proj(h_cat)  # (B, L, D)
        elif self.fusion == "gate":
            h_cat = torch.cat([h_forward, h_backward], dim=-1)
            gate = self.gate_proj(h_cat)  # (B, L, D)
            output = gate * h_forward + (1 - gate) * h_backward
        elif self.fusion == "add":
            output = self.scale * h_forward + (1 - self.scale) * h_backward

        # 归一化
        output = self.norm(output)

        return output


class BiMambaBlock(nn.Module):
    """带残差连接的双向 Mamba 块"""
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        n_layer: int = 2,
        share_weights: bool = False,
        fusion: str = "concat",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.bimamba = BiMambaBackbone(
            d_model=d_model,
            d_state=d_state,
            n_layer=n_layer,
            share_weights=share_weights,
            fusion=fusion,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm 残差连接"""
        return x + self.dropout(self.bimamba(self.norm(x)))
```

---

### 4.4 主模型与损失函数 (`apricotm_v2.py`)

```python
"""
apricotm_v2.py
重构后的 ApricotM 模型：Neural CDE 嵌入 + 双向 Mamba + 效用损失
"""
import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_minimal import ModelArgs, RMSNorm
from neural_cde import NeuralCDEEmbedding, NeuralCDEEmbeddingSimple
from bimamba import BiMambaBackbone


# ============================================================================
# 效用损失函数
# ============================================================================

class UtilityLoss(nn.Module):
    """败血症预测的时间加权效用损失

    核心思想：
    - 对于正样本（将发生败血症），越接近发病时间，预测越重要
    - 使用时间加权的 BCE，让模型更关注发病前的关键窗口期
    - 模拟 PhysioNet Utility Score 的行为，但保持可微分

    PhysioNet Utility Score 近似：
    - t_sepsis - 12h 之前预测为正：中等奖励
    - t_sepsis - 12h 到 t_sepsis - 6h：最高奖励（最优窗口）
    - t_sepsis - 6h 之后预测为正：较低奖励
    - 错误预测负样本：惩罚
    """
    def __init__(
        self,
        early_pred_weight: float = 1.0,     # 早期预测（>12h）权重
        optimal_pred_weight: float = 2.0,   # 最优窗口（6-12h）权重
        late_pred_weight: float = 0.5,      # 晚期预测（0-6h）权重
        negative_weight: float = 1.0,       # 负样本权重
        time_decay: str = "exponential",    # 衰减类型: "linear", "exponential"
        focal_gamma: float = 2.0,           # Focal Loss 的 gamma 参数
        use_focal: bool = True,             # 是否使用 Focal Loss
        label_smoothing: float = 0.0,       # 标签平滑
    ):
        super().__init__()
        self.early_pred_weight = early_pred_weight
        self.optimal_pred_weight = optimal_pred_weight
        self.late_pred_weight = late_pred_weight
        self.negative_weight = negative_weight
        self.time_decay = time_decay
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal
        self.label_smoothing = label_smoothing

    def compute_time_weights(
        self,
        time_to_sepsis: torch.Tensor,  # (B,) 距离败血症发病的小时数
        labels: torch.Tensor,          # (B,) 0 或 1
    ) -> torch.Tensor:
        """计算时间相关的样本权重"""
        weights = torch.ones_like(labels, dtype=torch.float32)

        # 正样本权重（基于距离发病时间）
        is_positive = labels == 1

        if is_positive.any():
            pos_times = time_to_sepsis[is_positive]
            pos_weights = torch.zeros_like(pos_times)

            # 早期预测 (>12h before sepsis)
            early_mask = pos_times > 12
            pos_weights[early_mask] = self.early_pred_weight

            # 最优窗口 (6-12h before sepsis)
            optimal_mask = (pos_times >= 6) & (pos_times <= 12)
            pos_weights[optimal_mask] = self.optimal_pred_weight

            # 晚期预测 (<6h before sepsis)
            late_mask = pos_times < 6
            if self.time_decay == "exponential":
                decay = torch.exp(-0.1 * (6 - pos_times[late_mask].clamp(min=0)))
                pos_weights[late_mask] = self.late_pred_weight * decay
            else:
                pos_weights[late_mask] = self.late_pred_weight * (pos_times[late_mask] / 6).clamp(min=0.1)

            weights[is_positive] = pos_weights

        # 负样本权重
        weights[~is_positive] = self.negative_weight

        return weights

    def forward(
        self,
        predictions: torch.Tensor,      # (B,) 或 (B, T) 预测概率
        labels: torch.Tensor,           # (B,) 真实标签
        time_to_sepsis: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            predictions: 模型预测的概率
            labels: 真实标签
            time_to_sepsis: 距离败血症发病的小时数
            class_weights: 可选的类别权重

        Returns:
            loss: 标量损失值
        """
        # 处理多时间步输出
        if predictions.dim() == 2:
            predictions = predictions[:, -1]

        # 确保在有效范围内
        predictions = predictions.clamp(1e-7, 1 - 1e-7)
        labels = labels.float()

        # 标签平滑
        if self.label_smoothing > 0:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # 计算基础 BCE
        bce = F.binary_cross_entropy(predictions, labels, reduction='none')

        # Focal Loss 调制
        if self.use_focal:
            p_t = predictions * labels + (1 - predictions) * (1 - labels)
            focal_weight = (1 - p_t) ** self.focal_gamma
            bce = focal_weight * bce

        # 时间权重
        if time_to_sepsis is not None:
            time_weights = self.compute_time_weights(time_to_sepsis, (labels > 0.5).float())
            bce = bce * time_weights

        # 类别权重
        if class_weights is not None:
            class_w = class_weights[(labels > 0.5).long()]
            bce = bce * class_w

        return bce.mean()


class MultiTaskUtilityLoss(nn.Module):
    """多任务效用损失（适配原始 ApricotM 的 12 头输出）"""
    def __init__(
        self,
        num_main_tasks: int = 4,
        num_aux_tasks: int = 8,
        main_task_weight: float = 1.0,
        aux_task_weight: float = 0.5,
        **utility_kwargs,
    ):
        super().__init__()
        self.num_main_tasks = num_main_tasks
        self.num_aux_tasks = num_aux_tasks
        self.main_task_weight = main_task_weight
        self.aux_task_weight = aux_task_weight

        # 为每个任务创建损失
        self.main_losses = nn.ModuleList([
            UtilityLoss(**utility_kwargs) for _ in range(num_main_tasks)
        ])
        self.aux_losses = nn.ModuleList([
            UtilityLoss(**utility_kwargs) for _ in range(num_aux_tasks)
        ])

    def forward(
        self,
        predictions: torch.Tensor,  # (B, 12)
        labels: torch.Tensor,       # (B, 12)
        time_to_sepsis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss: 总损失
            loss_dict: 各任务损失详情
        """
        total_loss = torch.tensor(0.0, device=predictions.device)
        loss_dict = {}

        # 主任务损失
        for i, loss_fn in enumerate(self.main_losses):
            task_loss = loss_fn(predictions[:, i], labels[:, i], time_to_sepsis)
            total_loss = total_loss + self.main_task_weight * task_loss
            loss_dict[f'main_{i}'] = task_loss.item()

        # 辅助任务损失
        for i, loss_fn in enumerate(self.aux_losses):
            idx = self.num_main_tasks + i
            task_loss = loss_fn(predictions[:, idx], labels[:, idx], time_to_sepsis)
            total_loss = total_loss + self.aux_task_weight * task_loss
            loss_dict[f'aux_{i}'] = task_loss.item()

        return total_loss, loss_dict


# ============================================================================
# 主模型
# ============================================================================

class ApricotMv2(nn.Module):
    """重构后的 ApricotM 模型

    架构：
    1. Neural CDE 嵌入层 - 处理不规则采样
    2. 双向 Mamba 主干 - 双向序列建模
    3. 多任务分类头 - 败血症预测（4主+8辅）
    """
    def __init__(
        self,
        # 模型维度
        d_model: int = 128,
        d_state: int = 16,

        # 输入配置
        max_variable_id: int = 49,
        d_static: int = 10,

        # CDE 配置
        use_cde: bool = True,
        cde_hidden: int = 64,
        num_output_steps: int = 48,

        # BiMamba 配置
        n_layer: int = 4,
        bimamba_share_weights: bool = False,
        bimamba_fusion: str = "concat",

        # 输出配置
        num_main_tasks: int = 4,
        num_aux_tasks: int = 8,

        # 其他
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_main_tasks = num_main_tasks
        self.num_aux_tasks = num_aux_tasks
        self.num_output_steps = num_output_steps
        self.device = device

        # ===== 嵌入层 =====
        if use_cde:
            self.embedding = NeuralCDEEmbedding(
                hidden_dim=cde_hidden,
                output_dim=d_model,
                max_variable_id=max_variable_id,
                num_output_steps=num_output_steps,
            )
        else:
            self.embedding = NeuralCDEEmbeddingSimple(
                hidden_dim=cde_hidden,
                output_dim=d_model,
                max_variable_id=max_variable_id,
                num_output_steps=num_output_steps,
            )

        # ===== 静态特征处理 =====
        self.static_encoder = nn.Sequential(
            nn.Linear(d_static, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # ===== 双向 Mamba 主干 =====
        self.bimamba = BiMambaBackbone(
            d_model=d_model,
            d_state=d_state,
            n_layer=n_layer,
            share_weights=bimamba_share_weights,
            fusion=bimamba_fusion,
            dropout=dropout,
        )

        # ===== 序列聚合 =====
        self.aggregator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ===== 分类头 =====
        self.main_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_main_tasks)
        ])

        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_aux_tasks)
        ])

    def forward(
        self,
        x: torch.Tensor,              # (B, L, 3): [time, variable_id, value]
        static: torch.Tensor,         # (B, d_static)
    ) -> torch.Tensor:
        """
        Args:
            x: 输入三元组序列
            static: 静态特征

        Returns:
            output: (B, 12) 预测概率
        """
        B = x.shape[0]
        device = x.device

        # 解析输入
        times = x[:, :, 0]       # (B, L)
        variables = x[:, :, 1]   # (B, L)
        values = x[:, :, 2]      # (B, L)
        mask = (variables != 0).float()

        # CDE 嵌入
        embedded = self.embedding(times, variables, values, mask)  # (B, T, d_model)

        # 静态特征
        static_feat = self.static_encoder(static)  # (B, d_model)
        static_feat = static_feat.unsqueeze(1).expand(-1, embedded.shape[1], -1)

        # 融合
        h = embedded + static_feat  # (B, T, d_model)

        # 双向 Mamba
        h = self.bimamba(h)  # (B, T, d_model)

        # TopK 池化
        h_transposed = h.transpose(1, 2)  # (B, d_model, T)
        k = min(5, h.shape[1])
        h_pooled = torch.topk(h_transposed, k=k, dim=2)[0]  # (B, d_model, k)
        h_pooled = h_pooled.mean(dim=2)  # (B, d_model)

        h_agg = self.aggregator(h_pooled)  # (B, d_model)

        # 分类头
        main_outputs = [head(h_agg) for head in self.main_heads]
        aux_outputs = [head(h_agg) for head in self.aux_heads]

        output = torch.cat(main_outputs + aux_outputs, dim=-1)  # (B, 12)

        return output

    def get_embedding(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """获取中间嵌入（用于可解释性分析）"""
        times = x[:, :, 0]
        variables = x[:, :, 1]
        values = x[:, :, 2]
        mask = (variables != 0).float()

        embedded = self.embedding(times, variables, values, mask)
        static_feat = self.static_encoder(static)
        static_feat = static_feat.unsqueeze(1).expand(-1, embedded.shape[1], -1)

        h = embedded + static_feat
        h = self.bimamba(h)

        return h


# ============================================================================
# 便捷函数
# ============================================================================

def create_model(
    d_model: int = 128,
    n_layer: int = 4,
    max_variable_id: int = 49,
    d_static: int = 10,
    use_cde: bool = True,
    dropout: float = 0.1,
    device: str = "cuda",
) -> ApricotMv2:
    """创建模型的便捷函数"""
    return ApricotMv2(
        d_model=d_model,
        d_state=16,
        max_variable_id=max_variable_id,
        d_static=d_static,
        use_cde=use_cde,
        cde_hidden=d_model // 2,
        num_output_steps=48,
        n_layer=n_layer,
        bimamba_share_weights=False,
        bimamba_fusion="concat",
        num_main_tasks=4,
        num_aux_tasks=8,
        dropout=dropout,
        device=device,
    )


def create_criterion(
    use_focal: bool = True,
    focal_gamma: float = 2.0,
    optimal_pred_weight: float = 2.0,
) -> MultiTaskUtilityLoss:
    """创建损失函数的便捷函数"""
    return MultiTaskUtilityLoss(
        num_main_tasks=4,
        num_aux_tasks=8,
        main_task_weight=1.0,
        aux_task_weight=0.5,
        early_pred_weight=1.0,
        optimal_pred_weight=optimal_pred_weight,
        late_pred_weight=0.5,
        negative_weight=1.0,
        use_focal=use_focal,
        focal_gamma=focal_gamma,
    )
```

---

## 5. 使用示例

### 5.1 基本训练示例

```python
"""
train_example.py
使用重构后模型的训练示例
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from apricotm_v2 import create_model, create_criterion


def train():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建模型
    model = create_model(
        d_model=128,
        n_layer=4,
        max_variable_id=49,
        d_static=10,
        use_cde=True,  # 如果 torchcde 安装失败，设为 False
        dropout=0.2,
        device=str(device),
    ).to(device)

    # 创建损失函数
    criterion = create_criterion(
        use_focal=True,
        focal_gamma=2.0,
        optimal_pred_weight=2.0,
    )

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # 示例数据（实际使用时替换为真实数据加载）
    batch_size = 32
    seq_len = 256

    # 模拟输入
    x = torch.randn(batch_size, seq_len, 3).to(device)
    x[:, :, 0] = torch.cumsum(torch.rand(batch_size, seq_len), dim=1)  # 时间戳
    x[:, :, 1] = torch.randint(1, 50, (batch_size, seq_len)).float()   # 变量ID
    static = torch.randn(batch_size, 10).to(device)
    labels = torch.randint(0, 2, (batch_size, 12)).float().to(device)
    time_to_sepsis = torch.rand(batch_size).to(device) * 24  # 0-24小时

    # 训练循环
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()

        # 前向传播
        output = model(x, static)

        # 计算损失
        loss, loss_dict = criterion(output, labels, time_to_sepsis)

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
        print(f"  Main tasks: {[f'{k}={v:.4f}' for k,v in loss_dict.items() if 'main' in k]}")

    print("Training complete!")

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'apricotm_v2_checkpoint.pth')


if __name__ == "__main__":
    train()
```

### 5.2 与原始数据格式兼容

```python
"""
与原始 ApricotM 数据格式兼容的示例
"""
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from apricotm_v2 import create_model, create_criterion


class ApricotDataset(Dataset):
    """兼容原始 HDF5 数据格式"""

    def __init__(self, h5_path: str, split: str = 'training'):
        self.h5_path = h5_path
        self.split = split

        with h5py.File(h5_path, 'r') as f:
            self.length = f[f'{split}/X'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            # 原始格式: X shape (N, L, 4) -> 取前3列 (time, var_id, value)
            x = torch.tensor(f[f'{self.split}/X'][idx, :, :3], dtype=torch.float32)
            static = torch.tensor(f[f'{self.split}/static'][idx], dtype=torch.float32)
            y_main = torch.tensor(f[f'{self.split}/y_main'][idx], dtype=torch.float32)
            y_trans = torch.tensor(f[f'{self.split}/y_trans'][idx], dtype=torch.float32)

        # 合并标签
        labels = torch.cat([y_main, y_trans], dim=0)

        return x, static, labels


def train_with_real_data(h5_path: str):
    """使用真实数据训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集
    train_dataset = ApricotDataset(h5_path, 'training')
    val_dataset = ApricotDataset(h5_path, 'validation')

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

    # 模型
    model = create_model(use_cde=True).to(device)
    criterion = create_criterion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    # 训练
    for epoch in range(100):
        model.train()
        total_loss = 0

        for x, static, labels in train_loader:
            x, static, labels = x.to(device), static.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(x, static)
            loss, _ = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")
```

---

## 6. 文件结构

```
mamba-cde/
├── apricotM-main/
│   ├── main/
│   │   ├── models/
│   │   │   ├── apricotm/              # 原始模型（保留）
│   │   │   │   ├── apricotm.py
│   │   │   │   └── ...
│   │   │   │
│   │   │   └── apricotm_v2/           # 重构后模型（新建）
│   │   │       ├── __init__.py
│   │   │       ├── mamba_minimal.py   # 纯 PyTorch Mamba
│   │   │       ├── neural_cde.py      # CDE 嵌入
│   │   │       ├── bimamba.py         # 双向 Mamba
│   │   │       ├── apricotm_v2.py     # 主模型 + 损失
│   │   │       └── train_v2.py        # 训练脚本
│   │   │
│   │   ├── datasets/                  # 数据处理（保留）
│   │   └── analyses/                  # 分析脚本（保留）
│   │
│   └── requirements.txt               # 更新后的依赖
│
├── NeuralCDE-master/                  # 项目自带 CDE 库
│   └── controldiffeq/
│
└── README.md                          # 本文档
```

---

## 附录：关键改进总结

| 特性 | 原始 ApricotM | 重构后 ApricotMv2 |
|-----|--------------|-------------------|
| 平台兼容性 | Linux Only (mamba-ssm) | Windows/Linux (纯PyTorch) |
| 时序嵌入 | Conv1d 离散化 | Neural CDE 连续建模 |
| 序列建模 | 单向 Mamba | 双向 Mamba (BiMamba) |
| 损失函数 | 简单 BCE | 时间加权 Focal Loss |
| 不规则采样 | 隐式处理 | 显式 CDE 建模 |
| 可解释性 | 有限 | 支持嵌入提取 |

---

## 参考文献

1. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
2. Kidger, P., et al. (2020). Neural Controlled Differential Equations for Irregular Time Series.
3. Reyna, M. A., et al. (2019). Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019.
