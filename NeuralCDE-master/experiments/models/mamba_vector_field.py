"""
Mamba-NCDE集成：用Mamba替代失效的Spectral分支

核心思想：
- Time Branch：局部时间动态（MLP + FiLM）
- Mamba Branch：全局序列建模（状态空间模型）
- Learned Fusion：动态融合两个分支

作者：根据ApricotM和NeuralCDE设计
日期：2026-02-10
"""

import math
import torch
import torch.nn as nn
from functools import partial

# 导入Mamba核心模块
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("警告: mamba-ssm未安装，请运行: pip install mamba-ssm")
    MAMBA_AVAILABLE = False
    Mamba = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm = None
    layer_norm_fn = None
    rms_norm_fn = None

from .vector_fields import TimeEncoder


class Block(nn.Module):
    """
    简单的Block包装器，用于兼容mamba-ssm 2.x API
    包含：Norm -> Mixer -> Residual
    """
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

    def forward(self, hidden_states, residual=None, inference_params=None):
        """
        前向传播：norm -> mixer -> residual
        """
        if residual is None:
            residual = hidden_states
        else:
            hidden_states = hidden_states + residual

        hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)

        return hidden_states, residual


def create_mamba_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    """
    创建单个Mamba Block

    Args:
        d_model: 模型维度
        ssm_cfg: SSM配置字典
        norm_epsilon: LayerNorm的epsilon
        rms_norm: 是否使用RMSNorm
        residual_in_fp32: 残差连接是否使用FP32
        fused_add_norm: 是否使用融合的add+norm
        layer_idx: 层索引
        device: 设备
        dtype: 数据类型

    Returns:
        Block: Mamba Block实例
    """
    if not MAMBA_AVAILABLE:
        raise ImportError("mamba-ssm未安装，无法创建Mamba Block")

    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}

    # Mamba mixer
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)

    # Normalization
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm,
        eps=norm_epsilon,
        **factory_kwargs
    )

    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class MambaBranch(nn.Module):
    """
    Mamba分支：用状态空间模型捕捉隐状态的长程时序依赖

    工作流程：
    1. 投影：hidden_channels -> d_model
    2. Mamba处理：通过n_layer个Mamba Block
    3. 投影回：d_model -> hidden_channels

    注意：
    - Mamba需要序列维度，所以会自动添加/移除seq维度
    - 支持可变长度序列（通过mask）
    """
    def __init__(self, hidden_channels, d_model=64, n_layer=2,
                 ssm_cfg=None, dropout=0.1):
        """
        Args:
            hidden_channels: NCDE隐状态维度
            d_model: Mamba内部维度
            n_layer: Mamba Block层数
            ssm_cfg: SSM配置（d_state, d_conv等）
            dropout: Dropout比率
        """
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("MambaBranch需要安装mamba-ssm")

        self.hidden_channels = hidden_channels
        self.d_model = d_model
        self.n_layer = n_layer

        # 默认SSM配置
        if ssm_cfg is None:
            ssm_cfg = {
                'd_state': 16,    # SSM状态维度
                'd_conv': 4,      # 卷积核大小
                'expand': 2,      # 扩展因子
            }

        # 投影到Mamba维度
        self.proj_in = nn.Linear(hidden_channels, d_model)

        # Mamba Blocks
        self.mamba_blocks = nn.ModuleList([
            create_mamba_block(
                d_model,
                ssm_cfg=ssm_cfg,
                layer_idx=i,
            )
            for i in range(n_layer)
        ])

        # LayerNorm
        self.norm_f = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 投影回NCDE维度
        self.proj_out = nn.Linear(d_model, hidden_channels)

    def forward(self, z, t=None):
        """
        Args:
            z: [..., hidden_channels] - NCDE隐状态
            t: 时间（当前未使用，但保留接口一致性）

        Returns:
            [..., hidden_channels] - Mamba处理后的全局特征
        """
        original_shape = z.shape

        # 投影到Mamba维度
        h = self.proj_in(z)  # [..., d_model]

        # Mamba需要[batch, seq_len, d_model]格式
        # 如果z是[batch, hidden_channels]，添加seq维度
        if h.dim() == 2:
            h = h.unsqueeze(1)  # [batch, 1, d_model]
            squeeze_seq = True
        else:
            squeeze_seq = False

        # 通过Mamba Blocks（带残差连接）
        residual = None
        for block in self.mamba_blocks:
            h, residual = block(h, residual)

        # 最后的LayerNorm
        if residual is not None:
            h = (h + residual).to(dtype=self.norm_f.weight.dtype)
        h = self.norm_f(h)

        # 移除序列维度（如果之前添加的）
        if squeeze_seq:
            h = h.squeeze(1)  # [batch, d_model]

        # Dropout
        h = self.dropout(h)

        # 投影回原维度
        return self.proj_out(h)  # [..., hidden_channels]


class MambaModulatedVectorField(nn.Module):
    """
    Mamba调制的Vector Field：双分支架构

    架构：
    ┌─────────────────────────────────────┐
    │   Time Branch (Local)               │
    │   MLP + FiLM(t)                     │
    │   ↓                                 │
    │   out_time (tanh bounded)           │
    └─────────────────────────────────────┘
                    ↓
                 Fusion ← alpha(t, z)
                    ↓
    ┌─────────────────────────────────────┐
    │   Mamba Branch (Global)             │
    │   Mamba(z) + Projection             │
    │   ↓                                 │
    │   out_mamba (tanh bounded)          │
    └─────────────────────────────────────┘

    输出：alpha * out_time + (1-alpha) * out_mamba
    """
    def __init__(self, input_channels, hidden_channels,
                 time_dim=32, hidden_hidden_channels=49,
                 num_hidden_layers=4, mamba_d_model=64,
                 mamba_n_layer=2, fusion_mode='learned',
                 mamba_dropout=0.1):
        """
        Args:
            input_channels: 控制信号X的通道数
            hidden_channels: 隐状态z的维度
            time_dim: 时间编码维度
            hidden_hidden_channels: Time Branch的隐层维度
            num_hidden_layers: Time Branch的层数
            mamba_d_model: Mamba内部维度
            mamba_n_layer: Mamba Block层数
            fusion_mode: 融合模式 ('learned', 'fixed', 'add')
            mamba_dropout: Mamba dropout率
        """
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.time_dim = time_dim
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.fusion_mode = fusion_mode

        # ========== Time Encoder ==========
        self.time_encoder = TimeEncoder(time_dim)

        # ========== Time Branch (Local) ==========
        # FiLM参数生成器
        self.film_generator = nn.Linear(time_dim, 2 * hidden_hidden_channels)

        # Multi-layer MLP
        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linears = nn.ModuleList([
            nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
            for _ in range(num_hidden_layers - 1)
        ])
        self.linear_out = nn.Linear(hidden_hidden_channels,
                                     input_channels * hidden_channels)

        # ========== Mamba Branch (Global) ==========
        self.mamba_branch = MambaBranch(
            hidden_channels=hidden_channels,
            d_model=mamba_d_model,
            n_layer=mamba_n_layer,
            dropout=mamba_dropout,
        )
        self.mamba_out = nn.Linear(hidden_channels,
                                     input_channels * hidden_channels)

        # ========== Fusion Gate ==========
        if fusion_mode == 'learned':
            # 学习动态融合权重 alpha = f(t, z)
            self.fusion_gate = nn.Sequential(
                nn.Linear(time_dim + hidden_channels, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # 输出[0, 1]
            )
        elif fusion_mode == 'fixed':
            # 固定权重（可通过超参数调整）
            self.register_buffer('alpha_fixed', torch.tensor(0.7))
        # 'add'模式不需要额外参数

        # ========== 初始化 ==========
        self._init_weights()

        # ========== 日志记录（用于可视化） ==========
        self.logging_enabled = False
        self.logs = []

    def _init_weights(self):
        """初始化策略：
        1. FiLM初始化为恒等变换（gamma=0, beta=0）
        2. Fusion gate初始化为0.5（平衡两分支）
        3. 输出层使用较小的初始化（稳定训练）
        """
        # FiLM: 恒等变换
        nn.init.zeros_(self.film_generator.weight)
        nn.init.zeros_(self.film_generator.bias)

        # Fusion gate: 初始化为0.5
        if self.fusion_mode == 'learned':
            with torch.no_grad():
                # 让最后一层bias接近0，使sigmoid(0)=0.5
                self.fusion_gate[-2].bias.fill_(0.0)

        # 输出层：小初始化
        nn.init.xavier_uniform_(self.linear_out.weight, gain=0.1)
        nn.init.xavier_uniform_(self.mamba_out.weight, gain=0.1)

    def set_logging(self, enabled):
        """启用或禁用日志记录模式，用于内部动态可视化

        Args:
            enabled: 布尔值，开启/关闭日志记录
        """
        self.logging_enabled = enabled
        if enabled:
            self.logs = []  # 启用时清空日志

    def clear_logs(self):
        """清空所有存储的日志"""
        self.logs = []

    def extract_logs(self):
        """提取日志数据为NumPy数组用于分析

        Returns:
            字典包含：
                - 'time': 时间点数组，shape (num_steps,)
                - 'gamma': FiLM gamma参数，shape (num_steps, hidden_hidden_channels)
                - 'beta': FiLM beta参数，shape (num_steps, hidden_hidden_channels)
                - 'time_branch_norm': 时间分支的L2范数，shape (num_steps,)
                - 'mamba_branch_norm': Mamba分支的L2范数，shape (num_steps,)
                - 'fusion_alpha': 融合权重（learned模式），shape (num_steps,) 或 None
                - 'contribution_ratio': mamba/time分支比值，shape (num_steps,)
        """
        import numpy as np

        if not self.logs:
            return {
                'time': np.array([]),
                'gamma': np.array([]),
                'beta': np.array([]),
                'time_branch_norm': np.array([]),
                'mamba_branch_norm': np.array([]),
                'fusion_alpha': np.array([]),
                'contribution_ratio': np.array([])
            }

        # 提取并堆叠所有日志条目
        times = np.array([log['time'] for log in self.logs])
        gammas = np.stack([log['gamma'] for log in self.logs], axis=0)
        betas = np.stack([log['beta'] for log in self.logs], axis=0)
        time_norms = np.array([log['time_branch_norm'] for log in self.logs])
        mamba_norms = np.array([log['mamba_branch_norm'] for log in self.logs])

        # 融合权重（可能为None）
        if self.logs[0]['fusion_alpha'] is not None:
            alphas = np.array([log['fusion_alpha'] for log in self.logs])
        else:
            alphas = None

        # 计算贡献率（避免除零）
        contribution_ratio = mamba_norms / (time_norms + 1e-8)

        return {
            'time': times,
            'gamma': gammas,
            'beta': betas,
            'time_branch_norm': time_norms,
            'mamba_branch_norm': mamba_norms,
            'fusion_alpha': alphas,
            'contribution_ratio': contribution_ratio
        }

    def extra_repr(self):
        return (f"input_channels: {self.input_channels}, "
                f"hidden_channels: {self.hidden_channels}, "
                f"time_dim: {self.time_dim}, "
                f"hidden_hidden_channels: {self.hidden_hidden_channels}, "
                f"num_hidden_layers: {self.num_hidden_layers}, "
                f"fusion_mode: {self.fusion_mode}")

    def forward(self, t, z):
        """
        Args:
            t: 当前时间（scalar或[batch]）
            z: 隐状态 [..., hidden_channels]

        Returns:
            vector_field: [..., hidden_channels, input_channels]
        """
        batch_shape = z.shape[:-1]

        # ========== Time Encoding ==========
        # 处理t的维度，使其与z的batch维度匹配
        if t.dim() == 0:
            # t是标量，扩展到batch大小
            batch_size = z.shape[0] if z.dim() > 1 else 1
            t_expanded = t.expand(batch_size)
        else:
            t_expanded = t

        time_enc = self.time_encoder(t_expanded)  # [batch, time_dim]

        # ========== Time Branch (Local) ==========
        # FiLM参数
        film_params = self.film_generator(time_enc)
        gamma = film_params[..., :self.hidden_hidden_channels]
        beta = film_params[..., self.hidden_hidden_channels:]

        # 处理batch维度对齐
        if z.dim() > 2:
            extra_dims = z.dim() - 2
            for _ in range(extra_dims):
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)

        # MLP with FiLM modulation
        h_time = self.linear_in(z)
        h_time = (1 + gamma) * h_time + beta
        h_time = torch.relu(h_time)

        for linear in self.linears:
            h_time = torch.relu(linear(h_time))

        out_time = self.linear_out(h_time)
        out_time = torch.tanh(out_time)  # 关键：输出bounding到[-1, 1]

        # ========== Mamba Branch (Global) ==========
        h_mamba = self.mamba_branch(z, t)  # [..., hidden_channels]
        out_mamba = self.mamba_out(h_mamba)
        out_mamba = torch.tanh(out_mamba)  # 关键：输出bounding

        # ========== Fusion ==========
        if self.fusion_mode == 'learned':
            # 动态融合：alpha基于当前时间和状态
            # 展平z用于gate输入
            z_flat = z.reshape(-1, self.hidden_channels) if z.dim() > 2 else z
            time_enc_flat = time_enc.reshape(-1, self.time_dim) if time_enc.dim() > 2 else time_enc

            gate_input = torch.cat([time_enc_flat, z_flat], dim=-1)
            alpha = self.fusion_gate(gate_input)  # [..., 1]

            # 恢复原始shape
            if z.dim() > 2:
                alpha = alpha.reshape(*batch_shape, 1)

            out = alpha * out_time + (1 - alpha) * out_mamba

        elif self.fusion_mode == 'fixed':
            # 固定权重融合
            alpha = self.alpha_fixed
            out = alpha * out_time + (1 - alpha) * out_mamba

        else:  # 'add'
            # 简单相加（可能需要额外的归一化）
            out = out_time + out_mamba
            out = torch.tanh(out)  # 再次bound
            alpha = None

        # ========== 日志记录（如果启用） ==========
        if self.logging_enabled:
            # 记录时间
            if t.dim() == 0:
                time_log = t.item()
            else:
                time_log = t[0].item() if t.shape[0] > 0 else 0.0

            # 记录FiLM参数（取batch平均）
            gamma_log = gamma.mean(dim=0).detach().cpu().numpy()
            beta_log = beta.mean(dim=0).detach().cpu().numpy()

            # 记录分支范数
            time_branch_norm = torch.norm(out_time, p=2, dim=-1).mean().item()
            mamba_branch_norm = torch.norm(out_mamba, p=2, dim=-1).mean().item()

            # 记录融合权重
            if self.fusion_mode == 'learned':
                alpha_log = alpha.mean().item()
            elif self.fusion_mode == 'fixed':
                alpha_log = alpha.item()
            else:
                alpha_log = None

            self.logs.append({
                'time': time_log,
                'gamma': gamma_log,
                'beta': beta_log,
                'time_branch_norm': time_branch_norm,
                'mamba_branch_norm': mamba_branch_norm,
                'fusion_alpha': alpha_log
            })

        # ========== Reshape to Vector Field ==========
        # out: [..., input_channels * hidden_channels]
        # -> [..., hidden_channels, input_channels]
        out = out.view(*batch_shape, self.hidden_channels, self.input_channels)

        return out

    def get_branch_contributions(self, t, z):
        """
        分析工具：返回两个分支的独立输出，用于可视化

        Returns:
            dict: {
                'time_branch': Time分支输出,
                'mamba_branch': Mamba分支输出,
                'fusion_alpha': 融合权重（如果是learned模式）,
                'final_output': 最终融合输出
            }
        """
        with torch.no_grad():
            batch_shape = z.shape[:-1]

            # Time encoding
            if t.dim() == 0:
                batch_size = z.shape[0] if z.dim() > 1 else 1
                t_expanded = t.expand(batch_size)
            else:
                t_expanded = t
            time_enc = self.time_encoder(t_expanded)

            # Time Branch
            film_params = self.film_generator(time_enc)
            gamma = film_params[..., :self.hidden_hidden_channels]
            beta = film_params[..., self.hidden_hidden_channels:]

            h_time = self.linear_in(z)
            h_time = (1 + gamma) * h_time + beta
            h_time = torch.relu(h_time)
            for linear in self.linears:
                h_time = torch.relu(linear(h_time))
            out_time = torch.tanh(self.linear_out(h_time))

            # Mamba Branch
            h_mamba = self.mamba_branch(z, t)
            out_mamba = torch.tanh(self.mamba_out(h_mamba))

            # Fusion alpha
            alpha = None
            if self.fusion_mode == 'learned':
                gate_input = torch.cat([time_enc, z], dim=-1)
                alpha = self.fusion_gate(gate_input)
            elif self.fusion_mode == 'fixed':
                alpha = self.alpha_fixed

            # Final output
            final = self(t, z)

            return {
                'time_branch': out_time.view(*batch_shape, self.hidden_channels, self.input_channels),
                'mamba_branch': out_mamba.view(*batch_shape, self.hidden_channels, self.input_channels),
                'fusion_alpha': alpha,
                'final_output': final,
            }


# ========== 工厂函数 ==========
def create_mamba_modulated_vector_field(input_channels, hidden_channels, **kwargs):
    """
    创建MambaModulatedVectorField的工厂函数

    使用示例：
    >>> vf = create_mamba_modulated_vector_field(
    ...     input_channels=69,
    ...     hidden_channels=64,
    ...     time_dim=32,
    ...     mamba_d_model=64,
    ...     mamba_n_layer=2,
    ...     fusion_mode='learned'
    ... )
    """
    if not MAMBA_AVAILABLE:
        raise ImportError(
            "MambaModulatedVectorField需要安装mamba-ssm。\n"
            "请运行: pip install mamba-ssm\n"
            "或从源码安装: pip install causal-conv1d>=1.2.0 && pip install mamba-ssm --no-cache-dir"
        )

    return MambaModulatedVectorField(input_channels, hidden_channels, **kwargs)
