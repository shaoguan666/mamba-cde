"""
CCF-B论文图表生成脚本
用于生成高质量的论文级别可视化图表

用法:
    # 方法1: 从训练好的模型生成图表
    python paper_plots.py --model-path <path> --output-dir <dir>

    # 方法2: 从已保存的日志数据生成图表
    python paper_plots.py --logs-path <path> --output-dir <dir>

    # 方法3: 在代码中直接调用
    import paper_plots
    paper_plots.plot_mechanism_dynamics(logs, save_dir='./figures')
"""

import sys
import os
import argparse
import numpy as np

# 设置matplotlib后端为非交互式(避免Windows下的显示问题)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import torch

# 设置论文级别的绘图风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Seaborn配置
sns.set_palette("husl")


def plot_mechanism_dynamics(logs, save_dir='./figures'):
    """
    生成3个论文级别的图表,用于展示SpectralModulatedVectorField的内部机制

    参数:
        logs (dict): 日志数据字典,包含以下键:
            - 't' 或 'time': 时间步数组
            - 'gamma': FiLM gamma参数 (shape: [T, D])
            - 'beta': FiLM beta参数 (shape: [T, D])
            - 'spectral_weights': 频谱权重 (shape: [T, F])
            - 'time_branch_norm': 时间分支的L2范数 (shape: [T])
            - 'spectral_branch_norm': 频谱分支的L2范数 (shape: [T])
            - 'contribution_ratio': 贡献率比值 (shape: [T])

    save_dir (str): 图表保存目录
    """

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 兼容两种时间键名
    time = logs.get('t', logs.get('time', None))
    if time is None:
        raise ValueError("日志中必须包含 't' 或 'time' 键")

    # 如果是torch.Tensor,转换为numpy
    time = _to_numpy(time)
    gamma = _to_numpy(logs['gamma'])
    beta = _to_numpy(logs['beta'])
    spectral_weights = _to_numpy(logs['spectral_weights'])
    time_branch_norm = _to_numpy(logs['time_branch_norm'])
    spectral_branch_norm = _to_numpy(logs['spectral_branch_norm'])
    contribution_ratio = _to_numpy(logs['contribution_ratio'])

    print("\n" + "=" * 70)
    print("生成论文图表")
    print("=" * 70)

    # Fig 1: Spectral Gating Mechanism (频率域)
    print("\n[1/3] 生成 Fig 1: Spectral Gating Mechanism...")
    fig1_path = os.path.join(save_dir, 'fig1_spectral_gating.pdf')
    _plot_spectral_gating(spectral_weights, time, fig1_path)
    print(f"      已保存: {fig1_path}")

    # Fig 2: FiLM Temporal Adaptation (时域,双轴)
    print("[2/3] 生成 Fig 2: FiLM Temporal Adaptation...")
    fig2_path = os.path.join(save_dir, 'fig2_film_adaptation.pdf')
    _plot_film_adaptation(time, gamma, beta, contribution_ratio, fig2_path)
    print(f"      已保存: {fig2_path}")

    # Fig 3: Branch Contribution Analysis (堆叠面积图)
    print("[3/3] 生成 Fig 3: Branch Contribution Analysis...")
    fig3_path = os.path.join(save_dir, 'fig3_branch_contribution.pdf')
    _plot_branch_contribution(time, time_branch_norm, spectral_branch_norm, fig3_path)
    print(f"      已保存: {fig3_path}")

    print("\n" + "=" * 70)
    print("所有图表生成完成!")
    print(f"保存位置: {os.path.abspath(save_dir)}")
    print("=" * 70 + "\n")


def plot_mamba_dynamics(logs, save_dir='./figures'):
    """
    生成3个论文级别的图表，用于展示MambaModulatedVectorField的内部机制

    参数:
        logs (dict): 日志数据字典，包含以下键:
            - 't' 或 'time': 时间步数组
            - 'gamma': FiLM gamma参数 (shape: [T, D])
            - 'beta': FiLM beta参数 (shape: [T, D])
            - 'time_branch_norm': 时间分支的L2范数 (shape: [T])
            - 'mamba_branch_norm': Mamba分支的L2范数 (shape: [T])
            - 'fusion_alpha': 融合权重 (shape: [T]) 或 None
            - 'contribution_ratio': 贡献率比值 (shape: [T])

    save_dir (str): 图表保存目录
    """

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 兼容两种时间键名
    time = logs.get('t', logs.get('time', None))
    if time is None:
        raise ValueError("日志中必须包含 't' 或 'time' 键")

    # 如果是torch.Tensor，转换为numpy
    time = _to_numpy(time)
    gamma = _to_numpy(logs['gamma'])
    beta = _to_numpy(logs['beta'])
    time_branch_norm = _to_numpy(logs['time_branch_norm'])
    mamba_branch_norm = _to_numpy(logs['mamba_branch_norm'])
    contribution_ratio = _to_numpy(logs['contribution_ratio'])

    # 融合权重（可能为None）
    fusion_alpha = logs.get('fusion_alpha', None)
    if fusion_alpha is not None:
        fusion_alpha = _to_numpy(fusion_alpha)

    print("\n" + "=" * 70)
    print("生成Mamba-NCDE论文图表")
    print("=" * 70)

    # Fig 1: Fusion Gate Dynamics (如果有learned fusion)
    if fusion_alpha is not None:
        print("\n[1/3] 生成 Fig 1: Fusion Gate Dynamics...")
        fig1_path = os.path.join(save_dir, 'fig1_mamba_fusion_gate.pdf')
        _plot_mamba_fusion_gate(time, fusion_alpha, fig1_path)
        print(f"      已保存: {fig1_path}")
    else:
        print("\n[1/3] 跳过 Fig 1: Fusion Gate (固定权重或相加模式)")

    # Fig 2: FiLM Temporal Adaptation (双轴)
    print("[2/3] 生成 Fig 2: FiLM Temporal Adaptation...")
    fig2_path = os.path.join(save_dir, 'fig2_mamba_film_adaptation.pdf')
    _plot_film_adaptation(time, gamma, beta, contribution_ratio, fig2_path)
    print(f"      已保存: {fig2_path}")

    # Fig 3: Branch Contribution Analysis (Time vs Mamba)
    print("[3/3] 生成 Fig 3: Branch Contribution Analysis...")
    fig3_path = os.path.join(save_dir, 'fig3_mamba_branch_contribution.pdf')
    _plot_mamba_branch_contribution(time, time_branch_norm, mamba_branch_norm, fig3_path)
    print(f"      已保存: {fig3_path}")

    print("\n" + "=" * 70)
    print("所有图表生成完成！")
    print(f"保存位置: {os.path.abspath(save_dir)}")
    print("=" * 70 + "\n")


def _to_numpy(tensor):
    """将torch.Tensor转换为numpy数组"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def _plot_spectral_gating(spectral_weights, time, save_path):
    """
    Fig 1: 频谱门控机制 (频率域)
    展示spectral_weights的幅度,并标注低通滤波行为
    """
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # 使用最后一个时间步或平均权重
    if len(spectral_weights.shape) == 2:
        # 如果是2D (time, freq),取时间维度的平均
        weights = spectral_weights.mean(axis=0)
    else:
        weights = spectral_weights

    freq_indices = np.arange(len(weights))

    # 绘制频谱权重
    ax.plot(freq_indices, weights, linewidth=2.5, color='#2E86AB', marker='o',
            markersize=6, markerfacecolor='white', markeredgewidth=1.5)

    # 检测低通滤波行为: 如果前半部分频率的平均权重 > 后半部分
    mid_point = len(weights) // 2
    low_freq_avg = weights[:mid_point].mean()
    high_freq_avg = weights[mid_point:].mean()

    if low_freq_avg > high_freq_avg * 1.2:  # 20%阈值
        # 添加低通滤波注释
        max_weight_idx = np.argmax(weights)
        ax.annotate('Low-pass filtering behavior',
                   xy=(max_weight_idx, weights[max_weight_idx]),
                   xytext=(max_weight_idx + len(weights)//4, weights[max_weight_idx] + 0.1),
                   fontsize=10,
                   arrowprops=dict(arrowstyle='->', color='#C73E1D', lw=1.5),
                   color='#C73E1D',
                   fontweight='bold')

    ax.set_xlabel('Frequency Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight Magnitude', fontsize=12, fontweight='bold')
    ax.set_title('Spectral Gating Mechanism', fontsize=13, fontweight='bold', pad=15)

    # 移除网格线
    ax.grid(False)

    # 设置刻度
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_film_adaptation(time, gamma, beta, contribution_ratio, save_path):
    """
    Fig 2: FiLM时间自适应 (双轴图)
    左轴: gamma/beta随时间变化
    右轴: contribution_ratio随时间变化
    """
    fig, ax1 = plt.subplots(figsize=(8, 4), dpi=300)

    # 计算gamma和beta的均值(如果是多维的)
    if len(gamma.shape) > 1:
        gamma_mean = gamma.mean(axis=1)
        gamma_std = gamma.std(axis=1)
    else:
        gamma_mean = gamma
        gamma_std = None

    if len(beta.shape) > 1:
        beta_mean = beta.mean(axis=1)
        beta_std = beta.std(axis=1)
    else:
        beta_mean = beta
        beta_std = None

    # 左轴: FiLM参数
    color_gamma = '#1E88E5'
    color_beta = '#FFC107'

    line1 = ax1.plot(time, gamma_mean, color=color_gamma, linewidth=2.5,
                     label=r'$\gamma$ (scale)', marker='s', markersize=4,
                     markevery=max(1, len(time)//10))

    if gamma_std is not None:
        ax1.fill_between(time, gamma_mean - gamma_std, gamma_mean + gamma_std,
                        color=color_gamma, alpha=0.15)

    line2 = ax1.plot(time, beta_mean, color=color_beta, linewidth=2.5,
                     label=r'$\beta$ (shift)', marker='^', markersize=4,
                     markevery=max(1, len(time)//10))

    if beta_std is not None:
        ax1.fill_between(time, beta_mean - beta_std, beta_mean + beta_std,
                        color=color_beta, alpha=0.15)

    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('FiLM Parameters', fontsize=12, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=10, width=1.2)
    ax1.tick_params(axis='x', labelsize=10, width=1.2)

    # 右轴: 贡献率
    ax2 = ax1.twinx()
    color_ratio = '#43A047'

    line3 = ax2.plot(time, contribution_ratio, color=color_ratio, linewidth=2.5,
                     linestyle='--', label='Contribution Ratio', marker='o',
                     markersize=4, markevery=max(1, len(time)//10))

    # 添加参考线 (ratio=1)
    ax2.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)

    ax2.set_ylabel('Spectral/Time Ratio', fontsize=12, fontweight='bold', color=color_ratio)
    ax2.tick_params(axis='y', labelcolor=color_ratio, labelsize=10, width=1.2)

    # 移除网格线
    ax1.grid(False)
    ax2.grid(False)

    # 合并图例
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=9, framealpha=0.9)

    ax1.set_title('FiLM Temporal Adaptation', fontsize=13, fontweight='bold', pad=15)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_branch_contribution(time, time_branch_norm, spectral_branch_norm, save_path):
    """
    Fig 3: 分支贡献分析 (堆叠面积图)
    X轴: 时间
    Y轴: L2 Norm
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    # 颜色方案
    color_time = '#4A90E2'      # 蓝色 - 时间分支(局部)
    color_spectral = '#E94B3C'  # 红色 - 频谱分支(全局)

    # 绘制堆叠面积图
    ax.fill_between(time, 0, time_branch_norm,
                    color=color_time, alpha=0.7, label='Time Branch (Local)')

    ax.fill_between(time, time_branch_norm,
                    time_branch_norm + spectral_branch_norm,
                    color=color_spectral, alpha=0.7, label='Spectral Branch (Global)')

    # 添加边界线使图形更清晰
    ax.plot(time, time_branch_norm, color=color_time, linewidth=1.5, alpha=0.8)
    ax.plot(time, time_branch_norm + spectral_branch_norm,
           color=color_spectral, linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('L2 Norm of Features', fontsize=12, fontweight='bold')
    ax.set_title('Branch Contribution Analysis', fontsize=13, fontweight='bold', pad=15)

    # 移除网格线
    ax.grid(False)

    # 图例
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # 设置刻度
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_mamba_fusion_gate(time, fusion_alpha, save_path):
    """
    Fig 1: Mamba融合门控机制
    展示learned fusion gate (alpha)随时间的动态变化
    alpha接近1表示更依赖Time Branch，接近0表示更依赖Mamba Branch
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    # 绘制融合权重曲线
    ax.plot(time, fusion_alpha, linewidth=2.5, color='#7B68EE',
            marker='o', markersize=5, markevery=max(1, len(time)//15),
            label=r'$\alpha$ (Fusion Weight)')

    # 添加参考线
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.5,
               label='Balanced (0.5)')
    ax.axhline(y=1.0, color='#4A90E2', linestyle=':', linewidth=1.2, alpha=0.4)
    ax.axhline(y=0.0, color='#E94B3C', linestyle=':', linewidth=1.2, alpha=0.4)

    # 添加区域标注
    ax.fill_between(time, 0.5, 1.0, alpha=0.1, color='#4A90E2',
                    label='Time-dominant')
    ax.fill_between(time, 0.0, 0.5, alpha=0.1, color='#E94B3C',
                    label='Mamba-dominant')

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Fusion Weight $\alpha$', fontsize=12, fontweight='bold')
    ax.set_title('Learned Fusion Gate Dynamics', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(-0.05, 1.05)

    # 移除网格线
    ax.grid(False)

    # 图例
    ax.legend(loc='best', fontsize=9, framealpha=0.9)

    # 设置刻度
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_mamba_branch_contribution(time, time_branch_norm, mamba_branch_norm, save_path):
    """
    Fig 3: Mamba分支贡献分析 (堆叠面积图)
    对比Time Branch（局部）和Mamba Branch（全局）的贡献
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    # 颜色方案
    color_time = '#4A90E2'      # 蓝色 - 时间分支(局部，FiLM-MLP)
    color_mamba = '#9B59B6'     # 紫色 - Mamba分支(全局，状态空间)

    # 绘制堆叠面积图
    ax.fill_between(time, 0, time_branch_norm,
                    color=color_time, alpha=0.7, label='Time Branch (Local FiLM-MLP)')

    ax.fill_between(time, time_branch_norm,
                    time_branch_norm + mamba_branch_norm,
                    color=color_mamba, alpha=0.7, label='Mamba Branch (Global SSM)')

    # 添加边界线使图形更清晰
    ax.plot(time, time_branch_norm, color=color_time, linewidth=1.5, alpha=0.8)
    ax.plot(time, time_branch_norm + mamba_branch_norm,
           color=color_mamba, linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('L2 Norm of Features', fontsize=12, fontweight='bold')
    ax.set_title('Branch Contribution Analysis (Time vs Mamba)', fontsize=13, fontweight='bold', pad=15)

    # 移除网格线
    ax.grid(False)

    # 图例
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # 设置刻度
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_deepfilm_dynamics(logs, save_dir='./figures'):
    """
    Generate 3 paper-quality figures for DeepFiLMVectorField internal dynamics.

    Arguments:
        logs (dict): Log data dict from DeepFiLMVectorField.extract_logs(), containing:
            - 't' or 'time': time point array [T]
            - 'gamma': FiLM gamma at layer 0 [T, D]
            - 'beta': FiLM beta at layer 0 [T, D]
            - 'layer_norms': L2 norms at each layer [T, num_layers]
        save_dir (str): Directory to save figures.
    """
    os.makedirs(save_dir, exist_ok=True)

    time = logs.get('t', logs.get('time', None))
    if time is None:
        raise ValueError("Logs must contain 't' or 'time' key")

    time = _to_numpy(time)
    gamma = _to_numpy(logs['gamma'])
    beta = _to_numpy(logs['beta'])
    layer_norms = _to_numpy(logs['layer_norms'])

    print("\n" + "=" * 70)
    print("Generating DeepFiLM paper figures")
    print("=" * 70)

    # Fig 1: FiLM Temporal Adaptation (gamma/beta over time)
    print("\n[1/3] Fig 1: FiLM Temporal Adaptation...")
    fig1_path = os.path.join(save_dir, 'fig1_deepfilm_film_adaptation.pdf')
    _plot_deepfilm_film_adaptation(time, gamma, beta, fig1_path)
    print("      Saved: {}".format(fig1_path))

    # Fig 2: Layer-wise Activation Norms over time
    print("[2/3] Fig 2: Layer-wise Activation Dynamics...")
    fig2_path = os.path.join(save_dir, 'fig2_deepfilm_layer_norms.pdf')
    _plot_deepfilm_layer_norms(time, layer_norms, fig2_path)
    print("      Saved: {}".format(fig2_path))

    # Fig 3: FiLM Modulation Intensity (|gamma|, |beta| over time)
    print("[3/3] Fig 3: FiLM Modulation Intensity...")
    fig3_path = os.path.join(save_dir, 'fig3_deepfilm_modulation_intensity.pdf')
    _plot_deepfilm_modulation_intensity(time, gamma, beta, fig3_path)
    print("      Saved: {}".format(fig3_path))

    print("\n" + "=" * 70)
    print("All DeepFiLM figures generated!")
    print("Location: {}".format(os.path.abspath(save_dir)))
    print("=" * 70 + "\n")


def _plot_deepfilm_film_adaptation(time, gamma, beta, save_path):
    """
    Fig 1: FiLM temporal adaptation for DeepFiLM.
    Shows mean gamma and beta over time with std shading.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    if len(gamma.shape) > 1:
        gamma_mean = gamma.mean(axis=1)
        gamma_std = gamma.std(axis=1)
    else:
        gamma_mean = gamma
        gamma_std = None

    if len(beta.shape) > 1:
        beta_mean = beta.mean(axis=1)
        beta_std = beta.std(axis=1)
    else:
        beta_mean = beta
        beta_std = None

    color_gamma = '#1E88E5'
    color_beta = '#FFC107'

    markevery = max(1, len(time) // 10)

    ax.plot(time, gamma_mean, color=color_gamma, linewidth=2.5,
            label=r'$\bar{\gamma}$ (scale)', marker='s', markersize=4,
            markevery=markevery)
    if gamma_std is not None:
        ax.fill_between(time, gamma_mean - gamma_std, gamma_mean + gamma_std,
                        color=color_gamma, alpha=0.15)

    ax.plot(time, beta_mean, color=color_beta, linewidth=2.5,
            label=r'$\bar{\beta}$ (shift)', marker='^', markersize=4,
            markevery=markevery)
    if beta_std is not None:
        ax.fill_between(time, beta_mean - beta_std, beta_mean + beta_std,
                        color=color_beta, alpha=0.15)

    ax.axhline(y=0.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.5)

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('FiLM Parameters (Layer 0)', fontsize=12, fontweight='bold')
    ax.set_title('FiLM Temporal Adaptation (DeepFiLM)', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_deepfilm_layer_norms(time, layer_norms, save_path):
    """
    Fig 2: Layer-wise activation L2 norms over time.
    Each line is one hidden layer, showing how representations evolve at different depths.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    colors = ['#1E88E5', '#43A047', '#FFC107', '#E53935']
    num_layers = layer_norms.shape[1] if layer_norms.ndim == 2 else 1
    markevery = max(1, len(time) // 10)

    if layer_norms.ndim == 1:
        ax.plot(time, layer_norms, color=colors[0], linewidth=2.5,
                label='Layer 1', marker='o', markersize=4, markevery=markevery)
    else:
        for i in range(num_layers):
            ax.plot(time, layer_norms[:, i], color=colors[i % len(colors)],
                    linewidth=2.5, label='Layer {}'.format(i + 1),
                    marker=['o', 's', '^', 'D'][i % 4], markersize=4,
                    markevery=markevery)

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('L2 Norm of Hidden Features', fontsize=12, fontweight='bold')
    ax.set_title('Layer-wise Activation Dynamics (DeepFiLM)', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_deepfilm_modulation_intensity(time, gamma, beta, save_path):
    """
    Fig 3: FiLM modulation intensity over time.
    Shows |gamma|_mean and |beta|_mean to quantify how strongly time modulates features.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    if len(gamma.shape) > 1:
        gamma_intensity = np.abs(gamma).mean(axis=1)
    else:
        gamma_intensity = np.abs(gamma)

    if len(beta.shape) > 1:
        beta_intensity = np.abs(beta).mean(axis=1)
    else:
        beta_intensity = np.abs(beta)

    color_gamma = '#1E88E5'
    color_beta = '#E53935'
    markevery = max(1, len(time) // 10)

    ax.plot(time, gamma_intensity, color=color_gamma, linewidth=2.5,
            label=r'$|\gamma|$ (scale magnitude)', marker='s', markersize=4,
            markevery=markevery)
    ax.plot(time, beta_intensity, color=color_beta, linewidth=2.5,
            label=r'$|\beta|$ (shift magnitude)', marker='^', markersize=4,
            markevery=markevery)

    ax.fill_between(time, 0, gamma_intensity, color=color_gamma, alpha=0.1)
    ax.fill_between(time, 0, beta_intensity, color=color_beta, alpha=0.1)

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Absolute FiLM Parameters', fontsize=12, fontweight='bold')
    ax.set_title('FiLM Modulation Intensity over Time (DeepFiLM)', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_speech_confusion_matrix(y_true, y_pred, save_dir='./figures',
                                  class_names=None):
    """Plot confusion matrix for Speech Commands (10-class) classification.

    Arguments:
        y_true: numpy array of true labels [N]
        y_pred: numpy array of predicted labels [N]
        save_dir: directory to save figure
        class_names: list of class names (default: 10 speech commands)
    """
    from sklearn.metrics import confusion_matrix

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'fig_speech_confusion_matrix.pdf')

    if class_names is None:
        class_names = ['yes', 'no', 'up', 'down', 'left',
                       'right', 'on', 'off', 'stop', 'go']

    cm = confusion_matrix(y_true, y_pred)
    # Normalize by row (true label)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues',
                   vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion', fontsize=11, fontweight='bold')

    n_classes = len(class_names)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, fontsize=10, rotation=45, ha='right')
    ax.set_yticklabels(class_names, fontsize=10)

    # Annotate cells with counts and percentages
    thresh = cm_norm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            color = 'white' if cm_norm[i, j] > thresh else 'black'
            text = '{}\n({:.0f}%)'.format(cm[i, j], cm_norm[i, j] * 100)
            ax.text(j, i, text, ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix (Speech Commands)', fontsize=13,
                 fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Print per-class accuracy
    per_class_acc = cm_norm.diagonal()
    print("[OK] Confusion matrix saved: {}".format(save_path))
    print("     Per-class accuracy:")
    for name, acc in zip(class_names, per_class_acc):
        print("       {}: {:.1f}%".format(name, acc * 100))
    print("     Overall accuracy: {:.1f}%".format(
        cm.diagonal().sum() / cm.sum() * 100))

    return save_path


def plot_ablation_bar(results, save_dir='./figures', metric_name='AUROC'):
    """Plot ablation study bar chart for CCF-B paper.

    Arguments:
        results: ordered dict of {model_label: metric_value}
                 e.g. {'Baseline NCDE': 0.8986, 'Deep MLP': 0.889,
                        'Deep MLP + Dropout': 0.895, 'DeepFiLM (Ours)': 0.903}
        save_dir: directory to save figure
        metric_name: y-axis label (default 'AUROC')
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'fig_ablation_bar.pdf')

    labels = list(results.keys())
    values = list(results.values())
    n = len(labels)

    # Color scheme: last bar (ours) is highlighted
    colors = ['#95a5a6'] * (n - 1) + ['#2c3e50']
    edge_colors = ['#7f8c8d'] * (n - 1) + ['#1a252f']

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars = ax.bar(range(n), values, color=colors, edgecolor=edge_colors,
                  linewidth=1.2, width=0.55, zorder=3)

    # Annotate value on top of each bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.001,
                '{:.4f}'.format(val), ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    # Dashed reference line at baseline
    baseline = values[0]
    ax.axhline(y=baseline, color='#c0392b', linestyle='--', linewidth=1.2,
               alpha=0.7, label='Baseline ({:.4f})'.format(baseline))

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Component Contribution', fontsize=13,
                 fontweight='bold', pad=12)

    # Tight y-axis range to amplify differences
    y_min = min(values) - 0.01
    y_max = max(values) + 0.015
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("Ablation bar chart saved: {}".format(save_path))
    return save_path


def plot_nfe_comparison(nfe_data, save_dir='./figures'):
    """Plot NFE (Number of Function Evaluations) boxplot for CCF-B paper.

    Arguments:
        nfe_data: dict of {model_label: list_of_nfe_values}
                  e.g. {'Baseline NCDE': [120, 132, ...], 'DeepFiLM': [115, 128, ...]}
        save_dir: directory to save figure
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'fig_nfe_boxplot.pdf')

    labels = list(nfe_data.keys())
    data = [nfe_data[k] for k in labels]
    n = len(labels)

    colors = ['#95a5a6'] * (n - 1) + ['#2c3e50']

    fig, ax = plt.subplots(figsize=(6, 4))

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color='white', linewidth=2))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    for whisker in bp['whiskers']:
        whisker.set(color='#555555', linewidth=1.2)
    for cap in bp['caps']:
        cap.set(color='#555555', linewidth=1.2)
    for flier in bp['fliers']:
        flier.set(marker='o', color='#555555', alpha=0.5, markersize=4)

    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(labels, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel('NFE (Number of Function Evaluations)', fontsize=11,
                  fontweight='bold')
    ax.set_title('ODE Solver Efficiency Comparison', fontsize=13,
                 fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("NFE boxplot saved: {}".format(save_path))
    return save_path


def plot_tsne_trajectory(result, device='cuda', save_dir='./figures',
                          n_samples=600, time_aware=True):
    """Plot t-SNE visualization of hidden state trajectories at multiple time points.

    Extracts z(t_0), z(t_mid), z(t_end) from the NeuralCDE hidden state
    and applies t-SNE to show class separation evolution over time.

    Arguments:
        result: experiment result dict from common.main()
                must have: model, times, test_dataloader
        device: computation device
        save_dir: directory to save figure
        n_samples: max samples per time-point to use for t-SNE
        time_aware: whether the model uses time-aware CDE (default True for DeepFiLM)
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[ERROR] scikit-learn required for t-SNE. Install: pip install scikit-learn")
        return None

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'fig_tsne_trajectory.pdf')

    model = result.model.to(device)
    times = result.times.to(device)
    test_dataloader = result.test_dataloader
    kwargs = {'time_aware': time_aware} if time_aware else {}

    # Navigate model hierarchy to get the inner NeuralCDE
    # Chain: _SqueezeEnd -> InitialValueNetwork -> NeuralCDE
    inner = model
    while hasattr(inner, 'model'):
        inner = inner.model
    ncde_model = inner  # Should be NeuralCDE at this point

    # Collect hidden trajectories via forward hook on ncde_model.linear
    # With stream=True, input[0] to linear has shape (batch, T, hidden_channels)
    hidden_list = []
    label_list = []

    hook_handle = ncde_model.linear.register_forward_hook(
        lambda mod, inp, out: hidden_list.append(inp[0].detach().cpu())
    )

    model.eval()
    with torch.no_grad():
        total_collected = 0
        for batch in test_dataloader:
            if total_collected >= n_samples:
                break
            batch = tuple(b.to(device) for b in batch)
            *coeffs, true_y, lengths = batch

            try:
                # stream=True returns all intermediate hidden states
                _ = model(times, coeffs, lengths, stream=True, **kwargs)
                label_list.append(true_y.detach().cpu())
                total_collected += true_y.shape[0]
            except Exception:
                # Fallback: some model wrappers may not support stream=True cleanly
                pass

    hook_handle.remove()
    model.to('cpu')

    if not hidden_list:
        print("[WARNING] No hidden states collected. Skipping t-SNE.")
        return None

    # hidden_list: list of (batch, T, hidden_channels) or (batch, hidden_channels)
    # For stream=True the shape should be (batch, T, hidden_channels)
    all_hidden = torch.cat(hidden_list, dim=0)   # (N, T, H) or (N, H)
    all_labels = torch.cat(label_list, dim=0).numpy()  # (N,)

    if all_hidden.dim() == 2:
        print("[WARNING] Got 2D hidden states - stream=True may not have worked. "
              "Plotting single time-point t-SNE.")
        z_points = [all_hidden.numpy()]
        point_labels = ['t_final']
        time_colors = ['#2c3e50']
    else:
        T = all_hidden.shape[1]
        t0_idx = 0
        tmid_idx = T // 2
        tend_idx = T - 1

        z_points = [
            all_hidden[:, t0_idx, :].numpy(),
            all_hidden[:, tmid_idx, :].numpy(),
            all_hidden[:, tend_idx, :].numpy(),
        ]
        point_labels = ['t = t_0', 't = t_mid', 't = t_end']
        time_colors = ['#3498db', '#e67e22', '#2ecc71']

    # Subsample if too large
    N = min(n_samples, all_hidden.shape[0])
    idx = np.random.choice(all_hidden.shape[0], N, replace=False)
    all_labels = all_labels[idx]

    fig, axes = plt.subplots(1, len(z_points), figsize=(5 * len(z_points), 4.5))
    if len(z_points) == 1:
        axes = [axes]

    class_colors = {0: '#3498db', 1: '#e74c3c'}
    class_names = {0: 'Sepsis Neg.', 1: 'Sepsis Pos.'}

    for ax, z, tlabel in zip(axes, z_points, point_labels):
        z_sub = z[idx]

        print("  Running t-SNE for {} ({} samples, {} dims)...".format(
            tlabel, N, z_sub.shape[1]))
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, N // 4),
                    n_iter=1000)
        z_2d = tsne.fit_transform(z_sub)

        for cls in [0, 1]:
            mask = all_labels == cls
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                       c=class_colors[cls], label=class_names[cls],
                       alpha=0.55, s=12, linewidths=0)

        ax.set_title(tlabel, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=9, loc='best', framealpha=0.85)
        ax.set_aspect('equal', 'box')
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    fig.suptitle('Hidden State t-SNE Trajectory (DeepFiLM)', fontsize=13,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("t-SNE trajectory saved: {}".format(save_path))
    return save_path


def load_logs_from_npz(npz_path):
    """从.npz文件加载日志数据"""
    data = np.load(npz_path, allow_pickle=True)
    logs = {key: data[key] for key in data.files}
    return logs


def load_logs_from_model(model_path, device='cpu'):
    """
    从训练好的模型加载并提取日志

    参数:
        model_path: 模型文件路径(.pt)
        device: 运行设备

    返回:
        logs: 日志数据字典
    """
    print(f"\n从模型文件加载: {model_path}")

    # 加载result对象
    result = torch.load(model_path, map_location=device)

    # 导入分析工具
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    sys.path.insert(0, parent_dir)

    from analyze_spectral_dynamics import run_inference_with_logging

    # 提取vector_field
    vector_field = result.vector_field
    model = result.model
    times = result.times
    test_dataloader = result.test_dataloader

    # 运行推理并记录日志
    print("运行推理并记录内部动态...")
    kwargs = {'time_aware': True}
    logs = run_inference_with_logging(model, vector_field, test_dataloader,
                                     times, device, kwargs)

    print(f"记录了 {len(logs['time'])} 个时间步的数据")

    return logs


def run_dummy_experiment():
    """
    运行一个模拟实验用于演示
    生成假的日志数据并调用plot_mechanism_dynamics
    """
    print("\n" + "=" * 70)
    print("运行模拟实验 (用于演示)")
    print("=" * 70 + "\n")

    # 生成模拟数据
    T = 50  # 时间步数
    D = 32  # 隐藏维度
    F = 16  # 频率数量

    time = np.linspace(0, 1, T)

    # 模拟FiLM参数: 随时间变化
    gamma = np.sin(2 * np.pi * time[:, None]) * 0.3 + 1.0 + np.random.randn(T, D) * 0.05
    beta = np.cos(2 * np.pi * time[:, None]) * 0.2 + np.random.randn(T, D) * 0.05

    # 模拟频谱权重: 低通滤波器特性
    freq_indices = np.arange(F)
    spectral_weights = np.exp(-freq_indices / 3.0) + np.random.randn(T, F) * 0.05
    spectral_weights = np.maximum(spectral_weights, 0)  # 确保非负

    # 归一化
    spectral_weights = spectral_weights / spectral_weights.sum(axis=1, keepdims=True)

    # 模拟分支贡献: 时间分支逐渐减弱,频谱分支逐渐增强
    time_branch_norm = 2.0 - time * 1.0 + np.random.randn(T) * 0.1
    spectral_branch_norm = 0.5 + time * 1.5 + np.random.randn(T) * 0.1

    contribution_ratio = spectral_branch_norm / (time_branch_norm + 1e-6)

    # 构造日志字典
    logs = {
        't': time,
        'gamma': gamma,
        'beta': beta,
        'spectral_weights': spectral_weights,
        'time_branch_norm': time_branch_norm,
        'spectral_branch_norm': spectral_branch_norm,
        'contribution_ratio': contribution_ratio
    }

    print("模拟数据已生成:")
    print(f"  时间步数: {T}")
    print(f"  隐藏维度: {D}")
    print(f"  频率数量: {F}")

    # 调用绘图函数
    plot_mechanism_dynamics(logs, save_dir='./paper_figures_demo')

    print("\n提示: 这是使用模拟数据生成的演示图表")
    print("      实际使用时请传入真实的训练日志")


def plot_lowrankode_dynamics(logs, save_dir='./figures'):
    """
    Generate 4 paper-quality figures for LowRankODE-FiLM internal dynamics.

    Arguments:
        logs (dict): Log data from LowRankODE_FiLM_VectorField.extract_logs(), containing:
            - 'time': time points [T]
            - 'gamma': FiLM gamma [T, D]
            - 'beta': FiLM beta [T, D]
            - 'alpha': SSM scale factor [T]
            - 'gate_mean': mean gate activation [T]
            - 'A_base': S4D diagonal [D]
            - 'ssm_norm': SSM branch L2 norm [T]
            - 'mlp_norm': MLP branch L2 norm [T]
        save_dir (str): Directory to save figures.
    """
    os.makedirs(save_dir, exist_ok=True)

    time = _to_numpy(logs.get('time', logs.get('t', None)))
    if time is None or len(time) == 0:
        raise ValueError("Logs must contain 'time' or 't' key with non-empty data")

    gamma = _to_numpy(logs['gamma'])
    beta = _to_numpy(logs['beta'])
    alpha = _to_numpy(logs['alpha'])
    gate_mean = _to_numpy(logs['gate_mean'])
    A_base = _to_numpy(logs['A_base'])
    ssm_norm = _to_numpy(logs['ssm_norm'])
    mlp_norm = _to_numpy(logs['mlp_norm'])

    print("\n" + "=" * 70)
    print("Generating LowRankODE-FiLM paper figures")
    print("=" * 70)

    # Fig 1: SSM Structured Dynamics (A_base spectrum + alpha over time)
    print("\n[1/4] Fig 1: SSM Structured Dynamics...")
    fig1_path = os.path.join(save_dir, 'fig1_lowrankode_ssm_dynamics.pdf')
    _plot_lowrankode_ssm_dynamics(time, A_base, alpha, fig1_path)
    print("      Saved: {}".format(fig1_path))

    # Fig 2: Selective Gating over Time
    print("[2/4] Fig 2: Selective Gating...")
    fig2_path = os.path.join(save_dir, 'fig2_lowrankode_gating.pdf')
    _plot_lowrankode_gating(time, gate_mean, fig2_path)
    print("      Saved: {}".format(fig2_path))

    # Fig 3: Branch Contributions (SSM vs MLP norms)
    print("[3/4] Fig 3: Branch Contributions...")
    fig3_path = os.path.join(save_dir, 'fig3_lowrankode_branch_contributions.pdf')
    _plot_lowrankode_branches(time, ssm_norm, mlp_norm, fig3_path)
    print("      Saved: {}".format(fig3_path))

    # Fig 4: FiLM Temporal Adaptation
    print("[4/4] Fig 4: FiLM Temporal Adaptation...")
    fig4_path = os.path.join(save_dir, 'fig4_lowrankode_film_adaptation.pdf')
    _plot_lowrankode_film_adaptation(time, gamma, beta, fig4_path)
    print("      Saved: {}".format(fig4_path))

    print("\n" + "=" * 70)
    print("All LowRankODE-FiLM figures generated!")
    print("Location: {}".format(os.path.abspath(save_dir)))
    print("=" * 70 + "\n")


def _plot_lowrankode_ssm_dynamics(time, A_base, alpha, save_path):
    """
    Fig 1: SSM structured dynamics.
    Left: S4D diagonal spectrum (A_base values)
    Right: Alpha (low-rank scale) over time
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=300)

    # Left: S4D Diagonal Spectrum
    indices = np.arange(len(A_base))
    ax1.bar(indices, A_base, color='#1E88E5', alpha=0.7, edgecolor='black', linewidth=0.8)
    ax1.axhline(y=0, color='gray', linestyle=':', linewidth=1.2)
    ax1.set_xlabel('State Dimension Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('A_base Value (S4D Init)', fontsize=12, fontweight='bold')
    ax1.set_title('S4D Diagonal Base Matrix', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(False)
    ax1.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    # Right: Alpha over time
    markevery = max(1, len(time) // 10)
    ax2.plot(time, alpha, color='#E53935', linewidth=2.5, marker='o',
             markersize=4, markevery=markevery, label=r'$\alpha$ (low-rank scale)')
    ax2.fill_between(time, 0, alpha, color='#E53935', alpha=0.15)
    ax2.axhline(y=0.05, color='gray', linestyle='--', linewidth=1.2,
                alpha=0.5, label='Initial value (0.05)')
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'$\alpha$ Value', fontsize=12, fontweight='bold')
    ax2.set_title('Low-Rank Perturbation Scale', fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='best', fontsize=10, framealpha=0.9)
    ax2.grid(False)
    ax2.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_lowrankode_gating(time, gate_mean, save_path):
    """
    Fig 2: Selective gating over time.
    Shows mean gate activation (how much SSM features are used).
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    markevery = max(1, len(time) // 10)
    ax.plot(time, gate_mean, color='#43A047', linewidth=2.5, marker='s',
            markersize=4, markevery=markevery, label='Mean Gate Activation')
    ax.fill_between(time, 0, gate_mean, color='#43A047', alpha=0.15)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.2,
               alpha=0.5, label='50% activation')

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gate Value (0=off, 1=on)', fontsize=12, fontweight='bold')
    ax.set_title('Selective Gating: SSM Activation over Time', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_lowrankode_branches(time, ssm_norm, mlp_norm, save_path):
    """
    Fig 3: Branch contributions.
    Shows SSM vs MLP branch L2 norms to visualize which dominates over time.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    markevery = max(1, len(time) // 10)

    ax.plot(time, ssm_norm, color='#1E88E5', linewidth=2.5, marker='o',
            markersize=4, markevery=markevery, label='SSM Branch (structured)')
    ax.fill_between(time, 0, ssm_norm, color='#1E88E5', alpha=0.1)

    ax.plot(time, mlp_norm, color='#FFC107', linewidth=2.5, marker='^',
            markersize=4, markevery=markevery, label='MLP Branch (unstructured)')
    ax.fill_between(time, 0, mlp_norm, color='#FFC107', alpha=0.1)

    # Contribution ratio
    ratio = ssm_norm / (mlp_norm + 1e-8)
    ax_ratio = ax.twinx()
    ax_ratio.plot(time, ratio, color='#9C27B0', linewidth=2.0, linestyle=':',
                  marker='D', markersize=3, markevery=markevery, label='SSM/MLP ratio')
    ax_ratio.set_ylabel('SSM/MLP Ratio', fontsize=11, fontweight='bold', color='#9C27B0')
    ax_ratio.tick_params(axis='y', labelcolor='#9C27B0', labelsize=10)

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('L2 Norm of Features', fontsize=12, fontweight='bold')
    ax.set_title('Branch Contributions: SSM vs MLP', fontsize=13, fontweight='bold', pad=15)

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_ratio.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10, framealpha=0.9)

    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_lowrankode_film_adaptation(time, gamma, beta, save_path):
    """
    Fig 4: FiLM temporal adaptation (same as DeepFiLM but for LowRankODE-FiLM).
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

    if len(gamma.shape) > 1:
        gamma_mean = gamma.mean(axis=1)
        gamma_std = gamma.std(axis=1)
    else:
        gamma_mean = gamma
        gamma_std = None

    if len(beta.shape) > 1:
        beta_mean = beta.mean(axis=1)
        beta_std = beta.std(axis=1)
    else:
        beta_mean = beta
        beta_std = None

    color_gamma = '#1E88E5'
    color_beta = '#FFC107'
    markevery = max(1, len(time) // 10)

    ax.plot(time, gamma_mean, color=color_gamma, linewidth=2.5,
            label=r'$\bar{\gamma}$ (scale)', marker='s', markersize=4,
            markevery=markevery)
    if gamma_std is not None:
        ax.fill_between(time, gamma_mean - gamma_std, gamma_mean + gamma_std,
                        color=color_gamma, alpha=0.15)

    ax.plot(time, beta_mean, color=color_beta, linewidth=2.5,
            label=r'$\bar{\beta}$ (shift)', marker='^', markersize=4,
            markevery=markevery)
    if beta_std is not None:
        ax.fill_between(time, beta_mean - beta_std, beta_mean + beta_std,
                        color=color_beta, alpha=0.15)

    ax.axhline(y=0.0, color='gray', linestyle=':', linewidth=1.2, alpha=0.5)

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('FiLM Parameters (Layer 0)', fontsize=12, fontweight='bold')
    ax.set_title('FiLM Temporal Adaptation (LowRankODE-FiLM)', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_main_results(results_by_dataset, save_dir='./figures', metric='auroc'):
    """Plot grouped bar chart comparing models across datasets.

    Arguments:
        results_by_dataset: nested dict
            {dataset_name: {model_name: (mean, std)}}
            e.g. {
                'Sepsis': {
                    'NCDE': (0.8986, 0.003),
                    'ODE-RNN': (0.877, 0.004),
                    'GRU-ODE': (0.869, 0.005),
                    'DeepFiLM (Ours)': (0.903, 0.002),
                },
                'Decompensation': {...},
            }
        save_dir: directory to save figures
        metric: metric name for y-axis label (e.g. 'auroc' or 'auprc')
    """
    os.makedirs(save_dir, exist_ok=True)
    metric_upper = metric.upper()
    save_path = os.path.join(save_dir, 'fig_main_results_{}.pdf'.format(metric))

    datasets = list(results_by_dataset.keys())
    # Infer model order from first dataset
    models = list(results_by_dataset[datasets[0]].keys())
    n_datasets = len(datasets)
    n_models = len(models)

    # Color palette: grey tones for baselines, dark for ours (last model)
    palette = ['#aab4c4', '#7f8fa6', '#525f7f', '#2c3e50']
    colors = [palette[i % len(palette)] for i in range(n_models - 1)] + ['#c0392b']
    hatch_patterns = ['', '///', '\\\\\\', 'xxx', '...']

    x = np.arange(n_datasets)
    total_width = 0.72
    bar_width = total_width / n_models
    offsets = np.linspace(-(total_width - bar_width) / 2,
                           (total_width - bar_width) / 2, n_models)

    fig, ax = plt.subplots(figsize=(max(5, 2.5 * n_datasets), 5), dpi=300)

    for i, (model, color) in enumerate(zip(models, colors)):
        means = []
        stds = []
        for ds in datasets:
            val = results_by_dataset[ds].get(model, (0.0, 0.0))
            means.append(val[0])
            stds.append(val[1])

        bars = ax.bar(x + offsets[i], means, bar_width,
                      label=model, color=color,
                      hatch=hatch_patterns[i % len(hatch_patterns)],
                      edgecolor='white', linewidth=0.8,
                      yerr=stds, capsize=3,
                      error_kw=dict(elinewidth=1.2, ecolor='#444444'))

        # Annotate value on each bar
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + max(stds) + 0.003,
                    '{:.3f}'.format(m),
                    ha='center', va='bottom', fontsize=7.5, rotation=90,
                    fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_upper, fontsize=13, fontweight='bold')
    ax.set_title('Performance Comparison ({})'.format(metric_upper),
                 fontsize=14, fontweight='bold', pad=14)

    all_means = [v[0] for ds in results_by_dataset.values() for v in ds.values()]
    y_min = max(0, min(all_means) - 0.03)
    y_max = min(1.0, max(all_means) + 0.05)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

    ax.legend(fontsize=9.5, framealpha=0.92, loc='lower right',
              edgecolor='#cccccc')
    ax.grid(axis='y', linestyle='--', alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("Main results chart saved: {}".format(save_path))
    return save_path


def plot_roc_pr_curves(curves_by_model, save_dir='./figures', dataset_name='Sepsis'):
    """Plot ROC and PR curves side by side for multiple models.

    Arguments:
        curves_by_model: dict
            {model_name: {'fpr': array, 'tpr': array, 'auroc': float,
                          'precision': array, 'recall': array, 'auprc': float}}
            Obtain these from sklearn.metrics.roc_curve and precision_recall_curve.
        save_dir: directory to save figures
        dataset_name: used in title and filename
    """
    import sklearn.metrics
    os.makedirs(save_dir, exist_ok=True)
    tag = dataset_name.lower().replace(' ', '_')
    save_path = os.path.join(save_dir, 'fig_roc_pr_{}.pdf'.format(tag))

    palette = ['#7f8fa6', '#525f7f', '#aab4c4', '#c0392b']
    linestyles = ['--', '-.', ':', '-']

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    models = list(curves_by_model.keys())
    for i, model in enumerate(models):
        d = curves_by_model[model]
        color = palette[i % len(palette)]
        ls = linestyles[i % len(linestyles)]
        lw = 2.8 if i == len(models) - 1 else 1.8

        ax_roc.plot(d['fpr'], d['tpr'], color=color, lw=lw, ls=ls,
                    label='{} (AUC={:.3f})'.format(model, d['auroc']))
        ax_pr.plot(d['recall'], d['precision'], color=color, lw=lw, ls=ls,
                   label='{} (AP={:.3f})'.format(model, d['auprc']))

    # Random baseline
    ax_roc.plot([0, 1], [0, 1], color='#bbbbbb', lw=1.2, ls=':', label='Random')

    ax_roc.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax_roc.set_title('ROC Curve - {}'.format(dataset_name),
                     fontsize=13, fontweight='bold', pad=12)
    ax_roc.legend(fontsize=9, framealpha=0.9, loc='lower right')
    ax_roc.set_xlim([-0.01, 1.01])
    ax_roc.set_ylim([-0.01, 1.01])
    ax_roc.set_aspect('equal')
    ax_roc.grid(False)
    ax_roc.tick_params(labelsize=10, width=1.2)

    ax_pr.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax_pr.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax_pr.set_title('PR Curve - {}'.format(dataset_name),
                    fontsize=13, fontweight='bold', pad=12)
    ax_pr.legend(fontsize=9, framealpha=0.9, loc='upper right')
    ax_pr.set_xlim([-0.01, 1.01])
    ax_pr.set_ylim([-0.01, 1.01])
    ax_pr.set_aspect('equal')
    ax_pr.grid(False)
    ax_pr.tick_params(labelsize=10, width=1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("ROC/PR curves saved: {}".format(save_path))
    return save_path


def plot_convergence_curves(history_by_model, save_dir='./figures',
                             dataset_name='Sepsis', metric='auroc'):
    """Plot training convergence curves (val metric vs epoch) for multiple models.

    Arguments:
        history_by_model: dict
            {model_name: list_of_dicts}
            Each dict has 'epoch' and the metric key (e.g. 'auroc').
            This is the 'history' list returned by common._train_loop().
        save_dir: directory to save figures
        dataset_name: used in title and filename
        metric: key to extract from each history entry's val_metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    tag = dataset_name.lower().replace(' ', '_')
    save_path = os.path.join(save_dir, 'fig_convergence_{}_{}.pdf'.format(tag, metric))

    palette = ['#aab4c4', '#7f8fa6', '#525f7f', '#c0392b']
    linestyles = ['--', '-.', ':', '-']

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    models = list(history_by_model.keys())
    for i, model in enumerate(models):
        history = history_by_model[model]
        epochs = [h.epoch for h in history]
        values = [getattr(h.val_metrics, metric) for h in history]
        color = palette[i % len(palette)]
        ls = linestyles[i % len(linestyles)]
        lw = 2.5 if i == len(models) - 1 else 1.8

        ax.plot(epochs, values, color=color, lw=lw, ls=ls,
                label='{} (best={:.4f})'.format(model, max(values)),
                marker='o' if i == len(models) - 1 else None,
                markersize=3,
                markevery=max(1, len(epochs) // 15))

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Val {}'.format(metric.upper()), fontsize=12, fontweight='bold')
    ax.set_title('Convergence Curves - {} ({})'.format(dataset_name, metric.upper()),
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10, width=1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("Convergence curves saved: {}".format(save_path))
    return save_path


def plot_param_efficiency(model_stats, save_dir='./figures'):
    """Plot parameter efficiency scatter: #params vs AUROC for all models.

    Arguments:
        model_stats: list of dicts, each with:
            {'name': str, 'params': int, 'auroc': float,
             'auroc_std': float (optional), 'ours': bool}
        e.g. [
            {'name': 'NCDE', 'params': 350000, 'auroc': 0.8986, 'ours': False},
            {'name': 'ODE-RNN', 'params': 420000, 'auroc': 0.877, 'ours': False},
            {'name': 'GRU-ODE', 'params': 390000, 'auroc': 0.869, 'ours': False},
            {'name': 'DeepFiLM', 'params': 240000, 'auroc': 0.903, 'ours': True},
        ]
        save_dir: directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'fig_param_efficiency.pdf')

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    for stat in model_stats:
        color = '#c0392b' if stat.get('ours', False) else '#7f8fa6'
        marker = '*' if stat.get('ours', False) else 'o'
        size = 220 if stat.get('ours', False) else 90
        zorder = 5 if stat.get('ours', False) else 3

        std = stat.get('auroc_std', 0.0)
        ax.errorbar(stat['params'] / 1000, stat['auroc'],
                    yerr=std, fmt=marker, color=color,
                    markersize=np.sqrt(size), capsize=4,
                    elinewidth=1.5, zorder=zorder)

        # Label offset: push 'ours' label up, others right
        x_off = stat['params'] / 1000 * 0.02
        y_off = 0.003 if not stat.get('ours', False) else 0.005
        ax.annotate(stat['name'],
                    xy=(stat['params'] / 1000, stat['auroc']),
                    xytext=(stat['params'] / 1000 + x_off,
                            stat['auroc'] + y_off),
                    fontsize=10,
                    fontweight='bold' if stat.get('ours', False) else 'normal',
                    color=color)

    ax.set_xlabel('Parameters (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUROC', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Efficiency Comparison', fontsize=13,
                 fontweight='bold', pad=12)

    # Pareto frontier annotation
    all_auroc = [s['auroc'] for s in model_stats]
    y_min = min(all_auroc) - 0.015
    y_max = max(all_auroc) + 0.015
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

    ax.grid(linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10, width=1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("Parameter efficiency scatter saved: {}".format(save_path))
    return save_path


def generate_all_paper_figures(results, save_dir='./paper_figures'):
    """Convenience wrapper: generate all paper figures from a results dict.

    Arguments:
        results: dict with keys:
            'main_results'   -> input to plot_main_results()
            'roc_pr'         -> {dataset: input to plot_roc_pr_curves()}
            'convergence'    -> {dataset: input to plot_convergence_curves()}
            'param_stats'    -> input to plot_param_efficiency()
            'ablation'       -> input to plot_ablation_bar()
            'deepfilm_logs'  -> input to plot_deepfilm_dynamics()
            'tsne_result'    -> result object for plot_tsne_trajectory()
    """
    os.makedirs(save_dir, exist_ok=True)
    generated = []

    if 'main_results' in results:
        for metric in ['auroc', 'auprc']:
            p = plot_main_results(results['main_results'], save_dir, metric)
            generated.append(p)

    if 'roc_pr' in results:
        for ds_name, curves in results['roc_pr'].items():
            p = plot_roc_pr_curves(curves, save_dir, ds_name)
            generated.append(p)

    if 'convergence' in results:
        for ds_name, hist in results['convergence'].items():
            for metric in ['auroc', 'loss']:
                p = plot_convergence_curves(hist, save_dir, ds_name, metric)
                generated.append(p)

    if 'param_stats' in results:
        p = plot_param_efficiency(results['param_stats'], save_dir)
        generated.append(p)

    if 'ablation' in results:
        p = plot_ablation_bar(results['ablation'], save_dir)
        generated.append(p)

    if 'deepfilm_logs' in results:
        plot_deepfilm_dynamics(results['deepfilm_logs'], save_dir)

    if 'tsne_result' in results:
        plot_tsne_trajectory(results['tsne_result'], save_dir=save_dir)

    print("\nAll paper figures generated in: {}".format(os.path.abspath(save_dir)))
    print("Total: {} files".format(len(generated)))
    return generated


def parse_args():
    parser = argparse.ArgumentParser(description='生成CCF-B论文图表')

    parser.add_argument('--model-path', type=str, default=None,
                       help='训练好的模型文件路径 (.pt)')
    parser.add_argument('--logs-path', type=str, default=None,
                       help='已保存的日志文件路径 (.npz)')
    parser.add_argument('--output-dir', type=str, default='./paper_figures',
                       help='图表保存目录 (default: ./paper_figures)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='设备 (default: cpu)')
    parser.add_argument('--demo', action='store_true',
                       help='运行模拟实验演示')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.demo:
        # 运行演示
        run_dummy_experiment()

    elif args.model_path:
        # 从模型文件加载
        logs = load_logs_from_model(args.model_path, args.device)
        plot_mechanism_dynamics(logs, args.output_dir)

    elif args.logs_path:
        # 从日志文件加载
        logs = load_logs_from_npz(args.logs_path)
        plot_mechanism_dynamics(logs, args.output_dir)

    else:
        print("错误: 请指定以下选项之一:")
        print("  --model-path <path>  : 从模型文件生成图表")
        print("  --logs-path <path>   : 从日志文件生成图表")
        print("  --demo               : 运行模拟实验")
        print("\n使用 --help 查看完整帮助")


if __name__ == '__main__':
    main()
