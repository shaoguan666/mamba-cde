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
