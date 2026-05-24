# 论文图表生成工具

## 简介

`paper_plots.py` 是用于生成 CCF-B 论文的高质量可视化图表的工具。它可以从训练好的模型中提取内部动态信息,并生成3个符合论文标准的图表。

## 生成的图表

### Fig 1: Spectral Gating Mechanism (频谱门控机制)
- **内容**: 展示频谱权重的幅度
- **X轴**: 频率索引
- **Y轴**: 权重幅度
- **特性**: 自动检测并标注低通滤波行为

### Fig 2: FiLM Temporal Adaptation (时间自适应调制)
- **内容**: 双轴图展示FiLM参数和贡献率的时间演化
- **左轴**: gamma (缩放) 和 beta (偏移) 参数
- **右轴**: 频谱/时间分支贡献率
- **特性**: 展示模型如何随时间调制特征

### Fig 3: Branch Contribution Analysis (分支贡献分析)
- **内容**: 堆叠面积图展示两个分支的L2范数
- **图例**: Time Branch (Local) 和 Spectral Branch (Global)
- **特性**: 直观展示局部和全局特征的相对贡献

## 使用方法

### 方法 1: 运行演示 (使用模拟数据)

```bash
cd experiments/analysis
python paper_plots.py --demo
```

这会在 `./paper_figures_demo` 目录生成3个示例图表。

### 方法 2: 从训练好的模型生成

```bash
python paper_plots.py --model-path <path_to_model.pt> --output-dir ./my_figures
```

### 方法 3: 从已保存的日志生成

如果你已经运行过 `analyze_spectral_dynamics.py` 并保存了 `.npz` 日志文件:

```bash
python paper_plots.py --logs-path <path_to_logs.npz> --output-dir ./my_figures
```

### 方法 4: 在Python代码中调用

```python
import paper_plots

# 假设你已经有了logs字典
logs = {
    't': time_array,
    'gamma': gamma_array,
    'beta': beta_array,
    'spectral_weights': spectral_weights_array,
    'time_branch_norm': time_branch_norm_array,
    'spectral_branch_norm': spectral_branch_norm_array,
    'contribution_ratio': contribution_ratio_array
}

# 生成图表
paper_plots.plot_mechanism_dynamics(logs, save_dir='./figures')
```

## 完整工作流

### 1. 训练模型并可视化

使用 `run_experiment.py` 训练模型并自动生成可视化:

```bash
cd experiments
python run_experiment.py --dataset sepsis --model ncde-spectral --visualize
```

这会:
1. 训练 ncde-spectral 模型
2. 调用 `analyze_spectral_dynamics.py` 生成4面板分析图
3. 保存日志数据为 `.npz` 文件

### 2. 生成论文级别图表

使用保存的日志数据生成符合论文要求的图表:

```bash
cd analysis
python paper_plots.py --logs-path ../sepsis_ncde-spectral_dynamics_logs.npz --output-dir ./paper_figures
```

### 3. 在论文中使用

生成的PDF文件 (300 DPI) 可以直接插入到 LaTeX 论文中:

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.48\textwidth]{figures/fig1_spectral_gating.pdf}
    \caption{Spectral gating mechanism showing low-pass filtering behavior.}
    \label{fig:spectral_gating}
\end{figure}
```

## 图表特性

- ✅ **高分辨率**: 300 DPI,适合出版
- ✅ **无网格线**: 符合多数期刊要求
- ✅ **Times New Roman字体**: 学术论文标准字体
- ✅ **PDF格式**: 矢量图形,无损缩放
- ✅ **专业配色**: 使用色盲友好的配色方案

## 命令行参数

```
--model-path PATH     从模型文件生成图表 (.pt)
--logs-path PATH      从日志文件生成图表 (.npz)
--output-dir DIR      图表保存目录 (default: ./paper_figures)
--device DEVICE       运行设备 (cpu/cuda, default: cpu)
--demo                运行模拟实验演示
--help                显示帮助信息
```

## 依赖项

- numpy
- matplotlib >= 3.3
- seaborn
- torch

## 故障排除

### 问题: "找不到Times New Roman字体"

如果系统没有Times New Roman,matplotlib会自动fallback到DejaVu Serif。你也可以手动安装字体。

### 问题: "日志中必须包含 't' 或 'time' 键"

确保你的日志数据包含时间信息。如果使用 `analyze_spectral_dynamics.py`,时间键名为 `'time'`。

### 问题: GPU内存不足

在从模型加载时,使用 `--device cpu` 参数:

```bash
python paper_plots.py --model-path model.pt --device cpu
```

## 与现有工具的对比

| 特性 | analyze_spectral_dynamics.py | paper_plots.py |
|------|------------------------------|----------------|
| 用途 | 快速分析和调试 | 论文图表生成 |
| 图表数量 | 4个子图在一张图上 | 3个独立的PDF图表 |
| 网格线 | 有 | 无 |
| 字体 | 默认 | Times New Roman |
| 格式 | PNG | PDF (矢量) |
| 堆叠面积图 | 无 | 有 |
| 注释 | 无 | 有 (低通滤波) |

## 示例输出

运行演示后,你会看到:

```
======================================================================
生成论文图表
======================================================================

[1/3] 生成 Fig 1: Spectral Gating Mechanism...
      已保存: ./paper_figures_demo/fig1_spectral_gating.pdf
[2/3] 生成 Fig 2: FiLM Temporal Adaptation...
      已保存: ./paper_figures_demo/fig2_film_adaptation.pdf
[3/3] 生成 Fig 3: Branch Contribution Analysis...
      已保存: ./paper_figures_demo/fig3_branch_contribution.pdf

======================================================================
所有图表生成完成!
保存位置: D:\实验室\mamba-cde\NeuralCDE-master\experiments\analysis\paper_figures_demo
======================================================================
```

## 贡献

如需修改图表样式或添加新的可视化,请编辑 `paper_plots.py` 中的相应函数:
- `_plot_spectral_gating()` - Fig 1
- `_plot_film_adaptation()` - Fig 2
- `_plot_branch_contribution()` - Fig 3
