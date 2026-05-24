"""
统一实验运行脚本
支持所有数据集 (sepsis, uea, speech_commands, decompensation) 和所有模型

用法示例:
    # Sepsis 数据集
    python run_experiment.py --dataset sepsis --model ncde-spectral
    python run_experiment.py --dataset sepsis --all-models
    python run_experiment.py --dataset sepsis --model ncde-spectral --visualize

    # UEA 数据集
    python run_experiment.py --dataset uea --model ncde --missing-rate 0.5
    python run_experiment.py --dataset uea --all-models --missing-rate 0.3

    # Speech Commands 数据集
    python run_experiment.py --dataset speech --model ncde

    # MIMIC-III Decompensation 数据集
    python run_experiment.py --dataset decompensation --model ncde-deepfilm --data-dir /path/to/data
    python run_experiment.py --dataset decompensation --model ncde --data-dir /path/to/data

    # 其他选项
    python run_experiment.py --dataset sepsis --model ncde --repeats 5
    python run_experiment.py --dataset sepsis --model ncde --cpu
    python run_experiment.py --dataset sepsis --model ncde --dry-run
"""

import sys
import os
import argparse
import gc

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

import torch
import numpy as np


# 统一的模型超参数配置
MODEL_CONFIGS = {
    'sepsis': {
        'ncde': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'ncde-film': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'ncde-spectral': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'ncde-spectral-v2': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'ncde-deepfilm': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'ncde-lowrankode-film': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'ncde-deepmlp': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'ncde-deepmlp-dropout': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'ncde-mamba': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'odernn': {'hidden_channels': 128, 'hidden_hidden_channels': 128, 'num_hidden_layers': 4},
        'gruode': {'hidden_channels': 187, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
        'dt': {'hidden_channels': 187, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
        'decay': {'hidden_channels': 187, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
    },
    'uea': {
        'ncde': {'hidden_channels': 32, 'hidden_hidden_channels': 32, 'num_hidden_layers': 3},
        'ncde-film': {'hidden_channels': 32, 'hidden_hidden_channels': 32, 'num_hidden_layers': 3},
        'ncde-spectral': {'hidden_channels': 32, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
        'ncde-spectral-v2': {'hidden_channels': 32, 'hidden_hidden_channels': 32, 'num_hidden_layers': 3},
        'ncde-deepfilm': {'hidden_channels': 32, 'hidden_hidden_channels': 32, 'num_hidden_layers': 3},
        'ncde-lowrankode-film': {'hidden_channels': 32, 'hidden_hidden_channels': 32, 'num_hidden_layers': 3},
        'ncde-deepmlp': {'hidden_channels': 32, 'hidden_hidden_channels': 32, 'num_hidden_layers': 3},
        'ncde-deepmlp-dropout': {'hidden_channels': 32, 'hidden_hidden_channels': 32, 'num_hidden_layers': 3},
        'ncde-mamba': {'hidden_channels': 32, 'hidden_hidden_channels': 32, 'num_hidden_layers': 3},
        'odernn': {'hidden_channels': 32, 'hidden_hidden_channels': 32, 'num_hidden_layers': 3},
        'gruode': {'hidden_channels': 47, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
        'dt': {'hidden_channels': 47, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
        'decay': {'hidden_channels': 47, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
    },
    'decompensation': {
        'ncde': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'ncde-film': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'ncde-spectral': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'ncde-spectral-v2': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'ncde-deepfilm': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'ncde-lowrankode-film': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'ncde-deepmlp': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'ncde-deepmlp-dropout': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'ncde-mamba': {'hidden_channels': 64, 'hidden_hidden_channels': 49, 'num_hidden_layers': 4},
        'odernn': {'hidden_channels': 128, 'hidden_hidden_channels': 128, 'num_hidden_layers': 4},
        'gruode': {'hidden_channels': 187, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
        'dt': {'hidden_channels': 187, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
        'decay': {'hidden_channels': 187, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
    },
    'speech': {
        'ncde': {'hidden_channels': 90, 'hidden_hidden_channels': 40, 'num_hidden_layers': 4},
        'ncde-film': {'hidden_channels': 90, 'hidden_hidden_channels': 40, 'num_hidden_layers': 4},
        'ncde-spectral': {'hidden_channels': 90, 'hidden_hidden_channels': 40, 'num_hidden_layers': 4},
        'ncde-spectral-v2': {'hidden_channels': 90, 'hidden_hidden_channels': 40, 'num_hidden_layers': 4},
        'ncde-deepfilm': {'hidden_channels': 90, 'hidden_hidden_channels': 40, 'num_hidden_layers': 4},
        'ncde-lowrankode-film': {'hidden_channels': 90, 'hidden_hidden_channels': 40, 'num_hidden_layers': 4},
        'ncde-deepmlp': {'hidden_channels': 90, 'hidden_hidden_channels': 40, 'num_hidden_layers': 4},
        'ncde-deepmlp-dropout': {'hidden_channels': 90, 'hidden_hidden_channels': 40, 'num_hidden_layers': 4},
        'ncde-mamba': {'hidden_channels': 90, 'hidden_hidden_channels': 40, 'num_hidden_layers': 4},
        'odernn': {'hidden_channels': 128, 'hidden_hidden_channels': 64, 'num_hidden_layers': 4},
        'gruode': {'hidden_channels': 160, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
        'dt': {'hidden_channels': 160, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
        'decay': {'hidden_channels': 160, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description='统一实验运行脚本')

    # 数据集选择
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['sepsis', 'uea', 'speech', 'decompensation'],
                        help='数据集: sepsis, uea, speech, 或 decompensation')

    # 模型选择
    parser.add_argument('--model', type=str, default='ncde',
                        choices=['ncde', 'ncde-film', 'ncde-spectral', 'ncde-spectral-v2',
                                 'ncde-deepfilm', 'ncde-lowrankode-film', 'ncde-deepmlp',
                                 'ncde-deepmlp-dropout', 'ncde-mamba', 'odernn', 'gruode',
                                 'dt', 'decay'],
                        help='模型类型 (default: ncde)')
    parser.add_argument('--all-models', action='store_true',
                        help='运行所有模型')
    parser.add_argument('--ablation', action='store_true',
                        help='运行消融实验: ncde + ncde-deepmlp + ncde-deepmlp-dropout + ncde-deepfilm')

    # 训练参数
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='设备: cuda 或 cpu (default: cuda)')
    parser.add_argument('--cpu', action='store_true',
                        help='使用 CPU (等同于 --device cpu)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数 (default: sepsis=200, uea=1000, speech=200)')
    parser.add_argument('--repeats', type=int, default=1,
                        help='重复实验次数 (default: 1)')
    parser.add_argument('--dry-run', action='store_true',
                        help='测试模式，不保存结果')

    # 数据集特定参数
    parser.add_argument('--intensity', action='store_true', default=True,
                        help='[Sepsis] 使用观察强度特征 (default: True)')
    parser.add_argument('--no-intensity', action='store_false', dest='intensity',
                        help='[Sepsis] 不使用观察强度特征')
    parser.add_argument('--missing-rate', type=float, default=0.5,
                        help='[UEA] 缺失率 (default: 0.5)')
    parser.add_argument('--uea-dataset', type=str, default='CharacterTrajectories',
                        help='[UEA] UEA 数据集名称 (default: CharacterTrajectories)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='[Decompensation] mimic3-benchmarks 数据目录路径')
    parser.add_argument('--max-hours', type=int, default=168,
                        help='[Decompensation] 最大小时数截断 (default: 168, 即7天)')

    # 可视化选项
    parser.add_argument('--visualize', action='store_true',
                        help='训练后进行可视化分析 (支持 ncde-spectral 和 ncde-mamba)')
    parser.add_argument('--output', type=str, default=None,
                        help='可视化输出路径 (default: auto)')
    parser.add_argument('--nfe', action='store_true',
                        help='测试后统计 NFE (Number of Function Evaluations)，需要使用自适应求解器 dopri5')
    parser.add_argument('--tsne', action='store_true',
                        help='测试后生成隐空间 t-SNE 轨迹图 (需要 ncde-deepfilm 或其他支持 stream=True 的模型)')

    return parser.parse_args()


def get_default_epochs(dataset):
    """获取默认训练轮数"""
    defaults = {'sepsis': 200, 'uea': 200, 'speech': 200, 'decompensation': 200}
    return defaults.get(dataset, 200)


def run_sepsis_experiment(model_name, intensity, device, max_epochs, dry_run, config):
    """运行 Sepsis 实验"""
    import sepsis

    print("=" * 70)
    print(f"模型: {model_name.upper()}")
    print(f"数据集: Sepsis")
    print(f"设备: {device}")
    print(f"强度特征: {'Yes' if intensity else 'No'}")
    print(f"超参数: {config}")
    print("=" * 70)

    result = sepsis.main(
        intensity=intensity,
        device=device,
        model_name=model_name,
        max_epochs=max_epochs,
        dry_run=dry_run,
        **config
    )

    return result


def run_uea_experiment(model_name, dataset_name, missing_rate, device, max_epochs, dry_run, config):
    """运行 UEA 实验"""
    import uea

    print("=" * 70)
    print(f"模型: {model_name.upper()}")
    print(f"数据集: UEA - {dataset_name}")
    print(f"设备: {device}")
    print(f"缺失率: {missing_rate:.0%}")
    print(f"超参数: {config}")
    print("=" * 70)

    result = uea.main(
        dataset_name=dataset_name,
        missing_rate=missing_rate,
        device=device,
        model_name=model_name,
        max_epochs=max_epochs,
        dry_run=dry_run,
        **config
    )

    return result


def run_speech_experiment(model_name, device, max_epochs, dry_run, config):
    """运行 Speech Commands 实验"""
    import speech_commands

    print("=" * 70)
    print(f"模型: {model_name.upper()}")
    print(f"数据集: Speech Commands")
    print(f"设备: {device}")
    print(f"超参数: {config}")
    print("=" * 70)

    result = speech_commands.main(
        device=device,
        model_name=model_name,
        max_epochs=max_epochs,
        dry_run=dry_run,
        **config
    )

    return result


def run_decompensation_experiment(model_name, device, max_epochs, dry_run, config,
                                  data_dir=None, max_hours=168):
    """运行 MIMIC-III Decompensation 实验"""
    import decompensation

    print("=" * 70)
    print("模型: {}".format(model_name.upper()))
    print("数据集: MIMIC-III Decompensation")
    print("设备: {}".format(device))
    print("数据目录: {}".format(data_dir or 'default'))
    print("超参数: {}".format(config))
    print("=" * 70)

    extra_kwargs = {}
    if data_dir is not None:
        extra_kwargs['data_dir'] = data_dir

    result = decompensation.main(
        device=device,
        model_name=model_name,
        max_epochs=max_epochs,
        dry_run=dry_run,
        max_hours=max_hours,
        **extra_kwargs,
        **config
    )

    return result


def print_results(result, model_name, dataset):
    """打印实验结果"""
    print("\n" + "=" * 70)
    print(f"[{model_name.upper()}] 实验结果 - {dataset}")
    print("=" * 70)

    print(f"\n训练集:")
    print(f"  准确率: {result.train_metrics.accuracy:.4f}")

    print(f"\n验证集:")
    print(f"  准确率: {result.val_metrics.accuracy:.4f}")

    print(f"\n测试集:")
    print(f"  准确率: {result.test_metrics.accuracy:.4f}")
    if hasattr(result.test_metrics, 'auroc'):
        print(f"  AUROC: {result.test_metrics.auroc:.4f}")
    if hasattr(result.test_metrics, 'average_precision'):
        print(f"  Average Precision: {result.test_metrics.average_precision:.4f}")

    print(f"\n模型信息:")
    print(f"  参数量: {result.parameters:,}")

    if hasattr(result, 'memory_usage') and result.memory_usage:
        print(f"  GPU 内存: {result.memory_usage / 1e9:.2f} GB")

    print("=" * 70)


def visualize_results(result, dataset_name, output_path, device, model_name):
    """可视化实验结果"""
    sys.path.insert(0, os.path.join(script_dir, 'analysis'))
    import paper_plots

    print("\n" + "=" * 70)
    print(f"开始分析 {model_name.upper()} 内部动态...")
    print("=" * 70 + "\n")

    # Extract vector field and run inference with logging
    model = result.model.to(device)
    model.eval()
    vector_field = result.vector_field
    times = result.times.to(device)
    test_dataloader = result.test_dataloader
    kwargs = {'time_aware': True}

    # Enable logging on the vector field
    vector_field.set_logging(True)

    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(b.to(device) for b in batch)
            *coeffs, true_y, lengths = batch
            model(times, coeffs, lengths, **kwargs)
            break  # one batch is enough for visualization

    # Extract logs and generate plots
    logs = vector_field.extract_logs()
    vector_field.set_logging(False)
    model.to('cpu')

    output_dir = os.path.dirname(output_path) or '.'

    # 根据模型类型选择可视化函数
    if model_name == 'ncde-spectral':
        paper_plots.plot_mechanism_dynamics(logs, save_dir=output_dir)
    elif model_name == 'ncde-mamba':
        paper_plots.plot_mamba_dynamics(logs, save_dir=output_dir)
    elif model_name == 'ncde-deepfilm':
        paper_plots.plot_deepfilm_dynamics(logs, save_dir=output_dir)
    elif model_name == 'ncde-lowrankode-film':
        paper_plots.plot_lowrankode_dynamics(logs, save_dir=output_dir)
    else:
        print(f"[WARNING] 未知的模型类型: {model_name}")
        return

    # Save logs as npz
    npz_path = output_path.replace('.png', '_logs.npz')
    np.savez(npz_path, **logs)

    # Generate confusion matrix for multi-class datasets (e.g. speech commands)
    num_classes = getattr(result, 'num_classes', 2)
    if num_classes > 2:
        print("\n[INFO] Generating confusion matrix for {}-class task...".format(num_classes))
        _generate_confusion_matrix(result, device, output_dir, model_name, paper_plots)

    print("\n可视化完成！")
    print("  - 可视化图表: {}".format(output_dir))
    print("  - 日志数据: {}".format(npz_path))


def _generate_confusion_matrix(result, device, output_dir, model_name, paper_plots):
    """Collect predictions on test set and generate confusion matrix."""
    model = result.model.to(device)
    model.eval()
    times = result.times.to(device)
    test_dataloader = result.test_dataloader

    time_aware = model_name in ('ncde-film', 'ncde-spectral', 'ncde-spectral-v2',
                                'ncde-mamba', 'ncde-deepfilm', 'ncde-lowrankode-film')
    kwargs = {'time_aware': True} if time_aware else {}

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(b.to(device) for b in batch)
            *coeffs, true_y, lengths = batch
            pred_y = model(times, coeffs, lengths, **kwargs)
            pred_labels = pred_y.argmax(dim=-1)
            all_true.append(true_y.cpu().numpy())
            all_pred.append(pred_labels.cpu().numpy())

    model.to('cpu')

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    paper_plots.plot_speech_confusion_matrix(y_true, y_pred, save_dir=output_dir)


def _collect_nfe(result, model_name, device, output_dir, time_aware=False, max_batches=30):
    """Collect NFE statistics using adaptive dopri5 solver on test set.

    Switches to dopri5 (adaptive solver) and counts function evaluations per batch.
    Results are printed and saved as a .npz file for later plotting.
    """
    print("\n" + "=" * 60)
    print("[NFE] Collecting NFE for {} (dopri5)...".format(model_name))
    print("=" * 60)

    model = result.model.to(device)
    times = result.times.to(device)
    test_dataloader = result.test_dataloader

    kwargs_nfe = {
        'adjoint': False,      # Must disable adjoint for NFE wrapping
        'method': 'dopri5',    # Adaptive solver - NFE varies per sample
        'nfe_counter': None,   # Will be replaced per batch
    }
    if time_aware:
        kwargs_nfe['time_aware'] = True

    model.eval()
    all_nfe = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            if batch_idx >= max_batches:
                break
            batch = tuple(b.to(device) for b in batch)
            *coeffs, true_y, lengths = batch

            nfe_counter = []
            kwargs_nfe['nfe_counter'] = nfe_counter
            try:
                model(times, coeffs, lengths, **kwargs_nfe)
                if nfe_counter:
                    all_nfe.append(nfe_counter[0])
            except Exception as e:
                print("[NFE WARNING] Batch {} failed: {}".format(batch_idx, e))

    model.to('cpu')

    if all_nfe:
        arr = np.array(all_nfe)
        print("[NFE] {} - mean={:.1f}  std={:.1f}  min={}  max={}".format(
            model_name, arr.mean(), arr.std(), arr.min(), arr.max()))
        os.makedirs(output_dir, exist_ok=True)
        nfe_path = os.path.join(output_dir, 'nfe_{}.npz'.format(model_name.replace('-', '_')))
        np.save(nfe_path, arr)
        print("[NFE] Saved: {}".format(nfe_path))
    else:
        print("[NFE WARNING] No NFE data collected for {}".format(model_name))

    return all_nfe


def main():
    args = parse_args()

    # 确定设备
    if args.cpu:
        device = 'cpu'
    elif args.device == 'cuda':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"[INFO] 使用 GPU: {torch.cuda.get_device_name(0)}\n")
        else:
            device = 'cpu'
            print("[WARNING] CUDA 不可用，使用 CPU\n")
    else:
        device = args.device

    # 确定训练轮数
    max_epochs = args.epochs if args.epochs else get_default_epochs(args.dataset)

    # 确定要运行的模型
    if args.all_models:
        models = ['ncde', 'ncde-film', 'ncde-spectral', 'ncde-deepfilm', 'ncde-mamba', 'odernn', 'gruode', 'dt', 'decay']
    elif args.ablation:
        models = ['ncde', 'ncde-deepmlp', 'ncde-deepmlp-dropout', 'ncde-deepfilm']
    else:
        models = [args.model]

    # 存储所有结果
    all_results = {}

    # 运行实验
    for model_name in models:
        model_results = []

        for i in range(args.repeats):
            if args.repeats > 1:
                print(f"\n>>> {model_name} - 第 {i+1}/{args.repeats} 次实验\n")

            # 获取模型配置
            config = MODEL_CONFIGS[args.dataset][model_name]

            # 根据数据集运行实验
            if args.dataset == 'sepsis':
                result = run_sepsis_experiment(
                    model_name=model_name,
                    intensity=args.intensity,
                    device=device,
                    max_epochs=max_epochs,
                    dry_run=args.dry_run,
                    config=config
                )
                dataset_display = 'Sepsis'

            elif args.dataset == 'uea':
                result = run_uea_experiment(
                    model_name=model_name,
                    dataset_name=args.uea_dataset,
                    missing_rate=args.missing_rate,
                    device=device,
                    max_epochs=max_epochs,
                    dry_run=args.dry_run,
                    config=config
                )
                dataset_display = f'{args.uea_dataset} (miss={args.missing_rate:.0%})'

            elif args.dataset == 'speech':
                result = run_speech_experiment(
                    model_name=model_name,
                    device=device,
                    max_epochs=max_epochs,
                    dry_run=args.dry_run,
                    config=config
                )
                dataset_display = 'Speech Commands'

            elif args.dataset == 'decompensation':
                result = run_decompensation_experiment(
                    model_name=model_name,
                    device=device,
                    max_epochs=max_epochs,
                    dry_run=args.dry_run,
                    config=config,
                    data_dir=args.data_dir,
                    max_hours=args.max_hours
                )
                dataset_display = 'MIMIC-III Decompensation'

            # 打印结果
            print_results(result, model_name, dataset_display)

            # 保存结果
            model_results.append(result)

            # 可视化 (仅第一次运行且用户请求)
            if i == 0 and args.visualize:
                if model_name in ('ncde-spectral', 'ncde-mamba', 'ncde-deepfilm', 'ncde-lowrankode-film'):
                    if args.output:
                        output_path = args.output
                    else:
                        output_path = '{}_{}_{}_dynamics.png'.format(args.dataset, model_name, i)

                    visualize_results(result, dataset_display, output_path, device, model_name)
                else:
                    print("\n[WARNING] 可视化仅支持 ncde-spectral、ncde-mamba、ncde-deepfilm 和 ncde-lowrankode-film 模型，跳过 {}".format(model_name))

            # NFE 统计 (仅第一次运行且用户请求)
            if i == 0 and args.nfe:
                nfe_output_dir = args.output if args.output else './paper_figures'
                _collect_nfe(result, model_name, device, nfe_output_dir,
                             time_aware=(model_name in ('ncde-film', 'ncde-spectral',
                                                        'ncde-spectral-v2', 'ncde-mamba',
                                                        'ncde-deepfilm', 'ncde-lowrankode-film')))

            # t-SNE 轨迹 (仅第一次运行且用户请求)
            if i == 0 and args.tsne:
                sys.path.insert(0, os.path.join(script_dir, 'analysis'))
                import paper_plots
                tsne_output_dir = args.output if args.output else './paper_figures'
                time_aware_flag = model_name in ('ncde-film', 'ncde-spectral',
                                                 'ncde-spectral-v2', 'ncde-mamba',
                                                 'ncde-deepfilm', 'ncde-lowrankode-film')
                paper_plots.plot_tsne_trajectory(
                    result, device=device, save_dir=tsne_output_dir,
                    time_aware=time_aware_flag
                )

        all_results[model_name] = model_results

        # 每个模型跑完后释放 GPU 内存，防止连续多模型实验 OOM
        for r in model_results:
            if hasattr(r, 'model') and r.model is not None:
                try:
                    r.model.to('cpu')
                except Exception:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 汇总多次运行的结果
    if args.all_models or args.ablation or args.repeats > 1:
        print("\n" + "=" * 70)
        print("实验汇总")
        print("=" * 70)

        for model_name, results in all_results.items():
            if args.dataset in ('sepsis', 'decompensation'):
                # Sepsis / Decompensation 使用 AUROC
                aurocs = [r.test_metrics.auroc for r in results]
                if len(aurocs) > 1:
                    print("{:25s}: AUROC = {:.4f} +/- {:.4f}".format(
                        model_name, np.mean(aurocs), np.std(aurocs)))
                else:
                    print("{:25s}: AUROC = {:.4f}".format(model_name, aurocs[0]))
            else:
                # UEA 和 Speech 使用准确率
                accs = [r.test_metrics.accuracy for r in results]
                if len(accs) > 1:
                    print("{:25s}: ACC = {:.4f} +/- {:.4f}".format(
                        model_name, np.mean(accs), np.std(accs)))
                else:
                    print("{:25s}: ACC = {:.4f}".format(model_name, accs[0]))

        print("=" * 70)

    print("\n任务完成！\n")


if __name__ == '__main__':
    main()
