"""
统一实验运行脚本
支持所有数据集 (sepsis, uea, speech_commands) 和所有模型

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

    # 其他选项
    python run_experiment.py --dataset sepsis --model ncde --repeats 5
    python run_experiment.py --dataset sepsis --model ncde --cpu
    python run_experiment.py --dataset sepsis --model ncde --dry-run
"""

import sys
import os
import argparse

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
        'odernn': {'hidden_channels': 128, 'hidden_hidden_channels': 128, 'num_hidden_layers': 4},
        'gruode': {'hidden_channels': 187, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
        'dt': {'hidden_channels': 187, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
        'decay': {'hidden_channels': 187, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
    },
    'uea': {
        'ncde': {'hidden_channels': 32, 'hidden_hidden_channels': 32, 'num_hidden_layers': 3},
        'ncde-film': {'hidden_channels': 32, 'hidden_hidden_channels': 32, 'num_hidden_layers': 3},
        'ncde-spectral': {'hidden_channels': 32, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
        'odernn': {'hidden_channels': 32, 'hidden_hidden_channels': 32, 'num_hidden_layers': 3},
        'gruode': {'hidden_channels': 47, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
        'dt': {'hidden_channels': 47, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
        'decay': {'hidden_channels': 47, 'hidden_hidden_channels': None, 'num_hidden_layers': None},
    },
    'speech': {
        'ncde': {'hidden_channels': 90, 'hidden_hidden_channels': 40, 'num_hidden_layers': 4},
        'ncde-film': {'hidden_channels': 90, 'hidden_hidden_channels': 40, 'num_hidden_layers': 4},
        'ncde-spectral': {'hidden_channels': 90, 'hidden_hidden_channels': 40, 'num_hidden_layers': 4},
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
                        choices=['sepsis', 'uea', 'speech'],
                        help='数据集: sepsis, uea, 或 speech')

    # 模型选择
    parser.add_argument('--model', type=str, default='ncde',
                        choices=['ncde', 'ncde-film', 'ncde-spectral', 'odernn', 'gruode', 'dt', 'decay'],
                        help='模型类型 (default: ncde)')
    parser.add_argument('--all-models', action='store_true',
                        help='运行所有模型')

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

    # 可视化选项
    parser.add_argument('--visualize', action='store_true',
                        help='训练后进行可视化分析 (仅支持 ncde-spectral)')
    parser.add_argument('--output', type=str, default=None,
                        help='可视化输出路径 (default: auto)')

    return parser.parse_args()


def get_default_epochs(dataset):
    """获取默认训练轮数"""
    defaults = {'sepsis': 200, 'uea': 1000, 'speech': 200}
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


def visualize_results(result, dataset_name, output_path, device):
    """可视化实验结果"""
    sys.path.insert(0, os.path.join(script_dir, 'analysis'))
    import paper_plots

    print("\n" + "=" * 70)
    print("开始分析内部动态...")
    print("=" * 70 + "\n")

    # Extract vector field and run inference with logging
    model = result.model.to(device)
    model.eval()
    vector_field = result.vector_field
    times = result.times.to(device)
    test_dataloader = result.test_dataloader
    kwargs = {'time_aware': True}

    # Enable logging on the spectral vector field
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
    paper_plots.plot_mechanism_dynamics(logs, save_dir=output_dir)

    # Save logs as npz
    npz_path = output_path.replace('.png', '_logs.npz')
    np.savez(npz_path, **logs)

    print("\n可视化完成！")
    print(f"  - 可视化图表: {output_dir}")
    print(f"  - 日志数据: {npz_path}")


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
        models = ['ncde', 'ncde-film', 'ncde-spectral', 'odernn', 'gruode', 'dt', 'decay']
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

            # 打印结果
            print_results(result, model_name, dataset_display)

            # 保存结果
            model_results.append(result)

            # 可视化 (仅第一次运行且用户请求)
            if i == 0 and args.visualize:
                if model_name == 'ncde-spectral':
                    if args.output:
                        output_path = args.output
                    else:
                        output_path = f'{args.dataset}_{model_name}_dynamics.png'

                    visualize_results(result, dataset_display, output_path, device)
                else:
                    print(f"\n[WARNING] 可视化仅支持 ncde-spectral 模型，跳过 {model_name}")

        all_results[model_name] = model_results

    # 汇总多次运行的结果
    if args.all_models or args.repeats > 1:
        print("\n" + "=" * 70)
        print("实验汇总")
        print("=" * 70)

        for model_name, results in all_results.items():
            if args.dataset == 'sepsis':
                # Sepsis 使用 AUROC
                aurocs = [r.test_metrics.auroc for r in results]
                if len(aurocs) > 1:
                    print(f"{model_name:15s}: AUROC = {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
                else:
                    print(f"{model_name:15s}: AUROC = {aurocs[0]:.4f}")
            else:
                # UEA 和 Speech 使用准确率
                accs = [r.test_metrics.accuracy for r in results]
                if len(accs) > 1:
                    print(f"{model_name:15s}: ACC = {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
                else:
                    print(f"{model_name:15s}: ACC = {accs[0]:.4f}")

        print("=" * 70)

    print("\n任务完成！\n")


if __name__ == '__main__':
    main()
