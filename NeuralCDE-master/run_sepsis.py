"""
Sepsis 预测实验运行脚本
Neural CDE 项目

用法:
    python run_sepsis.py              # 运行单个NCDE模型
    python run_sepsis.py --model ncde # 指定模型
    python run_sepsis.py --all        # 运行所有模型
    python run_sepsis.py --cpu        # 使用CPU
    python run_sepsis.py --dry-run    # 测试模式(不保存结果)
    python run_sepsis.py --modulated  # 启用时间调制机制(仅NCDE有效)
"""

import sys
import os
import argparse

# 设置项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
experiments_dir = os.path.join(project_root, 'experiments')
sys.path.insert(0, project_root)
sys.path.insert(0, experiments_dir)
os.chdir(experiments_dir)

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Sepsis预测实验')
    parser.add_argument('--model', type=str, default='ncde',
                        choices=['ncde', 'odernn', 'gruode', 'dt', 'decay'],
                        help='模型类型 (default: ncde)')
    parser.add_argument('--intensity', action='store_true', default=True,
                        help='使用观察强度特征 (default: True)')
    parser.add_argument('--no-intensity', action='store_false', dest='intensity',
                        help='不使用观察强度特征')
    parser.add_argument('--cpu', action='store_true',
                        help='使用CPU训练')
    parser.add_argument('--epochs', type=int, default=200,
                        help='最大训练轮数 (default: 200)')
    parser.add_argument('--all', action='store_true',
                        help='运行所有模型')
    parser.add_argument('--dry-run', action='store_true',
                        help='测试模式,不保存结果')
    parser.add_argument('--repeats', type=int, default=1,
                        help='重复实验次数 (default: 1)')
    parser.add_argument('--modulated', action='store_true',
                        help='启用时间调制机制 (仅对 NCDE 模型有效)')
    return parser.parse_args()


# 各模型的推荐超参数
MODEL_CONFIGS = {
    'ncde': {
        'hidden_channels': 49,
        'hidden_hidden_channels': 49,
        'num_hidden_layers': 4
    },
    'odernn': {
        'hidden_channels': 128,
        'hidden_hidden_channels': 128,
        'num_hidden_layers': 4
    },
    'gruode': {
        'hidden_channels': 187,
        'hidden_hidden_channels': None,
        'num_hidden_layers': None
    },
    'dt': {
        'hidden_channels': 187,
        'hidden_hidden_channels': None,
        'num_hidden_layers': None
    },
    'decay': {
        'hidden_channels': 187,
        'hidden_hidden_channels': None,
        'num_hidden_layers': None
    }
}


def run_single_model(model_name, intensity, device, max_epochs, dry_run, modulated=False):
    """运行单个模型"""
    import sepsis

    config = MODEL_CONFIGS[model_name]

    print("=" * 60)
    print(f"模型: {model_name.upper()}")
    print(f"设备: {device}")
    print(f"强度特征: {'Yes' if intensity else 'No'}")
    print(f"时间调制 (Modulated): {'Yes' if modulated else 'No'}")
    print(f"超参数: {config}")
    print("=" * 60)

    result = sepsis.main(
        intensity=intensity,
        device=device,
        model_name=model_name,
        max_epochs=max_epochs,
        dry_run=dry_run,
        modulated=modulated,
        **config
    )

    return result


def print_results(result, model_name):
    """打印实验结果"""
    print("\n" + "=" * 60)
    print(f"[{model_name.upper()}] 实验结果")
    print("=" * 60)

    print(f"\n训练集:")
    print(f"  准确率: {result.train_metrics.accuracy:.4f}")

    print(f"\n验证集:")
    print(f"  准确率: {result.val_metrics.accuracy:.4f}")

    print(f"\n测试集:")
    print(f"  准确率: {result.test_metrics.accuracy:.4f}")
    print(f"  AUROC: {result.test_metrics.auroc:.4f}")
    print(f"  Average Precision: {result.test_metrics.average_precision:.4f}")

    print(f"\n模型信息:")
    print(f"  参数量: {result.parameters:,}")

    if hasattr(result, 'memory_usage') and result.memory_usage:
        print(f"  GPU内存: {result.memory_usage / 1e9:.2f} GB")

    print("=" * 60)


def main():
    args = parse_args()

    # 检查CUDA
    if args.cpu:
        device = 'cpu'
    elif torch.cuda.is_available():
        device = 'cuda'
        print(f"[INFO] 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("[WARN] CUDA不可用,使用CPU")

    if args.all:
        # 运行所有模型
        models = ['ncde', 'odernn', 'gruode', 'dt', 'decay']
        all_results = {}

        for model_name in models:
            for i in range(args.repeats):
                if args.repeats > 1:
                    print(f"\n>>> {model_name} - 第 {i+1}/{args.repeats} 次")

                result = run_single_model(
                    model_name=model_name,
                    intensity=args.intensity,
                    device=device,
                    max_epochs=args.epochs,
                    dry_run=args.dry_run,
                    modulated=args.modulated
                )
                print_results(result, model_name)

                if model_name not in all_results:
                    all_results[model_name] = []
                all_results[model_name].append(result.test_metrics.auroc)

        # 汇总结果
        print("\n" + "=" * 60)
        print("所有模型 AUROC 汇总")
        print("=" * 60)
        for model_name, aurocs in all_results.items():
            if len(aurocs) > 1:
                import numpy as np
                print(f"{model_name:8s}: {np.mean(aurocs):.4f} +/- {np.std(aurocs):.4f}")
            else:
                print(f"{model_name:8s}: {aurocs[0]:.4f}")
    else:
        # 运行单个模型
        for i in range(args.repeats):
            if args.repeats > 1:
                print(f"\n>>> 第 {i+1}/{args.repeats} 次实验")

            result = run_single_model(
                model_name=args.model,
                intensity=args.intensity,
                device=device,
                max_epochs=args.epochs,
                dry_run=args.dry_run,
                modulated=args.modulated
            )
            print_results(result, args.model)


if __name__ == '__main__':
    main()
