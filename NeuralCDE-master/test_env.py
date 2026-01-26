"""
环境测试脚本
运行此脚本检查所有依赖是否正确安装
"""

import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def check_import(module_name, package_name=None):
    """检查模块是否可导入"""
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        print(f"[OK] {package_name}")
        return True
    except ImportError as e:
        print(f"[FAIL] {package_name}: {e}")
        return False


def main():
    print("=" * 50)
    print("Neural CDE 环境检查")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"项目路径: {project_root}")
    print()

    # 检查基本依赖
    print("[1] 基本依赖:")
    all_ok = True
    all_ok &= check_import('torch', 'PyTorch')
    all_ok &= check_import('numpy', 'NumPy')
    all_ok &= check_import('sklearn', 'scikit-learn')
    all_ok &= check_import('tqdm', 'tqdm')

    # 检查torchdiffeq
    print("\n[2] 神经ODE依赖:")
    all_ok &= check_import('torchdiffeq', 'torchdiffeq')

    # 检查CUDA
    print("\n[3] GPU支持:")
    import torch
    if torch.cuda.is_available():
        print(f"[OK] CUDA可用: {torch.cuda.get_device_name(0)}")
        print(f"     CUDA版本: {torch.version.cuda}")
        print(f"     GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("[WARN] CUDA不可用,将使用CPU训练(较慢)")

    # 检查项目模块
    print("\n[4] 项目模块:")
    experiments_dir = os.path.join(project_root, 'experiments')
    sys.path.insert(0, experiments_dir)
    os.chdir(experiments_dir)
    all_ok &= check_import('controldiffeq', 'controldiffeq')
    all_ok &= check_import('common', 'experiments/common')
    all_ok &= check_import('datasets', 'experiments/datasets')

    # 检查数据目录
    print("\n[5] 数据目录:")
    data_dir = os.path.join(project_root, 'experiments', 'datasets', 'data', 'sepsis')
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        print(f"[OK] 数据目录存在: {len(files)} 个文件/文件夹")
    else:
        print(f"[INFO] 数据目录不存在,首次运行时会自动下载")
        print(f"       目录: {data_dir}")

    # 总结
    print("\n" + "=" * 50)
    if all_ok:
        print("[SUCCESS] 环境检查通过!")
        print("\n运行实验:")
        print("  python run_sepsis.py              # 默认NCDE模型")
        print("  python run_sepsis.py --model ncde # 指定模型")
        print("  python run_sepsis.py --cpu        # 使用CPU")
        print("  python run_sepsis.py --dry-run    # 测试模式")
        print("  python run_sepsis.py --all        # 所有模型")
    else:
        print("[FAILED] 部分依赖缺失,请安装:")
        print("  pip install torch torchdiffeq scikit-learn tqdm numpy")
    print("=" * 50)


if __name__ == '__main__':
    main()
