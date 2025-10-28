"""
测试安装是否正确
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """测试所有模块是否能正常导入"""
    print("测试模块导入...")
    
    try:
        from config import config
        print("✓ config 模块导入成功")
    except Exception as e:
        print(f"✗ config 模块导入失败: {e}")
        return False
    
    try:
        from src import data_generation
        print("✓ data_generation 模块导入成功")
    except Exception as e:
        print(f"✗ data_generation 模块导入失败: {e}")
        return False
    
    try:
        from src.models import heston, dkl, mv
        print("✓ models 模块导入成功")
    except Exception as e:
        print(f"✗ models 模块导入失败: {e}")
        return False
    
    try:
        from src.utils import metrics
        print("✓ utils 模块导入成功")
    except Exception as e:
        print(f"✗ utils 模块导入失败: {e}")
        return False
    
    return True


def test_dependencies():
    """测试关键依赖包"""
    print("\n测试依赖包...")
    
    required_packages = [
        'numpy',
        'pandas',
        'scipy',
        'sklearn',
        'skopt',
        'matplotlib',
        'numba'
    ]
    
    all_ok = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装")
            all_ok = False
    
    return all_ok


def test_directories():
    """测试必要的目录是否存在"""
    print("\n测试目录结构...")
    
    required_dirs = [
        project_root / 'config',
        project_root / 'src',
        project_root / 'src' / 'models',
        project_root / 'src' / 'utils',
        project_root / 'scripts',
        project_root / 'data',
        project_root / 'data' / 'processed',
        project_root / 'data' / 'results'
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"✓ {dir_path.relative_to(project_root)} 存在")
        else:
            print(f"✗ {dir_path.relative_to(project_root)} 不存在")
            all_ok = False
    
    return all_ok


def main():
    """运行所有测试"""
    print("="*60)
    print("期权对冲比较项目 - 安装测试")
    print("="*60)
    
    results = []
    
    # 测试模块导入
    results.append(("模块导入", test_imports()))
    
    # 测试依赖包
    results.append(("依赖包", test_dependencies()))
    
    # 测试目录结构
    results.append(("目录结构", test_directories()))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "通过 ✓" if passed else "失败 ✗"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n恭喜！所有测试通过，项目配置正确！")
        print("现在可以运行: python scripts/run_all.py")
    else:
        print("\n部分测试失败，请检查上述错误信息。")
        print("确保已运行: pip install -r requirements.txt")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

