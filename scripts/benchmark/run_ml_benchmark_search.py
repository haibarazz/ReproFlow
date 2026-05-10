"""
参数搜索脚本
使用示例：
    python scripts/benchmark/run_ml_benchmark_search.py --model LogisticRegression --search-type grid
    python scripts/benchmark/run_ml_benchmark_search.py --model RandomForest --search-type random --n-iter 20
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_benchmark.models import MODEL_REGISTRY
from ml_benchmark.utils.param_search import ParamSearcher
from ml_benchmark.utils.data_loader import MLDataLoader
from Data_pre import get_data_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='ML 模型参数搜索',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python scripts/benchmark/run_ml_benchmark_search.py --model LogisticRegression --search-type grid
  python scripts/benchmark/run_ml_benchmark_search.py --model RandomForest --search-type random --n-iter 20
  python scripts/benchmark/run_ml_benchmark_search.py --model XGBoostClassifier --cv 3
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='模型名称（例如：LogisticRegression, RandomForest, XGBoostClassifier）'
    )

    parser.add_argument(
        '--search-type',
        type=str,
        default='grid',
        choices=['grid', 'random'],
        help='搜索类型：grid（网格搜索）或 random（随机搜索）'
    )

    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='交叉验证折数（默认：5）'
    )

    parser.add_argument(
        '--n-iter',
        type=int,
        default=10,
        help='随机搜索迭代次数（仅在 search-type=random 时有效，默认：10）'
    )

    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='并行任务数（默认：-1，使用所有 CPU）'
    )

    parser.add_argument(
        '--list-models',
        action='store_true',
        help='列出所有可用的模型'
    )

    return parser.parse_args()


def list_available_models():
    """列出所有可用模型"""
    print("=" * 60)
    print("📋 可用模型列表")
    print("=" * 60)

    # 按类别分组
    categories = {
        '线性模型': ['LogisticRegression', 'SGDClassifier', 'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet'],
        '树模型': ['DecisionTreeClassifier', 'DecisionTreeRegressor', 'RandomForestClassifier', 'RandomForestRegressor',
                  'ExtraTreesClassifier', 'ExtraTreesRegressor'],
        'Boosting': ['GradientBoostingClassifier', 'GradientBoostingRegressor', 'XGBoostClassifier', 'XGBoostRegressor',
                    'LightGBMClassifier', 'LightGBMRegressor', 'CatBoostClassifier', 'CatBoostRegressor',
                    'AdaBoostClassifier', 'AdaBoostRegressor'],
        'SVM': ['SVC', 'LinearSVC', 'SVR'],
        'KNN': ['KNeighborsClassifier', 'KNeighborsRegressor'],
        '朴素贝叶斯': ['GaussianNB', 'MultinomialNB', 'BernoulliNB'],
    }

    for category, models in categories.items():
        print(f"\n📦 {category}:")
        for model in models:
            if model in MODEL_REGISTRY:
                print(f"   ✓ {model}")
            else:
                print(f"   ✗ {model} (未注册)")


def main():
    args = parse_args()

    # 如果只是列出模型
    if args.list_models:
        list_available_models()
        return

    # 检查模型是否存在
    if args.model not in MODEL_REGISTRY:
        print(f"❌ 错误: 未知模型 '{args.model}'")
        print(f"\n请使用 --list-models 查看所有可用模型")
        return

    print("=" * 60)
    print(f"🔍 参数搜索配置")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"搜索类型: {args.search_type.upper()}")
    print(f"交叉验证: {args.cv} 折")
    if args.search_type == 'random':
        print(f"迭代次数: {args.n_iter}")
    print("=" * 60)

    # 加载数据
    print("\n⏳ 加载数据中...")

    # 创建一个简单的配置对象（用于数据路径）
    class SimpleConfig:
        def __init__(self):
            from pathlib import Path
            from omegaconf import OmegaConf

            config_path = ROOT / "configs" / "ml_benchmark.yaml"
            if config_path.exists():
                self._cfg = OmegaConf.load(config_path)
            else:
                self._cfg = OmegaConf.create({"data": {"path": "dataset/sample_binary.csv"}})

        def get(self, key, default=None):
            return self._cfg.get(key, default)

    cfg = SimpleConfig()

    data_cfg = get_data_config(cfg._cfg)
    feature_cols = list(data_cfg.get("numeric_cols", [])) + list(data_cfg.get("categorical_cols", []))

    loader = MLDataLoader(cfg)
    X_train, y_train, X_test, y_test = loader.prepare_data(
        feature_cols=feature_cols,
        label_col=str(data_cfg.label_col),
        train_ratio=float(data_cfg.get("split", {}).get("train_ratio", 0.8)),
        random_state=int(data_cfg.get("split", {}).get("random_state", 42)),
        cat_features=list(data_cfg.get("categorical_cols", [])),
        num_features=list(data_cfg.get("numeric_cols", [])),
    )

    # 创建模型
    model_wrapper = MODEL_REGISTRY[args.model]()

    # 参数搜索
    searcher = ParamSearcher(
        model_wrapper,
        search_type=args.search_type,
        cv=args.cv,
        n_iter=args.n_iter,
        n_jobs=args.n_jobs,
        verbose=1
    )

    best_params, best_score, best_model = searcher.search(X_train, y_train)

    # 打印最终结果
    print("\n" + "=" * 60)
    print("🎉 搜索完成！")
    print("=" * 60)
    print(f"🏆 最佳参数:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
    print(f"\n📈 最佳分数: {best_score:.4f}")
    print("=" * 60)

    # 保存最佳模型（可选）
    save_choice = input("\n是否保存最佳模型？(y/n): ").strip().lower()
    if save_choice == 'y':
        import joblib
        from pathlib import Path

        save_dir = Path("ml_results")
        save_dir.mkdir(exist_ok=True)

        model_path = save_dir / f"{args.model}_best_model.joblib"
        joblib.dump(best_model, model_path)
        print(f"✅ 最佳模型已保存到: {model_path}")


if __name__ == '__main__':
    main()
