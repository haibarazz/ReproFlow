"""
ML Benchmark 数据加载和预处理模块

独立于主框架的数据处理流程，专门为传统机器学习模型设计
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from hydra.utils import to_absolute_path
from typing import Tuple, Dict, List, Optional
import os


class MLDataLoader:
    """ML Benchmark 数据加载器"""

    def __init__(self, cfg):
        """
        初始化数据加载器

        Args:
            cfg: Hydra 配置对象，支持 data.path 或 legacy data_dir 字段
        """
        self.cfg = cfg
        self.encoders = {}
        self.scalers = {}

    def load_data(self) -> pd.DataFrame:
        """
        加载原始数据

        Returns:
            pd.DataFrame: 原始数据集
        """
        data_path_cfg = self.cfg.get("data", {}).get("path") if hasattr(self.cfg, "get") else None
        legacy_path = getattr(self.cfg, "data_dir", "dataset/sample_binary.csv")
        data_path = to_absolute_path(str(data_path_cfg or legacy_path))

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        print(f"📂 读取数据: {data_path}")
        df = pd.read_csv(data_path)
        print(f"✅ 数据加载完成，共 {len(df)} 条记录")
        return df

    def split_data(
        self,
        df: pd.DataFrame,
        label_col: str,
        train_ratio: float = 0.8,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        划分训练集和测试集（分层采样）

        Args:
            df: 原始数据集
            label_col: 标签列名
            train_ratio: 训练集比例
            random_state: 随机种子

        Returns:
            (train_df, test_df): 训练集和测试集
        """
        print(f"\n✂️  划分训练集和测试集 (比例 {train_ratio:.0%}:{1-train_ratio:.0%})")

        task_type = str(self.cfg.get("task_type", "classification")) if hasattr(self.cfg, "get") else "classification"
        stratify = df[label_col]
        if task_type == "regression" or df[label_col].value_counts(dropna=False).min() < 2:
            stratify = None

        train_df, test_df = train_test_split(
            df,
            test_size=1 - train_ratio,
            random_state=random_state,
            stratify=stratify,
        )

        print(f"   训练集: {len(train_df)} 条")
        print(f"   测试集: {len(test_df)} 条")

        return train_df, test_df

    def preprocess_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cat_features: List[str],
        num_features: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        预处理特征：分类特征编码 + 数值特征标准化

        Args:
            train_df: 训练集
            test_df: 测试集
            cat_features: 分类特征列名列表
            num_features: 数值特征列名列表

        Returns:
            (train_processed, test_processed): 处理后的训练集和测试集
        """
        print(f"\n🔧 特征预处理中...")

        # 创建副本避免修改原数据
        train_processed = train_df.copy()
        test_processed = test_df.copy()

        # 1. 分类特征：标签编码
        if cat_features:
            print(f"   📊 编码 {len(cat_features)} 个分类特征...")
            for col in cat_features:
                if col in train_processed.columns:
                    # 转换为字符串类型避免数值列被误编码
                    train_processed[col] = train_processed[col].astype(str)
                    test_processed[col] = test_processed[col].astype(str)

                    le = LabelEncoder()
                    train_processed[col] = le.fit_transform(train_processed[col])
                    test_processed[col] = le.transform(test_processed[col])

                    self.encoders[col] = le

        # 2. 数值特征：标准化 + 缺失值填充
        if num_features:
            print(f"   📈 标准化 {len(num_features)} 个数值特征...")
            for col in num_features:
                if col in train_processed.columns:
                    # 用训练集中位数填充缺失值
                    median_value = train_processed[col].median()
                    train_processed[col] = train_processed[col].fillna(median_value)
                    test_processed[col] = test_processed[col].fillna(median_value)

                    # StandardScaler 标准化
                    scaler = StandardScaler()
                    train_processed[col] = scaler.fit_transform(
                        train_processed[[col]]
                    ).flatten()
                    test_processed[col] = scaler.transform(
                        test_processed[[col]]
                    ).flatten()

                    self.scalers[col] = scaler

        print("✅ 特征预处理完成")
        return train_processed, test_processed

    def prepare_data(
        self,
        feature_cols: List[str],
        label_col: str,
        train_ratio: float = 0.8,
        random_state: int = 42,
        cat_features: Optional[List[str]] = None,
        num_features: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        完整数据准备流程：加载 -> 划分 -> 预处理 -> 返回特征矩阵

        Args:
            feature_cols: 所有特征列名列表
            label_col: 标签列名
            train_ratio: 训练集比例
            random_state: 随机种子
            cat_features: 分类特征列名列表（如果为 None，自动从 feature_cols 推断）
            num_features: 数值特征列名列表（如果为 None，自动从 feature_cols 推断）

        Returns:
            (X_train, y_train, X_test, y_test): 训练集和测试集的特征与标签
        """
        # 1. 加载数据
        df = self.load_data()

        # 2. 划分数据集
        train_df, test_df = self.split_data(
            df, label_col, train_ratio, random_state
        )

        # 3. 自动推断特征类型（如果未指定）
        if cat_features is None or num_features is None:
            cat_features = []
            num_features = []
            for col in feature_cols:
                if col in train_df.columns:
                    # 判断是否为数值类型
                    if pd.api.types.is_numeric_dtype(train_df[col]):
                        num_features.append(col)
                    else:
                        cat_features.append(col)

        # 4. 预处理特征
        train_df, test_df = self.preprocess_features(
            train_df, test_df, cat_features, num_features
        )

        # 5. 提取特征矩阵和标签向量
        X_train = train_df[feature_cols].values
        y_train = train_df[label_col].values.ravel()
        X_test = test_df[feature_cols].values
        y_test = test_df[label_col].values.ravel()

        print(f"\n📦 最终数据形状:")
        print(f"   X_train: {X_train.shape}")
        print(f"   y_train: {y_train.shape}")
        print(f"   X_test: {X_test.shape}")
        print(f"   y_test: {y_test.shape}")

        return X_train, y_train, X_test, y_test


def load_ml_data(cfg, feature_cols, label_col, **kwargs):
    """
    便捷函数：一键加载 ML Benchmark 数据

    Args:
        cfg: Hydra 配置对象
        feature_cols: 特征列名列表
        label_col: 标签列名
        **kwargs: 传递给 MLDataLoader.prepare_data 的其他参数

    Returns:
        (X_train, y_train, X_test, y_test)
    """
    loader = MLDataLoader(cfg)
    return loader.prepare_data(feature_cols, label_col, **kwargs)
