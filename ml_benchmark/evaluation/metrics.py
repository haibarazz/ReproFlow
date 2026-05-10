"""
评估指标计算

提供统一的评估指标计算接口
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from typing import Dict, List, Optional, Union


class MetricsCalculator:
    """
    统一评估指标计算器

    支持分类和回归任务的多种评估指标
    """

    # 分类指标注册表
    CLASSIFICATION_METRICS = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'roc_auc': roc_auc_score,
    }

    # 回归指标注册表
    REGRESSION_METRICS = {
        'mse': mean_squared_error,
        'mae': mean_absolute_error,
        'r2': r2_score,
        'rmse': None,
    }

    # 所有指标
    METRIC_REGISTRY = {
        **CLASSIFICATION_METRICS,
        **REGRESSION_METRICS,
    }

    @classmethod
    def calculate(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        task_type: str = 'classification',
        metrics: Optional[List[str]] = None,
        average: str = 'binary'
    ) -> Dict[str, float]:
        """
        计算评估指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率（用于 roc_auc 等指标）
            task_type: 任务类型，'classification' 或 'regression'
            metrics: 要计算的指标列表，如果为 None 则计算所有默认指标
            average: 多分类时的平均方式（'binary', 'micro', 'macro', 'weighted'）

        Returns:
            指标名称到指标值的字典
        """
        # 确定默认指标
        if metrics is None:
            if task_type == 'classification':
                metrics = ['accuracy', 'precision', 'recall', 'f1']
            else:
                metrics = ['mse', 'mae', 'r2']

        results = {}

        for metric in metrics:
            if metric not in cls.METRIC_REGISTRY:
                print(f"⚠️ 未知指标: {metric}，跳过")
                continue

            try:
                if metric == 'roc_auc':
                    # ROC AUC 需要概率预测
                    if y_proba is not None:
                        # 二分类：取正类的概率
                        if y_proba.shape[1] == 2:
                            results[metric] = cls.METRIC_REGISTRY[metric](y_true, y_proba[:, 1])
                        else:
                            # 多分类：使用 macro 平均
                            results[metric] = cls.METRIC_REGISTRY[metric](y_true, y_proba, multi_class='ovr', average=average)
                    else:
                        results[metric] = None
                        print(f"⚠️ {metric} 需要概率预测，但未提供 y_proba")
                elif metric in ['precision', 'recall', 'f1']:
                    # 这些指标需要指定 average 参数
                    results[metric] = cls.METRIC_REGISTRY[metric](y_true, y_pred, average=average, zero_division=0)
                elif metric == 'rmse':
                    results[metric] = cls.calculate_rmse(y_true, y_pred)
                else:
                    # 其他指标直接计算
                    results[metric] = cls.METRIC_REGISTRY[metric](y_true, y_pred)
            except Exception as e:
                results[metric] = None
                print(f"⚠️ 计算 {metric} 时出错: {str(e)}")

        return results

    @classmethod
    def calculate_confusion_matrix(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        计算混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签

        Returns:
            混淆矩阵
        """
        return confusion_matrix(y_true, y_pred)

    @classmethod
    def calculate_rmse(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        计算 RMSE（均方根误差）

        Args:
            y_true: 真实标签
            y_pred: 预测标签

        Returns:
            RMSE 值
        """
        mse = mean_squared_error(y_true, y_pred)
        return np.sqrt(mse)
