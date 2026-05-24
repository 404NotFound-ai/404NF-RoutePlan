"""路徑平滑工具函數

提供貝塞爾曲線平滑、梯度下降平滑、三次樣條插值等功能。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline


def bezier_curve(
    control_points: list[tuple[float, float]],
    num_points: int = 100,
) -> list[tuple[float, float]]:
    """生成貝塞爾曲線

    使用 De Casteljau 演算法計算貝塞爾曲線上的點。

    Args:
        control_points: 控制點列表
        num_points: 生成的曲線點數量

    Returns:
        曲線上的點列表
    """
    if len(control_points) < 2:
        return control_points.copy()

    n = len(control_points) - 1
    curve: list[tuple[float, float]] = []

    for i in range(num_points):
        t = i / (num_points - 1)
        # De Casteljau 演算法
        points = list(control_points)
        while len(points) > 1:
            new_points: list[tuple[float, float]] = []
            for j in range(len(points) - 1):
                x = (1 - t) * points[j][0] + t * points[j + 1][0]
                y = (1 - t) * points[j][1] + t * points[j + 1][1]
                new_points.append((x, y))
            points = new_points
        curve.append(points[0])

    return curve


def gradient_descent_smooth(
    path: list[tuple[float, float]],
    alpha: float = 0.5,
    iterations: int = 100,
    weight_smooth: float = 0.1,
    weight_length: float = 0.3,
) -> list[tuple[float, float]]:
    """使用梯度下降法平滑路徑

    通過最小化路徑的曲率和長度來平滑路徑。

    Args:
        path: 原始路徑點列表
        alpha: 學習率
        iterations: 迭代次數
        weight_smooth: 平滑權重
        weight_length: 長度保持權重

    Returns:
        平滑後的路徑點列表
    """
    if len(path) < 3:
        return path.copy()

    new_path = np.array(path, dtype=float)
    n = len(new_path)

    for _ in range(iterations):
        # 計算平滑梯度（二階差分）
        smooth_grad = np.zeros_like(new_path)
        for i in range(1, n - 1):
            smooth_grad[i] = (
                new_path[i - 1] - 2 * new_path[i] + new_path[i + 1]
            )

        # 計算長度梯度（一階差分）
        length_grad = np.zeros_like(new_path)
        for i in range(1, n):
            diff = new_path[i] - new_path[i - 1]
            norm = np.linalg.norm(diff)
            if norm > 1e-6:
                length_grad[i - 1] -= diff / norm
                length_grad[i] += diff / norm

        # 更新路徑點（保持起點和終點不變）
        gradient = weight_smooth * smooth_grad + weight_length * length_grad
        new_path[1:-1] -= alpha * gradient[1:-1]

    return [(float(x), float(y)) for x, y in new_path]


def cubic_spline_interpolate(
    path: list[tuple[float, float]],
    num_points: int | None = None,
) -> list[tuple[float, float]]:
    """使用三次樣條插值平滑路徑

    Args:
        path: 原始路徑點列表
        num_points: 插值點數量（預設為路徑長度的 2 倍）

    Returns:
        插值後的路徑點列表
    """
    if len(path) < 3:
        return path.copy()

    path_arr = np.array(path, dtype=float)
    n = len(path_arr)

    # 計算累積弧長參數化
    t = np.zeros(n)
    for i in range(1, n):
        t[i] = t[i - 1] + np.linalg.norm(path_arr[i] - path_arr[i - 1])

    if t[-1] < 1e-6:
        return path.copy()

    t = t / t[-1]  # 歸一化到 [0, 1]

    if num_points is None:
        num_points = n * 2

    # 創建三次樣條
    try:
        cs_x = CubicSpline(t, path_arr[:, 0], bc_type="natural")
        cs_y = CubicSpline(t, path_arr[:, 1], bc_type="natural")

        t_new = np.linspace(0, 1, num_points)
        x_new = cs_x(t_new)
        y_new = cs_y(t_new)

        return [(float(x), float(y)) for x, y in zip(x_new, y_new)]
    except Exception:
        # 若樣條插值失敗，返回原始路徑
        return path.copy()
