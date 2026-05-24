"""幾何計算工具函數

提供距離計算、線段相交、點在多邊形內判斷、路徑長度計算、路徑插值等功能。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """計算兩點之間的歐幾里得距離

    Args:
        p1: 點1 (x, y)
        p2: 點2 (x, y)

    Returns:
        距離
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def path_length(path: list[tuple[float, float]]) -> float:
    """計算路徑總長度

    Args:
        path: 路徑點列表

    Returns:
        路徑總長度
    """
    if len(path) < 2:
        return 0.0

    total = 0.0
    for i in range(len(path) - 1):
        total += distance(path[i], path[i + 1])
    return total


def line_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> tuple[float, float] | None:
    """計算兩條線段的交點

    Args:
        p1: 線段1 起點
        p2: 線段1 終點
        p3: 線段2 起點
        p4: 線段2 終點

    Returns:
        交點座標 (x, y)，若不相交則返回 None
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # 平行或共線

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)

    return None


def point_in_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    """判斷點是否在多邊形內（射線法）

    Args:
        point: 點座標 (x, y)
        polygon: 多邊形頂點列表

    Returns:
        點是否在多邊形內
    """
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # 檢查射線是否與邊相交
        if (yi > y) != (yj > y):
            x_intersect = xi + (y - yi) * (xj - xi) / (yj - yi)
            if x < x_intersect:
                inside = not inside
        j = i

    return inside


def interpolate_path(
    path: list[tuple[float, float]],
    step_size: float = 1.0,
) -> list[tuple[float, float]]:
    """對路徑進行線性插值，使路徑點間距均勻

    Args:
        path: 原始路徑點列表
        step_size: 插值步長

    Returns:
        插值後的路徑點列表
    """
    if len(path) < 2:
        return path.copy()

    interpolated: list[tuple[float, float]] = [path[0]]

    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i + 1]
        seg_dist = distance(p1, p2)

        if seg_dist < 1e-6:
            continue

        num_steps = max(int(seg_dist / step_size), 1)
        for j in range(1, num_steps + 1):
            t = j / num_steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            interpolated.append((x, y))

    return interpolated
