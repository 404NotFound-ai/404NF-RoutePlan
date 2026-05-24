"""RRT (Rapidly-exploring Random Tree) 路徑規劃演算法

實現快速探索隨機樹演算法，用於在高維空間中尋找路徑。
特別適合具有非完整約束的機器人路徑規劃。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from nexus_plugin_path_planning.algorithms.astar import PathResult
from nexus_plugin_path_planning.map.grid_map import GridMap


@dataclass
class RRTNode:
    """RRT 樹節點"""

    x: float
    y: float
    parent: RRTNode | None = None
    cost: float = 0.0

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, other: RRTNode) -> float:
        """計算到另一個節點的距離"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance_to_point(self, point: tuple[float, float]) -> float:
        """計算到點的距離"""
        return math.sqrt((self.x - point[0]) ** 2 + (self.y - point[1]) ** 2)


class RRT:
    """RRT 路徑規劃器

    使用快速探索隨機樹演算法在柵格地圖上規劃路徑。

    Examples:
        >>> grid = GridMap(100, 100)
        >>> rrt = RRT(grid)
        >>> result = rrt.plan((0, 0), (90, 90))
        >>> result.success
        True
    """

    def __init__(
        self,
        grid_map: GridMap,
        step_size: float = 5.0,
        max_iterations: int = 5000,
        goal_sample_rate: float = 0.1,
        goal_tolerance: float = 3.0,
    ) -> None:
        """
        Args:
            grid_map: 柵格地圖
            step_size: 擴展步長
            max_iterations: 最大迭代次數
            goal_sample_rate: 直接採樣目標點的機率
            goal_tolerance: 到達目標的容忍距離
        """
        self._grid_map: GridMap = grid_map
        self._step_size: float = step_size
        self._max_iterations: int = max_iterations
        self._goal_sample_rate: float = goal_sample_rate
        self._goal_tolerance: float = goal_tolerance
        self._nodes: list[RRTNode] = []

    @property
    def grid_map(self) -> GridMap:
        return self._grid_map

    @property
    def nodes(self) -> list[RRTNode]:
        return self._nodes.copy()

    def plan(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> PathResult:
        """執行 RRT 路徑規劃

        Args:
            start: 起點座標 (x, y)
            goal: 終點座標 (x, y)

        Returns:
            路徑規劃結果
        """
        result = PathResult()
        self._nodes = []

        # 邊界檢查
        if not self._is_valid_point(start):
            return result

        start_node = RRTNode(x=start[0], y=start[1])
        goal_node = RRTNode(x=goal[0], y=goal[1])
        self._nodes.append(start_node)

        for iteration in range(self._max_iterations):
            # 以一定機率直接採樣目標點
            if random.random() < self._goal_sample_rate:
                sample = goal_node
            else:
                sample = self._random_sample()

            # 找到最近的節點
            nearest = self._nearest(sample)

            # 向採樣點擴展
            new_node = self._steer(nearest, sample)

            if new_node is None:
                continue

            # 檢查路徑是否碰撞
            if not self._collision_free(nearest, new_node):
                continue

            self._nodes.append(new_node)

            # 檢查是否到達目標
            if new_node.distance_to(goal_node) <= self._goal_tolerance:
                result.path = self._reconstruct_path(new_node)
                result.cost = new_node.cost
                result.visited = len(self._nodes)
                result.success = True
                result.iterations = iteration + 1
                return result

        # 未找到路徑
        result.visited = len(self._nodes)
        result.iterations = self._max_iterations
        result.success = False
        return result

    def _random_sample(self) -> RRTNode:
        """隨機採樣一個節點"""
        x = random.uniform(0, self._grid_map.width - 1)
        y = random.uniform(0, self._grid_map.height - 1)
        return RRTNode(x=x, y=y)

    def _nearest(self, node: RRTNode) -> RRTNode:
        """找到距離節點最近的樹中節點"""
        return min(self._nodes, key=lambda n: n.distance_to(node))

    def _steer(self, from_node: RRTNode, to_node: RRTNode) -> RRTNode | None:
        """從 from_node 向 to_node 方向擴展一步

        Args:
            from_node: 起始節點
            to_node: 目標節點

        Returns:
            新節點，若無效則返回 None
        """
        dist = from_node.distance_to(to_node)
        if dist < 1e-6:
            return None

        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        step = min(self._step_size, dist)

        new_x = from_node.x + step * math.cos(theta)
        new_y = from_node.y + step * math.sin(theta)

        # 邊界檢查
        if not (0 <= new_x < self._grid_map.width and 0 <= new_y < self._grid_map.height):
            return None

        new_node = RRTNode(x=new_x, y=new_y, parent=from_node)
        new_node.cost = from_node.cost + step
        return new_node

    def _collision_free(self, from_node: RRTNode, to_node: RRTNode) -> bool:
        """檢查兩節點之間的路徑是否無碰撞

        Args:
            from_node: 起始節點
            to_node: 目標節點

        Returns:
            是否無碰撞
        """
        # 離散化檢查路徑上的點
        dist = from_node.distance_to(to_node)
        steps = max(int(dist / 1.0), 1)

        for i in range(steps + 1):
            t = i / steps
            x = int(from_node.x + t * (to_node.x - from_node.x))
            y = int(from_node.y + t * (to_node.y - from_node.y))
            if self._grid_map.is_occupied(x, y):
                return False
        return True

    def _is_valid_point(self, point: tuple[float, float]) -> bool:
        """檢查點是否在地圖範圍內且可通行"""
        x, y = int(point[0]), int(point[1])
        if not (0 <= x < self._grid_map.width and 0 <= y < self._grid_map.height):
            return False
        return not self._grid_map.is_occupied(x, y)

    @staticmethod
    def _reconstruct_path(node: RRTNode) -> list[tuple[int, int]]:
        """重建路徑"""
        path: list[tuple[int, int]] = []
        current: RRTNode | None = node
        while current is not None:
            path.append((int(round(current.x)), int(round(current.y))))
            current = current.parent
        path.reverse()
        return path
