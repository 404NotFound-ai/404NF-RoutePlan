"""RRT* (Rapidly-exploring Random Tree Star) 路徑規劃演算法

RRT* 是 RRT 的改進版本，增加了：
1. 近鄰節點搜索（Near）：在插入新節點時搜索半徑內的近鄰
2. 選擇最優父節點（ChooseParent）：為新節點選擇代價最小的父節點
3. 重新佈線（Rewire）：優化近鄰節點的連接，保證漸近最優性
"""

from __future__ import annotations

import math
import random

from nexus_plugin_path_planning.algorithms.astar import PathResult
from nexus_plugin_path_planning.algorithms.rrt import RRT, RRTNode
from nexus_plugin_path_planning.map.grid_map import GridMap


class RRTStar(RRT):
    """RRT* 路徑規劃器

    在 RRT 基礎上增加漸近最優性保證。

    Examples:
        >>> grid = GridMap(100, 100)
        >>> rrt_star = RRTStar(grid)
        >>> result = rrt_star.plan((0, 0), (90, 90))
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
        search_radius: float = 15.0,
    ) -> None:
        """
        Args:
            grid_map: 柵格地圖
            step_size: 擴展步長
            max_iterations: 最大迭代次數
            goal_sample_rate: 直接採樣目標點的機率
            goal_tolerance: 到達目標的容忍距離
            search_radius: 近鄰搜索半徑
        """
        super().__init__(
            grid_map=grid_map,
            step_size=step_size,
            max_iterations=max_iterations,
            goal_sample_rate=goal_sample_rate,
            goal_tolerance=goal_tolerance,
        )
        self._search_radius: float = search_radius

    def plan(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> PathResult:
        """執行 RRT* 路徑規劃

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

            # RRT* 核心：近鄰搜索與最優父節點選擇
            near_nodes = self._near(new_node)
            best_parent = self._choose_parent(new_node, near_nodes)

            if best_parent is not None:
                new_node.parent = best_parent
                new_node.cost = best_parent.cost + best_parent.distance_to(new_node)
            else:
                new_node.parent = nearest
                new_node.cost = nearest.cost + nearest.distance_to(new_node)

            self._nodes.append(new_node)

            # RRT* 核心：重新佈線
            self._rewire(new_node, near_nodes)

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

    def _near(self, node: RRTNode) -> list[RRTNode]:
        """搜索節點半徑內的所有近鄰節點

        Args:
            node: 目標節點

        Returns:
            近鄰節點列表
        """
        return [n for n in self._nodes if n.distance_to(node) <= self._search_radius]

    def _choose_parent(
        self, new_node: RRTNode, near_nodes: list[RRTNode]
    ) -> RRTNode | None:
        """為新節點選擇代價最小的父節點

        Args:
            new_node: 新節點
            near_nodes: 近鄰節點列表

        Returns:
            最優父節點，若無則返回 None
        """
        best_parent: RRTNode | None = None
        min_cost: float = float("inf")

        for near_node in near_nodes:
            # 檢查連接是否無碰撞
            if not self._collision_free(near_node, new_node):
                continue

            potential_cost = near_node.cost + near_node.distance_to(new_node)
            if potential_cost < min_cost:
                min_cost = potential_cost
                best_parent = near_node

        return best_parent

    def _rewire(self, new_node: RRTNode, near_nodes: list[RRTNode]) -> None:
        """重新佈線：檢查是否可以通過新節點降低近鄰節點的代價

        Args:
            new_node: 新插入的節點
            near_nodes: 近鄰節點列表
        """
        for near_node in near_nodes:
            # 跳過自身
            if near_node is new_node:
                continue

            # 檢查通過新節點是否代價更小
            potential_cost = new_node.cost + new_node.distance_to(near_node)

            if potential_cost < near_node.cost:
                # 檢查連接是否無碰撞
                if self._collision_free(new_node, near_node):
                    near_node.parent = new_node
                    near_node.cost = potential_cost
