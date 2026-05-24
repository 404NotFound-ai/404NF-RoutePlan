"""A* 路徑規劃演算法

實現經典 A* 搜尋演算法，用於在柵格地圖上尋找最短路徑。
支援 Manhattan、Euclidean、Chebyshev 啟發式函數。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from nexus_plugin_path_planning.map.grid_map import GridMap


@dataclass
class PathResult:
    """路徑規劃結果"""

    path: list[tuple[int, int]] = field(default_factory=list)
    """路徑點列表（從起點到終點）"""

    cost: float = 0.0
    """路徑總代價"""

    visited: int = 0
    """訪問節點數"""

    success: bool = False
    """是否找到路徑"""

    iterations: int = 0
    """迭代次數"""

    def to_dict(self) -> dict:
        """轉為字典"""
        return {
            "path": self.path,
            "cost": self.cost,
            "visited": self.visited,
            "success": self.success,
            "iterations": self.iterations,
        }


@dataclass
class _Node:
    """A* 搜尋節點（內部使用）"""

    x: int
    y: int
    g: float = 0.0  # 起點到當前節點代價
    h: float = 0.0  # 啟發式估計代價
    f: float = 0.0  # 總代價 f = g + h
    parent: _Node | None = None

    def __lt__(self, other: _Node) -> bool:
        return self.f < other.f

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Node):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def to_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)


# 啟發式函數類型
HeuristicFunc = Callable[[int, int, int, int], float]


def manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """曼哈頓距離"""
    return float(abs(x1 - x2) + abs(y1 - y2))


def euclidean_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """歐幾里得距離"""
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


def chebyshev_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """切比雪夫距離"""
    return float(max(abs(x1 - x2), abs(y1 - y2)))


# 預設鄰居偏移（8 方向）
DEFAULT_NEIGHBORS: list[tuple[int, int, float]] = [
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (1, 1, 1.414),
    (1, -1, 1.414),
    (-1, 1, 1.414),
    (-1, -1, 1.414),
]


class AStar:
    """A* 路徑規劃器

    在柵格地圖上使用 A* 演算法搜尋最短路徑。

    Examples:
        >>> grid = GridMap(10, 10)
        >>> astar = AStar(grid)
        >>> result = astar.search((0, 0), (9, 9))
        >>> result.success
        True
    """

    def __init__(
        self,
        grid_map: GridMap,
        heuristic: HeuristicFunc | None = None,
        allow_diagonal: bool = True,
    ) -> None:
        """
        Args:
            grid_map: 柵格地圖
            heuristic: 啟發式函數，預設使用歐幾里得距離
            allow_diagonal: 是否允許對角線移動
        """
        self._grid_map: GridMap = grid_map
        self._heuristic: HeuristicFunc = heuristic or euclidean_distance
        self._neighbors: list[tuple[int, int, float]] = (
            DEFAULT_NEIGHBORS if allow_diagonal else DEFAULT_NEIGHBORS[:4]
        )

    @property
    def grid_map(self) -> GridMap:
        return self._grid_map

    def search(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        max_iterations: int = 100000,
    ) -> PathResult:
        """執行 A* 搜尋

        Args:
            start: 起點座標 (x, y)
            goal: 終點座標 (x, y)
            max_iterations: 最大迭代次數

        Returns:
            路徑規劃結果
        """
        result = PathResult()

        # 邊界檢查
        if not self._is_valid(start) or not self._is_valid(goal):
            result.success = False
            return result

        if start == goal:
            result.path = [start]
            result.cost = 0.0
            result.success = True
            return result

        start_node = _Node(x=start[0], y=start[1])
        goal_node = _Node(x=goal[0], y=goal[1])

        # 初始化啟發式
        start_node.h = self._heuristic(start_node.x, start_node.y, goal_node.x, goal_node.y)
        start_node.f = start_node.g + start_node.h

        open_set: list[_Node] = [start_node]
        closed_set: set[tuple[int, int]] = set()

        iterations = 0

        while open_set and iterations < max_iterations:
            current = heapq.heappop(open_set)
            iterations += 1

            if (current.x, current.y) in closed_set:
                continue

            # 到達終點
            if current.x == goal_node.x and current.y == goal_node.y:
                result.path = self._reconstruct_path(current)
                result.cost = current.g
                result.visited = len(closed_set)
                result.success = True
                result.iterations = iterations
                return result

            closed_set.add((current.x, current.y))

            # 擴展鄰居
            for dx, dy, cost in self._neighbors:
                nx, ny = current.x + dx, current.y + dy

                if not self._is_valid((nx, ny)):
                    continue
                if (nx, ny) in closed_set:
                    continue

                neighbor = _Node(x=nx, y=ny)
                neighbor.g = current.g + cost
                neighbor.h = self._heuristic(nx, ny, goal_node.x, goal_node.y)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current

                heapq.heappush(open_set, neighbor)

        # 未找到路徑
        result.visited = len(closed_set)
        result.iterations = iterations
        result.success = False
        return result

    def _is_valid(self, pos: tuple[int, int]) -> bool:
        """檢查位置是否可通行"""
        x, y = pos
        return not self._grid_map.is_occupied(x, y)

    @staticmethod
    def _reconstruct_path(node: _Node) -> list[tuple[int, int]]:
        """重建路徑"""
        path: list[tuple[int, int]] = []
        current: _Node | None = node
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        path.reverse()
        return path
