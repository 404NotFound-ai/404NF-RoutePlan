"""A* 路徑規劃演算法測試"""

from __future__ import annotations

import pytest

from nexus_plugin_path_planning.algorithms.astar import (
    AStar,
    chebyshev_distance,
    euclidean_distance,
    manhattan_distance,
)
from nexus_plugin_path_planning.map.grid_map import GridMap


class TestHeuristics:
    """啟發式函數測試"""

    def test_manhattan_distance(self) -> None:
        assert manhattan_distance(0, 0, 3, 4) == 7.0

    def test_euclidean_distance(self) -> None:
        assert euclidean_distance(0, 0, 3, 4) == 5.0

    def test_chebyshev_distance(self) -> None:
        assert chebyshev_distance(0, 0, 3, 4) == 4.0


class TestAStar:
    """A* 演算法測試"""

    def test_simple_path(self, grid_map: GridMap) -> None:
        """測試簡單路徑"""
        astar = AStar(grid_map)
        result = astar.search((0, 0), (5, 5))
        assert result.success
        assert len(result.path) >= 2
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (5, 5)
        assert result.cost > 0

    def test_same_start_goal(self, grid_map: GridMap) -> None:
        """測試起點等於終點"""
        astar = AStar(grid_map)
        result = astar.search((3, 3), (3, 3))
        assert result.success
        assert result.path == [(3, 3)]
        assert result.cost == 0.0

    def test_blocked_path(self, grid_map_with_obstacles: GridMap) -> None:
        """測試被阻擋的路徑"""
        astar = AStar(grid_map_with_obstacles)
        # 起點在牆左側，終點在牆右側，需要繞行
        result = astar.search((0, 15), (49, 15))
        assert result.success
        # 路徑不應穿過障礙物
        for x, y in result.path:
            assert not grid_map_with_obstacles.is_occupied(x, y)

    def test_no_path(self, grid_map: GridMap) -> None:
        """測試無路徑情況"""
        # 用障礙物包圍起點
        for x in range(3):
            for y in range(3):
                if (x, y) != (1, 1):
                    grid_map.set_occupied(x, y)
        astar = AStar(grid_map)
        result = astar.search((1, 1), (10, 10))
        assert not result.success

    def test_no_diagonal(self, grid_map: GridMap) -> None:
        """測試不允許對角線移動"""
        astar = AStar(grid_map, allow_diagonal=False)
        result = astar.search((0, 0), (2, 2))
        assert result.success
        # 不允許對角線時，路徑應為 (0,0)->(0,1)->(0,2)->(1,2)->(2,2) 等
        for i in range(len(result.path) - 1):
            dx = abs(result.path[i][0] - result.path[i + 1][0])
            dy = abs(result.path[i][1] - result.path[i + 1][1])
            assert dx + dy == 1  # 只能上下左右移動

    def test_custom_heuristic(self, grid_map: GridMap) -> None:
        """測試自定義啟發式函數"""
        astar = AStar(grid_map, heuristic=manhattan_distance)
        result = astar.search((0, 0), (10, 10))
        assert result.success

    def test_path_optimality(self, grid_map: GridMap) -> None:
        """測試路徑最優性"""
        astar = AStar(grid_map)
        result = astar.search((0, 0), (4, 0))
        assert result.success
        # 在無障礙物時，最短路徑應為直線
        expected_cost = 4.0  # 4 步水平移動
        assert abs(result.cost - expected_cost) < 0.01

    def test_visited_nodes(self, grid_map: GridMap) -> None:
        """測試訪問節點計數"""
        astar = AStar(grid_map)
        result = astar.search((0, 0), (3, 3))
        assert result.visited > 0
        assert result.iterations > 0
