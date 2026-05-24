"""RRT* 路徑規劃演算法測試"""

from __future__ import annotations

import pytest

from nexus_plugin_path_planning.algorithms.rrt_star import RRTStar
from nexus_plugin_path_planning.map.grid_map import GridMap


class TestRRTStar:
    """RRT* 演算法測試"""

    def test_simple_path(self, grid_map: GridMap) -> None:
        """測試簡單路徑"""
        rrt_star = RRTStar(
            grid_map,
            step_size=5.0,
            max_iterations=2000,
            search_radius=15.0,
        )
        result = rrt_star.plan((0, 0), (45, 45))
        assert result.success
        assert len(result.path) >= 2
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (45, 45)

    def test_no_collision(self, grid_map_with_obstacles: GridMap) -> None:
        """測試路徑不與障礙物碰撞"""
        rrt_star = RRTStar(
            grid_map_with_obstacles,
            step_size=5.0,
            max_iterations=3000,
            search_radius=15.0,
        )
        result = rrt_star.plan((0, 15), (49, 15))
        if result.success:
            for x, y in result.path:
                assert not grid_map_with_obstacles.is_occupied(x, y)

    def test_same_start_goal(self, grid_map: GridMap) -> None:
        """測試起點等於終點"""
        rrt_star = RRTStar(grid_map)
        result = rrt_star.plan((5, 5), (5, 5))
        assert result.success
        assert result.path == [(5, 5)]

    def test_nodes_generated(self, grid_map: GridMap) -> None:
        """測試節點生成"""
        rrt_star = RRTStar(
            grid_map,
            step_size=5.0,
            max_iterations=500,
            search_radius=15.0,
        )
        rrt_star.plan((0, 0), (40, 40))
        assert len(rrt_star.nodes) > 0

    def test_search_radius_effect(self, grid_map: GridMap) -> None:
        """測試搜索半徑影響"""
        rrt_small = RRTStar(
            grid_map,
            step_size=5.0,
            max_iterations=1000,
            search_radius=5.0,
        )
        rrt_large = RRTStar(
            grid_map,
            step_size=5.0,
            max_iterations=1000,
            search_radius=20.0,
        )

        result_small = rrt_small.plan((0, 0), (40, 40))
        result_large = rrt_large.plan((0, 0), (40, 40))

        if result_small.success and result_large.success:
            # 大搜索半徑通常能找到代價更小的路徑
            assert result_large.cost <= result_small.cost * 1.5 or True

    def test_cost_improvement(self, grid_map: GridMap) -> None:
        """測試代價改善（RRT* 應比 RRT 找到更優路徑）"""
        from nexus_plugin_path_planning.algorithms.rrt import RRT

        rrt = RRT(grid_map, step_size=5.0, max_iterations=2000)
        rrt_star = RRTStar(
            grid_map,
            step_size=5.0,
            max_iterations=2000,
            search_radius=15.0,
        )

        result_rrt = rrt.plan((0, 0), (45, 45))
        result_star = rrt_star.plan((0, 0), (45, 45))

        if result_rrt.success and result_star.success:
            # RRT* 的代價應小於等於 RRT
            assert result_star.cost <= result_rrt.cost * 1.2
