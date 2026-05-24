"""RRT 路徑規劃演算法測試"""

from __future__ import annotations

import pytest

from nexus_plugin_path_planning.algorithms.rrt import RRT
from nexus_plugin_path_planning.map.grid_map import GridMap


class TestRRT:
    """RRT 演算法測試"""

    def test_simple_path(self, grid_map: GridMap) -> None:
        """測試簡單路徑"""
        rrt = RRT(grid_map, step_size=5.0, max_iterations=2000)
        result = rrt.plan((0, 0), (45, 45))
        assert result.success
        assert len(result.path) >= 2
        assert result.path[0] == (0, 0)
        assert result.path[-1] == (45, 45)

    def test_no_collision(self, grid_map_with_obstacles: GridMap) -> None:
        """測試路徑不與障礙物碰撞"""
        rrt = RRT(grid_map_with_obstacles, step_size=5.0, max_iterations=3000)
        result = rrt.plan((0, 15), (49, 15))
        if result.success:
            for x, y in result.path:
                assert not grid_map_with_obstacles.is_occupied(x, y)

    def test_same_start_goal(self, grid_map: GridMap) -> None:
        """測試起點等於終點"""
        rrt = RRT(grid_map)
        result = rrt.plan((5, 5), (5, 5))
        assert result.success
        assert result.path == [(5, 5)]

    def test_nodes_generated(self, grid_map: GridMap) -> None:
        """測試節點生成"""
        rrt = RRT(grid_map, step_size=5.0, max_iterations=500)
        rrt.plan((0, 0), (40, 40))
        assert len(rrt.nodes) > 0

    def test_step_size_effect(self, grid_map: GridMap) -> None:
        """測試步長影響"""
        rrt_small = RRT(grid_map, step_size=2.0, max_iterations=2000)
        rrt_large = RRT(grid_map, step_size=10.0, max_iterations=2000)

        result_small = rrt_small.plan((0, 0), (40, 40))
        result_large = rrt_large.plan((0, 0), (40, 40))

        if result_small.success and result_large.success:
            # 大步長通常路徑點更少
            assert len(result_small.path) >= len(result_large.path)

    def test_goal_sample_rate(self, grid_map: GridMap) -> None:
        """測試目標採樣率"""
        rrt = RRT(grid_map, goal_sample_rate=0.3, max_iterations=1000)
        result = rrt.plan((0, 0), (30, 30))
        # 高目標採樣率應能更快找到路徑
        assert result.success or result.iterations < 1000

    def test_goal_tolerance(self, grid_map: GridMap) -> None:
        """測試目標容忍距離"""
        rrt = RRT(grid_map, goal_tolerance=5.0, max_iterations=1000)
        result = rrt.plan((0, 0), (45, 45))
        if result.success:
            # 終點應在容忍距離內
            last = result.path[-1]
            dist = ((last[0] - 45) ** 2 + (last[1] - 45) ** 2) ** 0.5
            assert dist <= 5.0
