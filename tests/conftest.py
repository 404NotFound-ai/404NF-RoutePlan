"""測試共用 fixture"""

from __future__ import annotations

from typing import Any

import pytest

from nexus_plugin_path_planning.algorithms.astar import AStar
from nexus_plugin_path_planning.algorithms.multi_agent import (
    AgentConfig,
    MultiAgentPlanner,
)
from nexus_plugin_path_planning.algorithms.rrt import RRT
from nexus_plugin_path_planning.algorithms.rrt_star import RRTStar
from nexus_plugin_path_planning.config import PathPlanningConfig
from nexus_plugin_path_planning.map.grid_map import GridMap
from nexus_plugin_path_planning.map.obstacle import ObstacleManager
from nexus_plugin_path_planning.plugin import PathPlanningPlugin


@pytest.fixture
def grid_map() -> GridMap:
    """創建測試用柵格地圖"""
    return GridMap(width=50, height=50, resolution=1.0)


@pytest.fixture
def grid_map_with_obstacles() -> GridMap:
    """創建帶有障礙物的測試用地圖"""
    gm = GridMap(width=50, height=50, resolution=1.0)
    # 添加一面牆
    for y in range(10, 30):
        gm.set_occupied(25, y)
    return gm


@pytest.fixture
def obstacle_manager(grid_map: GridMap) -> ObstacleManager:
    """創建測試用障礙物管理器"""
    return ObstacleManager(grid_map)


@pytest.fixture
def astar(grid_map: GridMap) -> AStar:
    """創建測試用 A* 規劃器"""
    return AStar(grid_map)


@pytest.fixture
def rrt(grid_map: GridMap) -> RRT:
    """創建測試用 RRT 規劃器"""
    return RRT(grid_map, step_size=5.0, max_iterations=1000)


@pytest.fixture
def rrt_star(grid_map: GridMap) -> RRTStar:
    """創建測試用 RRT* 規劃器"""
    return RRTStar(
        grid_map,
        step_size=5.0,
        max_iterations=1000,
        search_radius=15.0,
    )


@pytest.fixture
def multi_agent_planner(grid_map: GridMap) -> MultiAgentPlanner:
    """創建測試用多智能體規劃器"""
    return MultiAgentPlanner(grid_map)


@pytest.fixture
def plugin() -> PathPlanningPlugin:
    """創建測試用插件實例"""
    return PathPlanningPlugin()


@pytest.fixture
def config() -> PathPlanningConfig:
    """創建測試用配置"""
    return PathPlanningConfig(
        map_width=50,
        map_height=50,
        rrt_max_iterations=1000,
        rrt_star_max_iterations=1000,
    )
