"""PathPlanningPlugin 測試"""

from __future__ import annotations

from typing import Any

import pytest

from nexus_plugin_path_planning.plugin import PathPlanningPlugin


class TestPathPlanningPlugin:
    """插件基本功能測試"""

    @pytest.mark.asyncio
    async def test_initialize(self, plugin: PathPlanningPlugin) -> None:
        """測試初始化"""
        await plugin.initialize()
        assert plugin._config is not None
        assert plugin._grid_map is not None
        assert plugin._obstacle_manager is not None
        assert plugin._astar is not None
        assert plugin._rrt is not None
        assert plugin._rrt_star is not None
        assert plugin._multi_agent_planner is not None

    @pytest.mark.asyncio
    async def test_initialize_with_config(self, plugin: PathPlanningPlugin) -> None:
        """測試帶配置初始化"""
        config = {
            "map_width": 80,
            "map_height": 60,
            "rrt_max_iterations": 3000,
        }
        await plugin.initialize(config)
        assert plugin.config.map_width == 80
        assert plugin.config.map_height == 60
        assert plugin.config.rrt_max_iterations == 3000

    @pytest.mark.asyncio
    async def test_shutdown(self, plugin: PathPlanningPlugin) -> None:
        """測試關閉"""
        await plugin.initialize()
        await plugin.shutdown()
        assert plugin._config is None
        assert plugin._grid_map is None

    @pytest.mark.asyncio
    async def test_metadata(self, plugin: PathPlanningPlugin) -> None:
        """測試元數據"""
        metadata = plugin.metadata
        assert metadata.name == "path_planning"
        assert metadata.version == "0.1.0"
        assert "astar_path_planning" in metadata.capabilities
        assert "rrt_path_planning" in metadata.capabilities
        assert "multi_agent_planning" in metadata.capabilities

    @pytest.mark.asyncio
    async def test_add_obstacle(self, plugin: PathPlanningPlugin) -> None:
        """測試添加障礙物"""
        await plugin.initialize()
        result = await plugin._tool_plan_add_obstacle(
            shape="rectangle",
            params={"x": 10, "y": 10, "width": 5, "height": 5},
        )
        assert "obstacle_id" in result
        assert result["cells_count"] == 25

    @pytest.mark.asyncio
    async def test_list_obstacles(self, plugin: PathPlanningPlugin) -> None:
        """測試列出障礙物"""
        await plugin.initialize()
        await plugin._tool_plan_add_obstacle(
            shape="rectangle",
            params={"x": 0, "y": 0, "width": 3, "height": 3},
        )
        result = await plugin._tool_plan_list_obstacles()
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_clear_obstacles(self, plugin: PathPlanningPlugin) -> None:
        """測試清除障礙物"""
        await plugin.initialize()
        await plugin._tool_plan_add_obstacle(
            shape="rectangle",
            params={"x": 0, "y": 0, "width": 3, "height": 3},
        )
        result = await plugin._tool_plan_clear_obstacles()
        assert result["cleared"] == 1

    @pytest.mark.asyncio
    async def test_get_map_info(self, plugin: PathPlanningPlugin) -> None:
        """測試獲取地圖信息"""
        await plugin.initialize()
        result = await plugin._tool_plan_get_map_info()
        assert result["width"] == 100
        assert result["height"] == 100
        assert result["obstacle_count"] == 0

    @pytest.mark.asyncio
    async def test_astar_search(self, plugin: PathPlanningPlugin) -> None:
        """測試 A* 搜尋"""
        await plugin.initialize()
        result = await plugin._tool_plan_astar_search(
            start_x=0, start_y=0, goal_x=10, goal_y=10
        )
        assert result["success"]
        assert len(result["path"]) >= 2

    @pytest.mark.asyncio
    async def test_rrt_plan(self, plugin: PathPlanningPlugin) -> None:
        """測試 RRT 規劃"""
        await plugin.initialize()
        result = await plugin._tool_plan_rrt_plan(
            start_x=0.0, start_y=0.0, goal_x=30.0, goal_y=30.0
        )
        assert result["success"]

    @pytest.mark.asyncio
    async def test_rrt_star_plan(self, plugin: PathPlanningPlugin) -> None:
        """測試 RRT* 規劃"""
        await plugin.initialize()
        result = await plugin._tool_plan_rrt_star_plan(
            start_x=0.0, start_y=0.0, goal_x=30.0, goal_y=30.0
        )
        assert result["success"]

    @pytest.mark.asyncio
    async def test_multi_agent_plan(self, plugin: PathPlanningPlugin) -> None:
        """測試多智能體規劃"""
        await plugin.initialize()
        agents = [
            {"id": "a1", "start": [0, 0], "goal": [40, 40], "priority": 2},
            {"id": "a2", "start": [40, 0], "goal": [0, 40], "priority": 1},
        ]
        result = await plugin._tool_plan_multi_agent_plan(agents=agents)
        assert result["success"]

    @pytest.mark.asyncio
    async def test_execute_astar(self, plugin: PathPlanningPlugin) -> None:
        """測試 execute 路由 A*"""
        from nexus.core.types import Task

        await plugin.initialize()
        task = Task(
            id="test-1",
            input={
                "action": "astar_search",
                "params": {"start_x": 0, "start_y": 0, "goal_x": 5, "goal_y": 5},
            },
        )
        result = await plugin.execute(task)
        assert result.success
        assert result.output["success"]

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, plugin: PathPlanningPlugin) -> None:
        """測試未知操作"""
        from nexus.core.types import Task

        await plugin.initialize()
        task = Task(
            id="test-unknown",
            input={"action": "unknown_action", "params": {}},
        )
        result = await plugin.execute(task)
        assert not result.success
        assert "未知操作" in (result.error or "")

    @pytest.mark.asyncio
    async def test_smooth_path_bezier(self, plugin: PathPlanningPlugin) -> None:
        """測試貝塞爾曲線平滑"""
        await plugin.initialize()
        path = [[0.0, 0.0], [5.0, 10.0], [10.0, 0.0]]
        result = await plugin._tool_plan_smooth_path(
            path=path, method="bezier", num_points=50
        )
        assert result["success"]
        assert result["method"] == "bezier"
        assert result["smoothed_points"] == 50

    @pytest.mark.asyncio
    async def test_compare_algorithms(self, plugin: PathPlanningPlugin) -> None:
        """測試演算法對比"""
        await plugin.initialize()
        result = await plugin._tool_plan_compare_algorithms(
            start_x=0, start_y=0, goal_x=30, goal_y=30
        )
        assert "astar" in result
        assert "rrt" in result
        assert "rrt_star" in result
        assert "comparison_image" in result
