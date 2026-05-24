"""PathPlanningPlugin — 路徑規劃插件主類

提供 A*、RRT、RRT* 路徑規劃及多智能體協同規劃功能。
"""

from __future__ import annotations

from typing import Any

from nexus.core.types import PluginMetadata, Task, TaskResult
from nexus.plugin.base import NexusPlugin
from nexus.tool.registry import tool

from nexus_plugin_path_planning.algorithms.astar import AStar, PathResult
from nexus_plugin_path_planning.algorithms.multi_agent import (
    AgentConfig,
    MultiAgentPlanner,
)
from nexus_plugin_path_planning.algorithms.rrt import RRT
from nexus_plugin_path_planning.algorithms.rrt_star import RRTStar
from nexus_plugin_path_planning.config import PathPlanningConfig
from nexus_plugin_path_planning.map.grid_map import GridMap
from nexus_plugin_path_planning.map.obstacle import Obstacle, ObstacleManager
from nexus_plugin_path_planning.utils.smoothing import (
    bezier_curve,
    cubic_spline_interpolate,
    gradient_descent_smooth,
)
from nexus_plugin_path_planning.utils.visualization import (
    compare_paths,
    visualize_multi_agent,
    visualize_path,
)


class PathPlanningPlugin(NexusPlugin):
    """路徑規劃插件

    提供多種路徑規劃演算法和地圖管理功能。

    Capabilities:
        - 柵格地圖管理（障礙物添加/移除/膨脹）
        - A* 最短路徑規劃
        - RRT 快速探索隨機樹
        - RRT* 漸近最優路徑規劃
        - 多智能體協同規劃
        - 路徑平滑（貝塞爾曲線/梯度下降/樣條插值）
        - 路徑可視化（base64 PNG）
    """

    def __init__(self) -> None:
        self._config: PathPlanningConfig | None = None
        self._grid_map: GridMap | None = None
        self._obstacle_manager: ObstacleManager | None = None
        self._astar: AStar | None = None
        self._rrt: RRT | None = None
        self._rrt_star: RRTStar | None = None
        self._multi_agent_planner: MultiAgentPlanner | None = None

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="path_planning",
            version="0.1.0",
            description="路徑規劃插件 — A*, RRT/RRT*, 多智能體協同規劃",
            author="404NotFound-ai",
            capabilities=[
                "grid_map_management",
                "astar_path_planning",
                "rrt_path_planning",
                "rrt_star_path_planning",
                "multi_agent_planning",
                "path_smoothing",
                "path_visualization",
            ],
            config_schema=PathPlanningConfig.model_json_schema(),
        )

    @property
    def config(self) -> PathPlanningConfig:
        if self._config is None:
            return PathPlanningConfig()
        return self._config

    @property
    def grid_map(self) -> GridMap:
        if self._grid_map is None:
            self._grid_map = GridMap(
                width=self.config.map_width,
                height=self.config.map_height,
                resolution=self.config.map_resolution,
            )
        return self._grid_map

    @property
    def obstacle_manager(self) -> ObstacleManager:
        if self._obstacle_manager is None:
            self._obstacle_manager = ObstacleManager(self.grid_map)
        return self._obstacle_manager

    @property
    def astar(self) -> AStar:
        if self._astar is None:
            self._astar = AStar(self.grid_map)
        return self._astar

    @property
    def rrt(self) -> RRT:
        if self._rrt is None:
            self._rrt = RRT(
                self.grid_map,
                step_size=self.config.rrt_step_size,
                max_iterations=self.config.rrt_max_iterations,
            )
        return self._rrt

    @property
    def rrt_star(self) -> RRTStar:
        if self._rrt_star is None:
            self._rrt_star = RRTStar(
                self.grid_map,
                step_size=self.config.rrt_star_step_size,
                max_iterations=self.config.rrt_star_max_iterations,
                search_radius=self.config.rrt_star_search_radius,
            )
        return self._rrt_star

    @property
    def multi_agent_planner(self) -> MultiAgentPlanner:
        if self._multi_agent_planner is None:
            self._multi_agent_planner = MultiAgentPlanner(self.grid_map)
        return self._multi_agent_planner

    async def initialize(self, config: dict[str, Any] | None = None) -> None:
        """初始化插件

        Args:
            config: 插件配置字典
        """
        if config:
            self._config = PathPlanningConfig.from_dict(config)
        else:
            self._config = PathPlanningConfig()

        # 初始化各模組
        self._grid_map = GridMap(
            width=self._config.map_width,
            height=self._config.map_height,
            resolution=self._config.map_resolution,
        )
        self._obstacle_manager = ObstacleManager(self._grid_map)
        self._astar = AStar(self._grid_map)
        self._rrt = RRT(
            self._grid_map,
            step_size=self._config.rrt_step_size,
            max_iterations=self._config.rrt_max_iterations,
        )
        self._rrt_star = RRTStar(
            self._grid_map,
            step_size=self._config.rrt_star_step_size,
            max_iterations=self._config.rrt_star_max_iterations,
            search_radius=self._config.rrt_star_search_radius,
        )
        self._multi_agent_planner = MultiAgentPlanner(self._grid_map)

    async def shutdown(self) -> None:
        """關閉插件，釋放資源"""
        self._config = None
        self._grid_map = None
        self._obstacle_manager = None
        self._astar = None
        self._rrt = None
        self._rrt_star = None
        self._multi_agent_planner = None

    async def execute(self, task: Task) -> TaskResult:
        """執行任務路由

        根據 task.input 中的 action 分發到對應的 @tool 方法。

        Args:
            task: 任務對象

        Returns:
            任務執行結果
        """
        action = task.input.get("action", "")
        params = task.input.get("params", {})

        action_map: dict[str, str] = {
            # 地圖管理
            "add_obstacle": "plan_add_obstacle",
            "remove_obstacle": "plan_remove_obstacle",
            "list_obstacles": "plan_list_obstacles",
            "clear_obstacles": "plan_clear_obstacles",
            "inflate_obstacles": "plan_inflate_obstacles",
            "get_map_info": "plan_get_map_info",
            # 路徑規劃
            "astar_search": "plan_astar_search",
            "rrt_plan": "plan_rrt_plan",
            "rrt_star_plan": "plan_rrt_star_plan",
            "multi_agent_plan": "plan_multi_agent_plan",
            # 路徑平滑
            "smooth_path": "plan_smooth_path",
            # 可視化
            "visualize_path": "plan_visualize_path",
            "visualize_multi_agent": "plan_visualize_multi_agent",
            "compare_algorithms": "plan_compare_algorithms",
        }

        tool_name = action_map.get(action)
        if tool_name is None:
            return TaskResult(
                success=False,
                error=f"未知操作: {action}",
                output={},
            )

        # 查找對應的 tool 方法
        method = getattr(self, f"_tool_{tool_name}", None)
        if method is None:
            return TaskResult(
                success=False,
                error=f"未實現的操作: {action}",
                output={},
            )

        try:
            result = await method(**params)
            return TaskResult(success=True, output=result)
        except Exception as e:
            return TaskResult(
                success=False,
                error=f"執行 {action} 失敗: {e!s}",
                output={},
            )

    # ============================================================
    # @tool 方法 — 地圖管理
    # ============================================================

    @tool(name="plan_add_obstacle", description="添加障礙物到地圖")
    async def _tool_plan_add_obstacle(
        self,
        shape: str = "rectangle",
        params: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """添加障礙物

        Args:
            shape: 形狀 rectangle/circle/polygon
            params: 形狀參數
            metadata: 可選元數據

        Returns:
            障礙物信息
        """
        obstacle = self.obstacle_manager.add(
            shape=shape,
            params=params or {},
            metadata=metadata,
        )
        return {
            "obstacle_id": obstacle.id,
            "shape": obstacle.shape,
            "cells_count": len(obstacle.cells),
        }

    @tool(name="plan_remove_obstacle", description="從地圖移除障礙物")
    async def _tool_plan_remove_obstacle(
        self, obstacle_id: str
    ) -> dict[str, Any]:
        """移除障礙物

        Args:
            obstacle_id: 障礙物 ID

        Returns:
            操作結果
        """
        success = self.obstacle_manager.remove(obstacle_id)
        return {"success": success, "obstacle_id": obstacle_id}

    @tool(name="plan_list_obstacles", description="列出地圖中所有障礙物")
    async def _tool_plan_list_obstacles(self) -> dict[str, Any]:
        """列出所有障礙物"""
        obstacles = self.obstacle_manager.list()
        return {
            "count": len(obstacles),
            "obstacles": [
                {
                    "id": o.id,
                    "shape": o.shape,
                    "params": o.params,
                    "cells_count": len(o.cells),
                    "metadata": o.metadata,
                }
                for o in obstacles
            ],
        }

    @tool(name="plan_clear_obstacles", description="清除地圖中所有障礙物")
    async def _tool_plan_clear_obstacles(self) -> dict[str, Any]:
        """清除所有障礙物"""
        count = len(self.obstacle_manager.list())
        self.obstacle_manager.clear()
        return {"cleared": count}

    @tool(name="plan_inflate_obstacles", description="膨脹地圖中的障礙物")
    async def _tool_plan_inflate_obstacles(
        self, radius: int = 1
    ) -> dict[str, Any]:
        """膨脹障礙物

        Args:
            radius: 膨脹半徑（格數）

        Returns:
            操作結果
        """
        self.grid_map.inflate_obstacles(radius)
        return {
            "radius": radius,
            "obstacle_count": len(self.grid_map.get_obstacles()),
        }

    @tool(name="plan_get_map_info", description="獲取地圖信息")
    async def _tool_plan_get_map_info(self) -> dict[str, Any]:
        """獲取地圖信息"""
        return self.grid_map.to_dict()

    # ============================================================
    # @tool 方法 — 路徑規劃
    # ============================================================

    @tool(name="plan_astar_search", description="使用 A* 演算法搜尋最短路徑")
    async def _tool_plan_astar_search(
        self,
        start_x: int = 0,
        start_y: int = 0,
        goal_x: int = 50,
        goal_y: int = 50,
    ) -> dict[str, Any]:
        """A* 路徑搜尋

        Args:
            start_x: 起點 X
            start_y: 起點 Y
            goal_x: 終點 X
            goal_y: 終點 Y

        Returns:
            路徑規劃結果
        """
        result = self.astar.search(
            start=(start_x, start_y),
            goal=(goal_x, goal_y),
        )
        return result.to_dict()

    @tool(name="plan_rrt_plan", description="使用 RRT 演算法規劃路徑")
    async def _tool_plan_rrt_plan(
        self,
        start_x: float = 0.0,
        start_y: float = 0.0,
        goal_x: float = 50.0,
        goal_y: float = 50.0,
    ) -> dict[str, Any]:
        """RRT 路徑規劃

        Args:
            start_x: 起點 X
            start_y: 起點 Y
            goal_x: 終點 X
            goal_y: 終點 Y

        Returns:
            路徑規劃結果
        """
        result = self.rrt.plan(
            start=(start_x, start_y),
            goal=(goal_x, goal_y),
        )
        return result.to_dict()

    @tool(name="plan_rrt_star_plan", description="使用 RRT* 演算法規劃路徑")
    async def _tool_plan_rrt_star_plan(
        self,
        start_x: float = 0.0,
        start_y: float = 0.0,
        goal_x: float = 50.0,
        goal_y: float = 50.0,
    ) -> dict[str, Any]:
        """RRT* 路徑規劃

        Args:
            start_x: 起點 X
            start_y: 起點 Y
            goal_x: 終點 X
            goal_y: 終點 Y

        Returns:
            路徑規劃結果
        """
        result = self.rrt_star.plan(
            start=(start_x, start_y),
            goal=(goal_x, goal_y),
        )
        return result.to_dict()

    @tool(
        name="plan_multi_agent_plan",
        description="多智能體協同路徑規劃",
    )
    async def _tool_plan_multi_agent_plan(
        self, agents: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """多智能體協同規劃

        Args:
            agents: 智能體配置列表
                [{"id": str, "start": [x,y], "goal": [x,y], "priority": int}]

        Returns:
            多智能體規劃結果
        """
        if not agents:
            return {"success": False, "error": "未提供智能體配置"}

        agent_configs = [
            AgentConfig(
                id=a.get("id", f"agent_{i}"),
                start=tuple(a.get("start", [0, 0])),
                goal=tuple(a.get("goal", [50, 50])),
                radius=a.get("radius", 1.0),
                priority=a.get("priority", 0),
            )
            for i, a in enumerate(agents)
        ]

        result = self.multi_agent_planner.plan(agent_configs)
        return result.to_dict()

    # ============================================================
    # @tool 方法 — 路徑平滑
    # ============================================================

    @tool(name="plan_smooth_path", description="平滑路徑")
    async def _tool_plan_smooth_path(
        self,
        path: list[list[float]] | None = None,
        method: str = "bezier",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """平滑路徑

        Args:
            path: 路徑點列表 [[x1,y1], [x2,y2], ...]
            method: 平滑方法 bezier/gradient_descent/cubic_spline
            **kwargs: 平滑參數

        Returns:
            平滑後的路徑
        """
        if not path or len(path) < 2:
            return {"success": False, "error": "路徑點不足"}

        path_tuples = [(p[0], p[1]) for p in path]

        if method == "bezier":
            num_points = kwargs.get("num_points", 100)
            smoothed = bezier_curve(path_tuples, num_points=num_points)
        elif method == "gradient_descent":
            alpha = kwargs.get("alpha", 0.5)
            iterations = kwargs.get("iterations", 100)
            smoothed = gradient_descent_smooth(
                path_tuples,
                alpha=alpha,
                iterations=iterations,
            )
        elif method == "cubic_spline":
            num_points = kwargs.get("num_points")
            smoothed = cubic_spline_interpolate(
                path_tuples,
                num_points=num_points,
            )
        else:
            return {"success": False, "error": f"未知平滑方法: {method}"}

        return {
            "success": True,
            "method": method,
            "original_points": len(path),
            "smoothed_points": len(smoothed),
            "path": [[float(x), float(y)] for x, y in smoothed],
        }

    # ============================================================
    # @tool 方法 — 可視化
    # ============================================================

    @tool(name="plan_visualize_path", description="可視化路徑規劃結果")
    async def _tool_plan_visualize_path(
        self,
        path: list[list[int]] | None = None,
        start: list[int] | None = None,
        goal: list[int] | None = None,
        title: str = "路徑規劃結果",
    ) -> dict[str, Any]:
        """可視化路徑

        Args:
            path: 路徑點列表
            start: 起點 [x, y]
            goal: 終點 [x, y]
            title: 圖表標題

        Returns:
            base64 編碼的 PNG 圖像
        """
        path_tuples = [(p[0], p[1]) for p in path] if path else None
        start_tuple = (start[0], start[1]) if start else None
        goal_tuple = (goal[0], goal[1]) if goal else None

        image_b64 = visualize_path(
            grid_map=self.grid_map,
            path=path_tuples,
            start=start_tuple,
            goal=goal_tuple,
            title=title,
            dpi=self.config.visualization_dpi,
            colors=self.config.visualization_colors,
        )
        return {"image_base64": image_b64, "format": "png"}

    @tool(
        name="plan_visualize_multi_agent",
        description="可視化多智能體路徑規劃結果",
    )
    async def _tool_plan_visualize_multi_agent(
        self,
        paths: dict[str, list[list[int]]] | None = None,
        title: str = "多智能體路徑規劃",
    ) -> dict[str, Any]:
        """可視化多智能體路徑

        Args:
            paths: 各智能體路徑 {agent_id: [[x,y], ...]}
            title: 圖表標題

        Returns:
            base64 編碼的 PNG 圖像
        """
        if not paths:
            return {"success": False, "error": "未提供路徑數據"}

        path_dict = {
            aid: [(p[0], p[1]) for p in pts]
            for aid, pts in paths.items()
        }

        image_b64 = visualize_multi_agent(
            grid_map=self.grid_map,
            paths=path_dict,
            title=title,
            dpi=self.config.visualization_dpi,
        )
        return {"image_base64": image_b64, "format": "png"}

    @tool(
        name="plan_compare_algorithms",
        description="對比不同演算法的路徑規劃結果",
    )
    async def _tool_plan_compare_algorithms(
        self,
        start_x: int = 0,
        start_y: int = 0,
        goal_x: int = 50,
        goal_y: int = 50,
    ) -> dict[str, Any]:
        """對比 A*、RRT、RRT* 演算法

        Args:
            start_x: 起點 X
            start_y: 起點 Y
            goal_x: 終點 X
            goal_y: 終點 Y

        Returns:
            各演算法結果及可視化圖像
        """
        start = (start_x, start_y)
        goal = (goal_x, goal_y)

        # A*
        astar_result = self.astar.search(start, goal)

        # RRT
        rrt_result = self.rrt.plan(
            (float(start_x), float(start_y)),
            (float(goal_x), float(goal_y)),
        )

        # RRT*
        rrt_star_result = self.rrt_star.plan(
            (float(start_x), float(start_y)),
            (float(goal_x), float(goal_y)),
        )

        # 可視化對比
        paths_dict: dict[str, list[tuple[int, int]]] = {}
        if astar_result.success:
            paths_dict["A*"] = astar_result.path
        if rrt_result.success:
            paths_dict["RRT"] = rrt_result.path
        if rrt_star_result.success:
            paths_dict["RRT*"] = rrt_star_result.path

        image_b64 = compare_paths(
            grid_map=self.grid_map,
            paths=paths_dict,
            start=start,
            goal=goal,
            title="A* vs RRT vs RRT* 路徑規劃對比",
            dpi=self.config.visualization_dpi,
        )

        return {
            "astar": astar_result.to_dict(),
            "rrt": rrt_result.to_dict(),
            "rrt_star": rrt_star_result.to_dict(),
            "comparison_image": image_b64,
        }
