"""工具函數 — 使用 @tool 裝飾器暴露給 LLM 調用

提供路徑規劃相關的工具函數別名，方便 LLM 直接調用。
"""

from __future__ import annotations

from nexus.tool.registry import tool


@tool(
    name="plan_add_obstacle",
    description="添加障礙物到地圖，支援矩形(rectangle)、圓形(circle)、多邊形(polygon)",
)
async def tool_plan_add_obstacle(
    self,
    shape: str = "rectangle",
    params: dict | None = None,
    metadata: dict | None = None,
) -> dict:
    """添加障礙物

    Args:
        shape: 障礙物形狀 (rectangle/circle/polygon)
        params: 形狀參數
            rectangle: {"x": 0, "y": 0, "width": 10, "height": 10}
            circle: {"cx": 50, "cy": 50, "radius": 10}
            polygon: {"points": [[0,0], [10,0], [10,10], [0,10]]}
        metadata: 可選元數據

    Returns:
        障礙物信息
    """
    return await self._tool_plan_add_obstacle(
        shape=shape, params=params, metadata=metadata
    )


@tool(
    name="plan_remove_obstacle",
    description="從地圖中移除指定障礙物",
)
async def tool_plan_remove_obstacle(self, obstacle_id: str) -> dict:
    """移除障礙物

    Args:
        obstacle_id: 障礙物 ID

    Returns:
        操作結果
    """
    return await self._tool_plan_remove_obstacle(obstacle_id=obstacle_id)


@tool(
    name="plan_list_obstacles",
    description="列出地圖中所有障礙物",
)
async def tool_plan_list_obstacles(self) -> dict:
    """列出所有障礙物"""
    return await self._tool_plan_list_obstacles()


@tool(
    name="plan_clear_obstacles",
    description="清除地圖中所有障礙物",
)
async def tool_plan_clear_obstacles(self) -> dict:
    """清除所有障礙物"""
    return await self._tool_plan_clear_obstacles()


@tool(
    name="plan_inflate_obstacles",
    description="膨脹地圖中的障礙物（用於機器人半徑補償）",
)
async def tool_plan_inflate_obstacles(self, radius: int = 1) -> dict:
    """膨脹障礙物

    Args:
        radius: 膨脹半徑（格數）

    Returns:
        操作結果
    """
    return await self._tool_plan_inflate_obstacles(radius=radius)


@tool(
    name="plan_get_map_info",
    description="獲取當前地圖信息（寬度、高度、障礙物數量等）",
)
async def tool_plan_get_map_info(self) -> dict:
    """獲取地圖信息"""
    return await self._tool_plan_get_map_info()


@tool(
    name="plan_astar_search",
    description="使用 A* 演算法搜尋最短路徑，返回路徑點列表和代價",
)
async def tool_plan_astar_search(
    self,
    start_x: int = 0,
    start_y: int = 0,
    goal_x: int = 50,
    goal_y: int = 50,
) -> dict:
    """A* 路徑搜尋

    Args:
        start_x: 起點 X 座標
        start_y: 起點 Y 座標
        goal_x: 終點 X 座標
        goal_y: 終點 Y 座標

    Returns:
        路徑規劃結果
    """
    return await self._tool_plan_astar_search(
        start_x=start_x,
        start_y=start_y,
        goal_x=goal_x,
        goal_y=goal_y,
    )


@tool(
    name="plan_rrt_plan",
    description="使用 RRT 演算法規劃路徑，適合高維空間或非完整約束機器人",
)
async def tool_plan_rrt_plan(
    self,
    start_x: float = 0.0,
    start_y: float = 0.0,
    goal_x: float = 50.0,
    goal_y: float = 50.0,
) -> dict:
    """RRT 路徑規劃

    Args:
        start_x: 起點 X 座標
        start_y: 起點 Y 座標
        goal_x: 終點 X 座標
        goal_y: 終點 Y 座標

    Returns:
        路徑規劃結果
    """
    return await self._tool_plan_rrt_plan(
        start_x=start_x,
        start_y=start_y,
        goal_x=goal_x,
        goal_y=goal_y,
    )


@tool(
    name="plan_rrt_star_plan",
    description="使用 RRT* 演算法規劃路徑，具有漸近最優性保證",
)
async def tool_plan_rrt_star_plan(
    self,
    start_x: float = 0.0,
    start_y: float = 0.0,
    goal_x: float = 50.0,
    goal_y: float = 50.0,
) -> dict:
    """RRT* 路徑規劃

    Args:
        start_x: 起點 X 座標
        start_y: 起點 Y 座標
        goal_x: 終點 X 座標
        goal_y: 終點 Y 座標

    Returns:
        路徑規劃結果
    """
    return await self._tool_plan_rrt_star_plan(
        start_x=start_x,
        start_y=start_y,
        goal_x=goal_x,
        goal_y=goal_y,
    )


@tool(
    name="plan_multi_agent_plan",
    description="多智能體協同路徑規劃，支援衝突檢測與解決",
)
async def tool_plan_multi_agent_plan(
    self, agents: list[dict] | None = None
) -> dict:
    """多智能體協同規劃

    Args:
        agents: 智能體配置列表
            [{"id": "agent1", "start": [0,0], "goal": [90,90], "priority": 2}]

    Returns:
        多智能體規劃結果
    """
    return await self._tool_plan_multi_agent_plan(agents=agents)


@tool(
    name="plan_smooth_path",
    description="平滑路徑，支援貝塞爾曲線、梯度下降、三次樣條三種方法",
)
async def tool_plan_smooth_path(
    self,
    path: list[list[float]] | None = None,
    method: str = "bezier",
    **kwargs,
) -> dict:
    """平滑路徑

    Args:
        path: 路徑點列表 [[x1,y1], [x2,y2], ...]
        method: 平滑方法 (bezier/gradient_descent/cubic_spline)
        **kwargs: 平滑參數

    Returns:
        平滑後的路徑
    """
    return await self._tool_plan_smooth_path(
        path=path, method=method, **kwargs
    )


@tool(
    name="plan_visualize_path",
    description="可視化路徑規劃結果，返回 base64 編碼的 PNG 圖像",
)
async def tool_plan_visualize_path(
    self,
    path: list[list[int]] | None = None,
    start: list[int] | None = None,
    goal: list[int] | None = None,
    title: str = "路徑規劃結果",
) -> dict:
    """可視化路徑

    Args:
        path: 路徑點列表
        start: 起點 [x, y]
        goal: 終點 [x, y]
        title: 圖表標題

    Returns:
        base64 編碼的 PNG 圖像
    """
    return await self._tool_plan_visualize_path(
        path=path, start=start, goal=goal, title=title
    )


@tool(
    name="plan_visualize_multi_agent",
    description="可視化多智能體路徑規劃結果，返回 base64 編碼的 PNG 圖像",
)
async def tool_plan_visualize_multi_agent(
    self,
    paths: dict[str, list[list[int]]] | None = None,
    title: str = "多智能體路徑規劃",
) -> dict:
    """可視化多智能體路徑

    Args:
        paths: 各智能體路徑 {agent_id: [[x,y], ...]}
        title: 圖表標題

    Returns:
        base64 編碼的 PNG 圖像
    """
    return await self._tool_plan_visualize_multi_agent(
        paths=paths, title=title
    )


@tool(
    name="plan_compare_algorithms",
    description="對比 A*、RRT、RRT* 三種演算法的路徑規劃結果",
)
async def tool_plan_compare_algorithms(
    self,
    start_x: int = 0,
    start_y: int = 0,
    goal_x: int = 50,
    goal_y: int = 50,
) -> dict:
    """對比演算法

    Args:
        start_x: 起點 X
        start_y: 起點 Y
        goal_x: 終點 X
        goal_y: 終點 Y

    Returns:
        各演算法結果及對比圖像
    """
    return await self._tool_plan_compare_algorithms(
        start_x=start_x,
        start_y=start_y,
        goal_x=goal_x,
        goal_y=goal_y,
    )
