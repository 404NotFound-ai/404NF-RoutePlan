"""多智能體協同路徑規劃

實現基於優先級規劃（Priority-Based Planning）的多智能體路徑協調。
支援衝突檢測、優先級分配、路徑重規劃。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from nexus_plugin_path_planning.algorithms.astar import AStar, PathResult
from nexus_plugin_path_planning.map.grid_map import GridMap


@dataclass
class AgentConfig:
    """智能體配置"""

    id: str
    """智能體唯一標識"""

    start: tuple[int, int]
    """起點座標"""

    goal: tuple[int, int]
    """終點座標"""

    radius: float = 1.0
    """智能體半徑"""

    priority: int = 0
    """優先級（數字越大優先級越高）"""


@dataclass
class Conflict:
    """路徑衝突"""

    time: int
    """衝突發生的時間步"""

    agent_a: str
    """智能體 A ID"""

    agent_b: str
    """智能體 B ID"""

    position: tuple[int, int]
    """衝突位置"""

    conflict_type: str = "vertex"
    """衝突類型: vertex / edge"""


@dataclass
class MultiAgentResult:
    """多智能體規劃結果"""

    paths: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    """各智能體的路徑 {agent_id: path}"""

    costs: dict[str, float] = field(default_factory=dict)
    """各智能體的路徑代價 {agent_id: cost}"""

    conflicts: list[Conflict] = field(default_factory=list)
    """檢測到的衝突列表"""

    success: bool = False
    """是否所有智能體都找到無衝突路徑"""

    total_cost: float = 0.0
    """總代價"""

    def to_dict(self) -> dict[str, Any]:
        """轉為字典"""
        return {
            "paths": self.paths,
            "costs": self.costs,
            "conflicts": [
                {
                    "time": c.time,
                    "agent_a": c.agent_a,
                    "agent_b": c.agent_b,
                    "position": c.position,
                    "conflict_type": c.conflict_type,
                }
                for c in self.conflicts
            ],
            "success": self.success,
            "total_cost": self.total_cost,
        }


class MultiAgentPlanner:
    """多智能體協同路徑規劃器

    使用基於優先級的規劃方法，為多個智能體規劃無衝突路徑。

    Examples:
        >>> grid = GridMap(20, 20)
        >>> planner = MultiAgentPlanner(grid)
        >>> agents = [
        ...     AgentConfig(id="agent1", start=(0, 0), goal=(19, 19), priority=2),
        ...     AgentConfig(id="agent2", start=(19, 0), goal=(0, 19), priority=1),
        ... ]
        >>> result = planner.plan(agents)
        >>> result.success
        True
    """

    def __init__(
        self,
        grid_map: GridMap,
        planner: AStar | None = None,
        max_iterations: int = 100,
    ) -> None:
        """
        Args:
            grid_map: 柵格地圖
            planner: 底層路徑規劃器，預設使用 AStar
            max_iterations: 最大重規劃迭代次數
        """
        self._grid_map: GridMap = grid_map
        self._planner: AStar = planner or AStar(grid_map)
        self._max_iterations: int = max_iterations

    @property
    def grid_map(self) -> GridMap:
        return self._grid_map

    def plan(self, agents: list[AgentConfig]) -> MultiAgentResult:
        """執行多智能體路徑規劃

        Args:
            agents: 智能體配置列表

        Returns:
            多智能體規劃結果
        """
        result = MultiAgentResult()

        if not agents:
            result.success = True
            return result

        # 按優先級排序（降序）
        sorted_agents = sorted(agents, key=lambda a: a.priority, reverse=True)

        # 為每個智能體規劃路徑
        for agent in sorted_agents:
            path_result = self._planner.search(agent.start, agent.goal)
            if path_result.success:
                result.paths[agent.id] = path_result.path
                result.costs[agent.id] = path_result.cost
            else:
                result.paths[agent.id] = []
                result.costs[agent.id] = float("inf")

        # 衝突檢測與解決
        for iteration in range(self._max_iterations):
            conflicts = self._detect_conflicts(result.paths)
            if not conflicts:
                break

            result.conflicts = conflicts
            self._resolve_conflicts(result, sorted_agents)

        # 檢查是否所有智能體都有路徑
        result.success = all(len(path) > 0 for path in result.paths.values())
        result.total_cost = sum(result.costs.values())

        return result

    def _detect_conflicts(
        self, paths: dict[str, list[tuple[int, int]]]
    ) -> list[Conflict]:
        """檢測路徑間的衝突

        Args:
            paths: 各智能體的路徑

        Returns:
            衝突列表
        """
        conflicts: list[Conflict] = []
        agent_ids = list(paths.keys())

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                aid_a = agent_ids[i]
                aid_b = agent_ids[j]
                path_a = paths[aid_a]
                path_b = paths[aid_b]

                max_len = max(len(path_a), len(path_b))

                for t in range(max_len):
                    # 獲取時間 t 的位置（若已到達終點則停留在最後位置）
                    pos_a = path_a[min(t, len(path_a) - 1)] if path_a else None
                    pos_b = path_b[min(t, len(path_b) - 1)] if path_b else None

                    if pos_a is None or pos_b is None:
                        continue

                    # 頂點衝突：同一時間佔用同一位置
                    if pos_a == pos_b:
                        conflicts.append(
                            Conflict(
                                time=t,
                                agent_a=aid_a,
                                agent_b=aid_b,
                                position=pos_a,
                                conflict_type="vertex",
                            )
                        )

                    # 邊衝突：互相交換位置
                    if t > 0:
                        prev_a = path_a[min(t - 1, len(path_a) - 1)] if path_a else None
                        prev_b = path_b[min(t - 1, len(path_b) - 1)] if path_b else None
                        if prev_a is not None and prev_b is not None:
                            if pos_a == prev_b and pos_b == prev_a:
                                conflicts.append(
                                    Conflict(
                                        time=t,
                                        agent_a=aid_a,
                                        agent_b=aid_b,
                                        position=pos_a,
                                        conflict_type="edge",
                                    )
                                )

        return conflicts

    def _resolve_conflicts(
        self,
        result: MultiAgentResult,
        sorted_agents: list[AgentConfig],
    ) -> None:
        """解決路徑衝突（降低優先級較低的智能體路徑）

        Args:
            result: 多智能體規劃結果
            sorted_agents: 按優先級排序的智能體列表
        """
        # 對低優先級智能體進行重規劃
        for agent in reversed(sorted_agents):
            if agent.id not in result.paths or not result.paths[agent.id]:
                continue

            # 將高優先級智能體的路徑視為臨時障礙物
            temp_grid = self._create_temp_grid(result, sorted_agents, agent.id)

            # 使用臨時地圖重新規劃
            temp_planner = AStar(temp_grid)
            path_result = temp_planner.search(agent.start, agent.goal)

            if path_result.success:
                result.paths[agent.id] = path_result.path
                result.costs[agent.id] = path_result.cost

    def _create_temp_grid(
        self,
        result: MultiAgentResult,
        sorted_agents: list[AgentConfig],
        current_agent_id: str,
    ) -> GridMap:
        """創建臨時地圖（將高優先級智能體路徑設為障礙物）

        Args:
            result: 多智能體規劃結果
            sorted_agents: 按優先級排序的智能體列表
            current_agent_id: 當前智能體 ID

        Returns:
            臨時柵格地圖
        """
        temp_grid = GridMap(
            width=self._grid_map.width,
            height=self._grid_map.height,
            resolution=self._grid_map.resolution,
        )

        # 複製原始障礙物
        for y in range(self._grid_map.height):
            for x in range(self._grid_map.width):
                if self._grid_map.is_occupied(x, y):
                    temp_grid.set_occupied(x, y)

        # 將高優先級智能體的路徑設為障礙物
        for agent in sorted_agents:
            if agent.id == current_agent_id:
                break  # 當前智能體之後的（更低優先級）不考慮
            if agent.id in result.paths:
                for x, y in result.paths[agent.id]:
                    temp_grid.set_occupied(x, y)

        return temp_grid
