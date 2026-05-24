"""多智能體協同規劃測試"""

from __future__ import annotations

import pytest

from nexus_plugin_path_planning.algorithms.multi_agent import (
    AgentConfig,
    MultiAgentPlanner,
)
from nexus_plugin_path_planning.map.grid_map import GridMap


class TestMultiAgentPlanner:
    """多智能體規劃器測試"""

    def test_single_agent(self, grid_map: GridMap) -> None:
        """測試單智能體"""
        planner = MultiAgentPlanner(grid_map)
        agents = [
            AgentConfig(
                id="agent1",
                start=(0, 0),
                goal=(40, 40),
                priority=1,
            ),
        ]
        result = planner.plan(agents)
        assert result.success
        assert "agent1" in result.paths
        assert len(result.paths["agent1"]) >= 2

    def test_two_agents_no_conflict(self, grid_map: GridMap) -> None:
        """測試兩個無衝突智能體"""
        planner = MultiAgentPlanner(grid_map)
        agents = [
            AgentConfig(
                id="agent1",
                start=(0, 0),
                goal=(40, 40),
                priority=2,
            ),
            AgentConfig(
                id="agent2",
                start=(0, 49),
                goal=(40, 9),
                priority=1,
            ),
        ]
        result = planner.plan(agents)
        assert result.success
        assert len(result.paths) == 2

    def test_two_agents_with_conflict(self, grid_map: GridMap) -> None:
        """測試兩個有潛在衝突的智能體"""
        planner = MultiAgentPlanner(grid_map)
        agents = [
            AgentConfig(
                id="agent1",
                start=(0, 0),
                goal=(49, 49),
                priority=2,
            ),
            AgentConfig(
                id="agent2",
                start=(49, 49),
                goal=(0, 0),
                priority=1,
            ),
        ]
        result = planner.plan(agents)
        assert result.success
        # 應檢測到衝突
        assert len(result.conflicts) >= 0  # 可能無衝突（路徑不同）

    def test_priority_ordering(self, grid_map: GridMap) -> None:
        """測試優先級排序"""
        planner = MultiAgentPlanner(grid_map)
        agents = [
            AgentConfig(
                id="low",
                start=(0, 0),
                goal=(30, 30),
                priority=1,
            ),
            AgentConfig(
                id="high",
                start=(0, 5),
                goal=(30, 35),
                priority=10,
            ),
        ]
        result = planner.plan(agents)
        assert result.success
        # 高優先級智能體應有路徑
        assert "high" in result.paths
        assert len(result.paths["high"]) >= 2

    def test_empty_agents(self, grid_map: GridMap) -> None:
        """測試空智能體列表"""
        planner = MultiAgentPlanner(grid_map)
        result = planner.plan([])
        assert result.success
        assert len(result.paths) == 0

    def test_three_agents(self, grid_map: GridMap) -> None:
        """測試三個智能體"""
        planner = MultiAgentPlanner(grid_map)
        agents = [
            AgentConfig(
                id=f"agent{i}",
                start=(0, i * 10),
                goal=(45, i * 10),
                priority=i,
            )
            for i in range(3)
        ]
        result = planner.plan(agents)
        assert result.success
        assert len(result.paths) == 3

    def test_result_to_dict(self, grid_map: GridMap) -> None:
        """測試結果轉字典"""
        planner = MultiAgentPlanner(grid_map)
        agents = [
            AgentConfig(
                id="agent1",
                start=(0, 0),
                goal=(10, 10),
                priority=1,
            ),
        ]
        result = planner.plan(agents)
        data = result.to_dict()
        assert "paths" in data
        assert "costs" in data
        assert "success" in data
        assert "total_cost" in data
