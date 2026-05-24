"""障礙物管理 — Obstacle, ObstacleManager"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from nexus_plugin_path_planning.map.grid_map import GridMap


@dataclass
class Obstacle:
    """障礙物定義"""

    id: str
    shape: str  # rectangle/circle/polygon
    params: dict[str, float]
    cells: list[tuple[int, int]]
    metadata: dict[str, Any] = field(default_factory=dict)


class ObstacleManager:
    """障礙物管理器

    管理地圖中的障礙物，支持添加、移除、列出和清除。
    """

    def __init__(self, grid_map: GridMap) -> None:
        self._grid_map: GridMap = grid_map
        self._obstacles: dict[str, Obstacle] = {}

    def add(
        self,
        shape: str,
        params: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> Obstacle:
        """添加障礙物

        Args:
            shape: 障礙物形狀 rectangle/circle/polygon
            params: 形狀參數
            metadata: 可選元數據

        Returns:
            創建的 Obstacle 對象
        """
        cells = self._grid_map.add_obstacle(shape, params)
        obstacle = Obstacle(
            id=str(uuid.uuid4())[:8],
            shape=shape,
            params=params,
            cells=cells,
            metadata=metadata or {},
        )
        self._obstacles[obstacle.id] = obstacle
        return obstacle

    def remove(self, obstacle_id: str) -> bool:
        """移除障礙物

        Args:
            obstacle_id: 障礙物 ID

        Returns:
            是否成功移除
        """
        if obstacle_id not in self._obstacles:
            return False

        obstacle = self._obstacles[obstacle_id]
        # 清除該障礙物佔用的格子
        for x, y in obstacle.cells:
            self._grid_map.set_occupied(x, y, occupied=False)
        del self._obstacles[obstacle_id]
        return True

    def list(self) -> list[Obstacle]:
        """列出所有障礙物"""
        return list(self._obstacles.values())

    def clear(self) -> None:
        """清除所有障礙物"""
        obstacle_ids = list(self._obstacles.keys())
        for oid in obstacle_ids:
            self.remove(oid)

    def get(self, obstacle_id: str) -> Obstacle | None:
        """獲取指定障礙物"""
        return self._obstacles.get(obstacle_id)
