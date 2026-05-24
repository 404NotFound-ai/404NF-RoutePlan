"""柵格地圖 — GridMap

管理障礙物和可通行區域的柵格地圖。
"""

from __future__ import annotations

from typing import Any

import numpy as np


class GridMap:
    """柵格地圖

    使用二維 numpy 數組表示地圖：
    - 0 = 空閒（可通行）
    - 1 = 障礙物
    - -1 = 未知
    """

    def __init__(self, width: int, height: int, resolution: float = 1.0) -> None:
        self._width: int = width
        self._height: int = height
        self._resolution: float = resolution
        # 0 = 空閒, 1 = 障礙物, -1 = 未知
        self._grid: np.ndarray = np.zeros((height, width), dtype=np.int8)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def resolution(self) -> float:
        return self._resolution

    @property
    def grid(self) -> np.ndarray:
        return self._grid.copy()

    def is_occupied(self, x: int, y: int) -> bool:
        """檢查格子是否被佔用

        Args:
            x: 列索引
            y: 行索引

        Returns:
            是否被佔用
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            return True  # 邊界外視為障礙物
        return bool(self._grid[y, x] == 1)

    def set_occupied(self, x: int, y: int, occupied: bool = True) -> None:
        """設置格子佔用狀態

        Args:
            x: 列索引
            y: 行索引
            occupied: 是否設為障礙物
        """
        if 0 <= x < self._width and 0 <= y < self._height:
            self._grid[y, x] = 1 if occupied else 0

    def add_obstacle(
        self, shape: str, params: dict[str, float], occupied: bool = True
    ) -> list[tuple[int, int]]:
        """添加障礙物（支持矩形/圓形/多邊形）

        Args:
            shape: 障礙物形狀 rectangle/circle/polygon
            params: 形狀參數
                rectangle: {x, y, width, height}
                circle: {cx, cy, radius}
                polygon: {points: [[x1,y1], [x2,y2], ...]}
            occupied: 是否設為障礙物

        Returns:
            受影響的格子座標列表
        """
        cells: list[tuple[int, int]] = []

        if shape == "rectangle":
            x = int(params.get("x", 0))
            y = int(params.get("y", 0))
            w = int(params.get("width", 1))
            h = int(params.get("height", 1))
            for i in range(max(0, x), min(self._width, x + w)):
                for j in range(max(0, y), min(self._height, y + h)):
                    self._grid[j, i] = 1 if occupied else 0
                    cells.append((i, j))

        elif shape == "circle":
            cx = float(params.get("cx", 0))
            cy = float(params.get("cy", 0))
            radius = float(params.get("radius", 1))
            for i in range(self._width):
                for j in range(self._height):
                    if (i - cx) ** 2 + (j - cy) ** 2 <= radius**2:
                        self._grid[j, i] = 1 if occupied else 0
                        cells.append((i, j))

        elif shape == "polygon":
            points = params.get("points", [])
            if points:
                import matplotlib.path as mpath

                polygon = mpath.Path(points)
                for i in range(self._width):
                    for j in range(self._height):
                        if polygon.contains_point((i, j)):
                            self._grid[j, i] = 1 if occupied else 0
                            cells.append((i, j))

        return cells

    def remove_obstacle(self, x: int, y: int, radius: int = 0) -> None:
        """移除障礙物

        Args:
            x: 中心列索引
            y: 中心行索引
            radius: 移除半徑（0 只移除指定格子）
        """
        for i in range(x - radius, x + radius + 1):
            for j in range(y - radius, y + radius + 1):
                if 0 <= i < self._width and 0 <= j < self._height:
                    self._grid[j, i] = 0

    def get_obstacles(self) -> list[tuple[int, int]]:
        """獲取所有障礙物座標"""
        obstacles: list[tuple[int, int]] = []
        for y in range(self._height):
            for x in range(self._width):
                if self._grid[y, x] == 1:
                    obstacles.append((x, y))
        return obstacles

    def to_dict(self) -> dict[str, Any]:
        """導出地圖為字典"""
        return {
            "width": self._width,
            "height": self._height,
            "resolution": self._resolution,
            "grid": self._grid.tolist(),
            "obstacle_count": int(np.sum(self._grid == 1)),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GridMap:
        """從字典導入地圖"""
        grid_map = cls(
            width=data.get("width", 100),
            height=data.get("height", 100),
            resolution=data.get("resolution", 1.0),
        )
        grid_data = data.get("grid")
        if grid_data is not None:
            grid_map._grid = np.array(grid_data, dtype=np.int8)
        return grid_map

    def inflate_obstacles(self, radius: int) -> None:
        """膨脹障礙物（用於機器人半徑補償）

        Args:
            radius: 膨脹半徑（格數）
        """
        if radius <= 0:
            return
        from scipy.ndimage import binary_dilation

        structure = np.ones((2 * radius + 1, 2 * radius + 1))
        obstacles = (self._grid == 1).astype(np.int8)
        dilated = binary_dilation(obstacles, structure=structure).astype(np.int8)
        self._grid = np.where(dilated == 1, 1, self._grid)
