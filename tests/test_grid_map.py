"""柵格地圖測試"""

from __future__ import annotations

import numpy as np
import pytest

from nexus_plugin_path_planning.map.grid_map import GridMap


class TestGridMap:
    """GridMap 基本功能測試"""

    def test_initialization(self) -> None:
        """測試初始化"""
        gm = GridMap(width=100, height=50, resolution=0.5)
        assert gm.width == 100
        assert gm.height == 50
        assert gm.resolution == 0.5
        assert gm.grid.shape == (50, 100)
        assert np.all(gm.grid == 0)

    def test_is_occupied_free(self) -> None:
        """測試空閒格子"""
        gm = GridMap(10, 10)
        assert not gm.is_occupied(5, 5)

    def test_is_occupied_obstacle(self) -> None:
        """測試障礙物格子"""
        gm = GridMap(10, 10)
        gm.set_occupied(3, 4)
        assert gm.is_occupied(3, 4)

    def test_is_occupied_out_of_bounds(self) -> None:
        """測試邊界外視為障礙物"""
        gm = GridMap(10, 10)
        assert gm.is_occupied(-1, 5)
        assert gm.is_occupied(10, 5)
        assert gm.is_occupied(5, -1)
        assert gm.is_occupied(5, 10)

    def test_set_occupied(self) -> None:
        """測試設置佔用狀態"""
        gm = GridMap(10, 10)
        gm.set_occupied(2, 3, True)
        assert gm.is_occupied(2, 3)

        gm.set_occupied(2, 3, False)
        assert not gm.is_occupied(2, 3)

    def test_set_occupied_out_of_bounds(self) -> None:
        """測試邊界外設置不報錯"""
        gm = GridMap(10, 10)
        gm.set_occupied(-1, 5)  # 不應報錯
        gm.set_occupied(100, 5)  # 不應報錯

    def test_add_rectangle_obstacle(self) -> None:
        """測試添加矩形障礙物"""
        gm = GridMap(20, 20)
        cells = gm.add_obstacle("rectangle", {"x": 5, "y": 5, "width": 3, "height": 4})
        assert len(cells) == 12  # 3 * 4
        assert gm.is_occupied(5, 5)
        assert gm.is_occupied(7, 8)
        assert not gm.is_occupied(4, 5)

    def test_add_circle_obstacle(self) -> None:
        """測試添加圓形障礙物"""
        gm = GridMap(20, 20)
        cells = gm.add_obstacle("circle", {"cx": 10, "cy": 10, "radius": 3})
        assert len(cells) > 0
        assert gm.is_occupied(10, 10)
        assert gm.is_occupied(10, 12)
        assert not gm.is_occupied(5, 5)

    def test_add_polygon_obstacle(self) -> None:
        """測試添加多邊形障礙物"""
        gm = GridMap(20, 20)
        cells = gm.add_obstacle(
            "polygon",
            {"points": [[5, 5], [10, 5], [10, 10], [5, 10]]},
        )
        assert len(cells) > 0
        assert gm.is_occupied(7, 7)

    def test_remove_obstacle(self) -> None:
        """測試移除障礙物"""
        gm = GridMap(10, 10)
        gm.set_occupied(3, 3)
        gm.set_occupied(3, 4)
        gm.remove_obstacle(3, 3, radius=1)
        assert not gm.is_occupied(3, 3)
        assert not gm.is_occupied(3, 4)

    def test_get_obstacles(self) -> None:
        """測試獲取障礙物列表"""
        gm = GridMap(10, 10)
        gm.set_occupied(1, 1)
        gm.set_occupied(2, 2)
        obstacles = gm.get_obstacles()
        assert len(obstacles) == 2
        assert (1, 1) in obstacles
        assert (2, 2) in obstacles

    def test_to_dict_and_from_dict(self) -> None:
        """測試序列化與反序列化"""
        gm = GridMap(10, 10)
        gm.set_occupied(3, 4)
        gm.set_occupied(5, 6)

        data = gm.to_dict()
        assert data["width"] == 10
        assert data["height"] == 10
        assert data["obstacle_count"] == 2

        restored = GridMap.from_dict(data)
        assert restored.width == 10
        assert restored.height == 10
        assert restored.is_occupied(3, 4)
        assert not restored.is_occupied(0, 0)

    def test_inflate_obstacles(self) -> None:
        """測試障礙物膨脹"""
        gm = GridMap(10, 10)
        gm.set_occupied(5, 5)
        gm.inflate_obstacles(radius=1)
        # 膨脹後周圍 8 格也應為障礙物
        assert gm.is_occupied(4, 4)
        assert gm.is_occupied(5, 4)
        assert gm.is_occupied(6, 4)
        assert gm.is_occupied(4, 5)
        assert gm.is_occupied(5, 5)
        assert gm.is_occupied(6, 5)
        assert gm.is_occupied(4, 6)
        assert gm.is_occupied(5, 6)
        assert gm.is_occupied(6, 6)

    def test_inflate_obstacles_zero_radius(self) -> None:
        """測試半徑為 0 時不膨脹"""
        gm = GridMap(10, 10)
        gm.set_occupied(5, 5)
        gm.inflate_obstacles(radius=0)
        assert gm.is_occupied(5, 5)
        assert not gm.is_occupied(4, 4)
