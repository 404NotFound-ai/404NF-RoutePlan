# 404NF RoutePlan

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Nexus 智能體生態系統的路徑規劃插件，提供多種路徑規劃演算法和地圖管理功能。

## 功能特性

- **柵格地圖**：基於 numpy 陣列的二維柵格地圖（0=空閒，1=障礙物，-1=未知）
- **障礙物管理**：支援矩形、圓形、多邊形障礙物，可膨脹補償機器人半徑
- **A\***：經典最短路徑演算法，支援曼哈頓、歐幾里得、切比雪夫啟發式
- **RRT / RRT\***：快速探索隨機樹，適合高維空間，具漸近最優變體
- **多智能體**：基於優先級的協同規劃，支援衝突檢測與解決
- **路徑平滑**：貝塞爾曲線、梯度下降、三次樣條插值
- **可視化**：Base64 編碼的 PNG 路徑可視化和演算法對比

## 安裝

```bash
pip install 404nf-routeplan
```

### 依賴項

- `404nf-sagent>=0.1.0`
- `numpy>=1.24.0`
- `scipy>=1.11.0`
- `matplotlib>=3.7.0`
- `pydantic>=2.5.0`

## 快速開始

```python
import asyncio
from nexus_plugin_path_planning.plugin import PathPlanningPlugin


async def main():
    plugin = PathPlanningPlugin()
    await plugin.initialize({
        "map_width": 100,
        "map_height": 100,
    })

    await plugin._tool_plan_add_obstacle(
        shape="rectangle",
        params={"x": 40, "y": 20, "width": 5, "height": 60},
    )

    result = await plugin._tool_plan_astar_search(
        start_x=5, start_y=50,
        goal_x=90, goal_y=50,
    )
    print(f"A* 找到路徑: {result['success']}")

    await plugin.shutdown()


asyncio.run(main())
```

## 工具函數

### 地圖管理

| 工具 | 描述 |
|------|------|
| `plan_add_obstacle` | 添加障礙物（矩形/圓形/多邊形） |
| `plan_remove_obstacle` | 按 ID 移除障礙物 |
| `plan_list_obstacles` | 列出所有障礙物 |
| `plan_clear_obstacles` | 清除所有障礙物 |
| `plan_inflate_obstacles` | 按半徑膨脹障礙物 |
| `plan_get_map_info` | 獲取地圖信息 |

### 路徑規劃

| 工具 | 描述 |
|------|------|
| `plan_astar_search` | A* 最短路徑搜尋 |
| `plan_rrt_plan` | RRT 路徑規劃 |
| `plan_rrt_star_plan` | RRT* 路徑規劃 |
| `plan_multi_agent_plan` | 多智能體協同規劃 |

### 路徑處理

| 工具 | 描述 |
|------|------|
| `plan_smooth_path` | 平滑路徑（貝塞爾/梯度下降/三次樣條） |
| `plan_visualize_path` | 可視化路徑為 base64 PNG |
| `plan_visualize_multi_agent` | 可視化多智能體路徑 |
| `plan_compare_algorithms` | 對比 A*/RRT/RRT* 結果 |

## 架構

```
src/nexus_plugin_path_planning/
├── __init__.py
├── config.py            # PathPlanningConfig
├── plugin.py            # 插件主類
├── tools.py             # 工具函數別名
├── algorithms/
│   ├── __init__.py
│   ├── astar.py         # A* 演算法
│   ├── rrt.py           # RRT 演算法
│   ├── rrt_star.py      # RRT* 演算法
│   └── multi_agent.py   # 多智能體規劃器
├── map/
│   ├── __init__.py
│   ├── grid_map.py      # GridMap（基於 numpy）
│   └── obstacle.py      # Obstacle, ObstacleManager
└── utils/
    ├── __init__.py
    ├── geometry.py       # 距離、相交、插值
    ├── smoothing.py      # 貝塞爾、梯度下降、樣條
    └── visualization.py  # 基於 Matplotlib 的可視化
```

## 測試

```bash
pytest tests/ -v
```

## 許可證

MIT License
