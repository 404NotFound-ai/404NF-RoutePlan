# 404NF RoutePlan

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A path planning plugin for the Nexus agent ecosystem, providing multiple path planning algorithms and map management capabilities.

## Features

- **Grid Map**: 2D grid-based map with numpy array (0=free, 1=obstacle, -1=unknown)
- **Obstacle Management**: Rectangle, circle, polygon obstacle support with inflation for robot radius compensation
- **A\***: Classic shortest path algorithm with Manhattan, Euclidean, Chebyshev heuristics
- **RRT / RRT\***: Rapidly-exploring Random Tree for high-dimensional spaces, with asymptotically optimal variant
- **Multi-Agent**: Priority-based collaborative planning with conflict detection and resolution
- **Path Smoothing**: Bezier curve, gradient descent, cubic spline interpolation
- **Visualization**: Base64-encoded PNG visualization of paths and algorithm comparisons

## Installation

```bash
pip install 404nf-routeplan
```

### Dependencies

- `404nf-sagent>=0.1.0`
- `numpy>=1.24.0`
- `scipy>=1.11.0`
- `matplotlib>=3.7.0`
- `pydantic>=2.5.0`

## Quick Start

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
    print(f"A* found path: {result['success']}")

    await plugin.shutdown()


asyncio.run(main())
```

## Tools

### Map Management

| Tool | Description |
|------|-------------|
| `plan_add_obstacle` | Add obstacle (rectangle/circle/polygon) |
| `plan_remove_obstacle` | Remove obstacle by ID |
| `plan_list_obstacles` | List all obstacles |
| `plan_clear_obstacles` | Clear all obstacles |
| `plan_inflate_obstacles` | Inflate obstacles by radius |
| `plan_get_map_info` | Get map information |

### Path Planning

| Tool | Description |
|------|-------------|
| `plan_astar_search` | A* shortest path search |
| `plan_rrt_plan` | RRT path planning |
| `plan_rrt_star_plan` | RRT* path planning |
| `plan_multi_agent_plan` | Multi-agent collaborative planning |

### Path Processing

| Tool | Description |
|------|-------------|
| `plan_smooth_path` | Smooth path (bezier/gradient_descent/cubic_spline) |
| `plan_visualize_path` | Visualize path as base64 PNG |
| `plan_visualize_multi_agent` | Visualize multi-agent paths |
| `plan_compare_algorithms` | Compare A*/RRT/RRT* results |

## Architecture

```
src/nexus_plugin_path_planning/
├── __init__.py
├── config.py            # PathPlanningConfig
├── plugin.py            # Main plugin class
├── tools.py             # Tool function aliases
├── algorithms/
│   ├── __init__.py
│   ├── astar.py         # A* algorithm
│   ├── rrt.py           # RRT algorithm
│   ├── rrt_star.py      # RRT* algorithm
│   └── multi_agent.py   # Multi-agent planner
├── map/
│   ├── __init__.py
│   ├── grid_map.py      # GridMap (numpy-based)
│   └── obstacle.py      # Obstacle, ObstacleManager
└── utils/
    ├── __init__.py
    ├── geometry.py       # Distance, intersection, interpolation
    ├── smoothing.py      # Bezier, gradient descent, spline
    └── visualization.py  # Matplotlib-based visualization
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT License
