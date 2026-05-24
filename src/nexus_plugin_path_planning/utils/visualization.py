"""可視化工具函數

提供路徑規劃結果的可視化功能，生成 base64 編碼的 PNG 圖像。
"""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def visualize_path(
    grid_map: Any,
    path: list[tuple[int, int]] | None = None,
    start: tuple[int, int] | None = None,
    goal: tuple[int, int] | None = None,
    tree: list[tuple[float, float]] | None = None,
    title: str = "路徑規劃結果",
    dpi: int = 100,
    colors: dict[str, str] | None = None,
) -> str:
    """可視化路徑規劃結果

    Args:
        grid_map: GridMap 實例
        path: 規劃出的路徑點列表
        start: 起點座標
        goal: 終點座標
        tree: RRT/RRT* 探索樹節點列表
        title: 圖表標題
        dpi: 圖像 DPI
        colors: 顏色配置

    Returns:
        base64 編碼的 PNG 圖像
    """
    if colors is None:
        colors = {
            "path": "#4ecca3",
            "obstacle": "#ff6b6b",
            "start": "#00ff00",
            "goal": "#ff0000",
            "tree": "#cccccc",
            "grid": "#eeeeee",
        }

    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)

    # 繪製柵格地圖
    grid = grid_map.grid if hasattr(grid_map, "grid") else grid_map._grid
    ax.imshow(grid, cmap="gray_r", origin="upper", alpha=0.8)

    # 繪製探索樹
    if tree:
        tree_arr = np.array(tree)
        ax.scatter(
            tree_arr[:, 0],
            tree_arr[:, 1],
            c=colors["tree"],
            s=1,
            alpha=0.5,
            label="探索樹",
        )

    # 繪製路徑
    if path and len(path) > 1:
        path_arr = np.array(path)
        ax.plot(
            path_arr[:, 0],
            path_arr[:, 1],
            c=colors["path"],
            linewidth=3,
            label="規劃路徑",
        )
        ax.scatter(
            path_arr[:, 0],
            path_arr[:, 1],
            c=colors["path"],
            s=20,
            alpha=0.8,
        )

    # 繪製起點
    if start:
        ax.scatter(
            start[0],
            start[1],
            c=colors["start"],
            s=200,
            marker="o",
            edgecolors="black",
            linewidths=2,
            zorder=5,
            label="起點",
        )

    # 繪製終點
    if goal:
        ax.scatter(
            goal[0],
            goal[1],
            c=colors["goal"],
            s=200,
            marker="*",
            edgecolors="black",
            linewidths=2,
            zorder=5,
            label="終點",
        )

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # 轉為 base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def visualize_multi_agent(
    grid_map: Any,
    paths: dict[str, list[tuple[int, int]]],
    title: str = "多智能體路徑規劃",
    dpi: int = 100,
) -> str:
    """可視化多智能體路徑規劃結果

    Args:
        grid_map: GridMap 實例
        paths: 各智能體的路徑 {agent_id: path}
        title: 圖表標題
        dpi: 圖像 DPI

    Returns:
        base64 編碼的 PNG 圖像
    """
    colors_list = [
        "#4ecca3",
        "#ff6b6b",
        "#3498db",
        "#f39c12",
        "#9b59b6",
        "#1abc9c",
        "#e74c3c",
        "#2ecc71",
        "#e67e22",
        "#34495e",
    ]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)

    # 繪製柵格地圖
    grid = grid_map.grid if hasattr(grid_map, "grid") else grid_map._grid
    ax.imshow(grid, cmap="gray_r", origin="upper", alpha=0.8)

    # 繪製各智能體路徑
    for i, (agent_id, agent_path) in enumerate(paths.items()):
        color = colors_list[i % len(colors_list)]
        if agent_path and len(agent_path) > 1:
            path_arr = np.array(agent_path)
            ax.plot(
                path_arr[:, 0],
                path_arr[:, 1],
                c=color,
                linewidth=2,
                label=f"智能體 {agent_id}",
            )
            # 起點
            ax.scatter(
                agent_path[0][0],
                agent_path[0][1],
                c=color,
                s=150,
                marker="o",
                edgecolors="black",
                linewidths=2,
                zorder=5,
            )
            # 終點
            ax.scatter(
                agent_path[-1][0],
                agent_path[-1][1],
                c=color,
                s=150,
                marker="*",
                edgecolors="black",
                linewidths=2,
                zorder=5,
            )

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # 轉為 base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def compare_paths(
    grid_map: Any,
    paths: dict[str, list[tuple[int, int]]],
    start: tuple[int, int] | None = None,
    goal: tuple[int, int] | None = None,
    title: str = "路徑演算法對比",
    dpi: int = 100,
) -> str:
    """對比不同演算法的路徑規劃結果

    Args:
        grid_map: GridMap 實例
        paths: 各演算法的路徑 {algorithm_name: path}
        start: 起點座標
        goal: 終點座標
        title: 圖表標題
        dpi: 圖像 DPI

    Returns:
        base64 編碼的 PNG 圖像
    """
    colors_list = [
        "#4ecca3",
        "#ff6b6b",
        "#3498db",
        "#f39c12",
        "#9b59b6",
    ]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)

    # 繪製柵格地圖
    grid = grid_map.grid if hasattr(grid_map, "grid") else grid_map._grid
    ax.imshow(grid, cmap="gray_r", origin="upper", alpha=0.8)

    # 繪製各演算法路徑
    for i, (name, alg_path) in enumerate(paths.items()):
        color = colors_list[i % len(colors_list)]
        if alg_path and len(alg_path) > 1:
            path_arr = np.array(alg_path)
            ax.plot(
                path_arr[:, 0],
                path_arr[:, 1],
                c=color,
                linewidth=2,
                label=f"{name} ({len(alg_path)} 點)",
            )

    # 繪製起點
    if start:
        ax.scatter(
            start[0],
            start[1],
            c="#00ff00",
            s=200,
            marker="o",
            edgecolors="black",
            linewidths=2,
            zorder=5,
            label="起點",
        )

    # 繪製終點
    if goal:
        ax.scatter(
            goal[0],
            goal[1],
            c="#ff0000",
            s=200,
            marker="*",
            edgecolors="black",
            linewidths=2,
            zorder=5,
            label="終點",
        )

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # 轉為 base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
