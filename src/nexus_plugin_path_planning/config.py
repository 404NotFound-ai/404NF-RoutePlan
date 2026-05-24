"""配置模型 — PathPlanningConfig"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PathPlanningConfig(BaseModel):
    """路徑規劃插件配置模型"""

    # 地圖配置
    map_width: int = Field(default=100, ge=10, le=1000, description="柵格地圖寬度")
    map_height: int = Field(default=100, ge=10, le=1000, description="柵格地圖高度")
    map_resolution: float = Field(default=1.0, ge=0.1, le=10.0, description="地圖解析度，單位/格")

    # RRT 配置
    rrt_max_iterations: int = Field(default=5000, ge=100, le=100000, description="RRT 最大迭代次數")
    rrt_step_size: float = Field(default=5.0, ge=0.5, le=50.0, description="RRT 步長")

    # RRT* 配置
    rrt_star_max_iterations: int = Field(default=5000, ge=100, le=100000, description="RRT* 最大迭代次數")
    rrt_star_step_size: float = Field(default=5.0, ge=0.5, le=50.0, description="RRT* 步長")
    rrt_star_search_radius: float = Field(default=15.0, ge=1.0, le=100.0, description="RRT* 搜索半徑")

    # 多智能體配置
    max_agents: int = Field(default=10, ge=1, le=100, description="最大智能體數量")
    collision_radius: float = Field(default=2.0, ge=0.5, le=10.0, description="智能體碰撞半徑")

    # 可視化配置
    visualization_dpi: int = Field(default=100, ge=50, le=300, description="可視化圖像 DPI")
    visualization_colors: dict[str, str] = Field(
        default_factory=lambda: {
            "path": "#4ecca3",
            "obstacle": "#ff6b6b",
            "start": "#00ff00",
            "goal": "#ff0000",
            "tree": "#cccccc",
            "grid": "#eeeeee",
        },
        description="可視化顏色配置",
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PathPlanningConfig:
        """從字典創建配置實例"""
        return cls(**{k: v for k, v in data.items() if k in cls.model_fields})
