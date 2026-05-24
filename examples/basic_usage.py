"""路徑規劃插件使用範例

展示如何使用 PathPlanningPlugin 進行路徑規劃。
"""

from __future__ import annotations

import asyncio

from nexus_plugin_path_planning.plugin import PathPlanningPlugin


async def main() -> None:
    """運行範例"""
    print("=" * 60)
    print("404NF Path Planning Plugin — 使用範例")
    print("=" * 60)

    # 1. 初始化插件
    print("\n[1] 初始化插件...")
    plugin = PathPlanningPlugin()
    await plugin.initialize(
        {
            "map_width": 50,
            "map_height": 50,
            "rrt_max_iterations": 3000,
            "rrt_star_max_iterations": 3000,
        }
    )
    print("   插件初始化完成")

    # 2. 添加障礙物
    print("\n[2] 添加障礙物...")
    # 添加一面牆
    obs1 = await plugin._tool_plan_add_obstacle(
        shape="rectangle",
        params={"x": 20, "y": 10, "width": 3, "height": 30},
        metadata={"name": "wall"},
    )
    print(f"   障礙物 1: {obs1}")

    # 添加圓形障礙物
    obs2 = await plugin._tool_plan_add_obstacle(
        shape="circle",
        params={"cx": 35, "cy": 15, "radius": 5},
        metadata={"name": "circle_obstacle"},
    )
    print(f"   障礙物 2: {obs2}")

    # 3. 獲取地圖信息
    print("\n[3] 地圖信息:")
    map_info = await plugin._tool_plan_get_map_info()
    print(f"   寬度: {map_info['width']}, 高度: {map_info['height']}")
    print(f"   障礙物數量: {map_info['obstacle_count']}")

    # 4. A* 路徑規劃
    print("\n[4] A* 路徑規劃...")
    astar_result = await plugin._tool_plan_astar_search(
        start_x=2, start_y=25, goal_x=45, goal_y=25
    )
    if astar_result["success"]:
        print(f"   ✅ A* 成功! 路徑長度: {len(astar_result['path'])} 點")
        print(f"   代價: {astar_result['cost']:.2f}")
        print(f"   訪問節點: {astar_result['visited']}")
    else:
        print("   ❌ A* 未找到路徑")

    # 5. RRT 路徑規劃
    print("\n[5] RRT 路徑規劃...")
    rrt_result = await plugin._tool_plan_rrt_plan(
        start_x=2.0, start_y=25.0, goal_x=45.0, goal_y=25.0
    )
    if rrt_result["success"]:
        print(f"   ✅ RRT 成功! 路徑長度: {len(rrt_result['path'])} 點")
        print(f"   代價: {rrt_result['cost']:.2f}")
        print(f"   迭代次數: {rrt_result['iterations']}")
    else:
        print("   ❌ RRT 未找到路徑")

    # 6. RRT* 路徑規劃
    print("\n[6] RRT* 路徑規劃...")
    rrt_star_result = await plugin._tool_plan_rrt_star_plan(
        start_x=2.0, start_y=25.0, goal_x=45.0, goal_y=25.0
    )
    if rrt_star_result["success"]:
        print(f"   ✅ RRT* 成功! 路徑長度: {len(rrt_star_result['path'])} 點")
        print(f"   代價: {rrt_star_result['cost']:.2f}")
        print(f"   迭代次數: {rrt_star_result['iterations']}")
    else:
        print("   ❌ RRT* 未找到路徑")

    # 7. 路徑平滑
    print("\n[7] 路徑平滑...")
    if astar_result["success"]:
        # 貝塞爾曲線平滑
        smooth_result = await plugin._tool_plan_smooth_path(
            path=astar_result["path"],
            method="bezier",
            num_points=200,
        )
        if smooth_result["success"]:
            print(
                f"   貝塞爾平滑: {smooth_result['original_points']} → "
                f"{smooth_result['smoothed_points']} 點"
            )

        # 梯度下降平滑
        smooth_result2 = await plugin._tool_plan_smooth_path(
            path=astar_result["path"],
            method="gradient_descent",
            alpha=0.3,
            iterations=50,
        )
        if smooth_result2["success"]:
            print(
                f"   梯度下降平滑: {smooth_result2['original_points']} → "
                f"{smooth_result2['smoothed_points']} 點"
            )

    # 8. 多智能體規劃
    print("\n[8] 多智能體協同規劃...")
    agents = [
        {"id": "robot_1", "start": [2, 5], "goal": [45, 5], "priority": 3},
        {"id": "robot_2", "start": [2, 25], "goal": [45, 25], "priority": 2},
        {"id": "robot_3", "start": [2, 45], "goal": [45, 45], "priority": 1},
    ]
    multi_result = await plugin._tool_plan_multi_agent_plan(agents=agents)
    if multi_result["success"]:
        print(f"   ✅ 多智能體規劃成功!")
        for agent_id, path in multi_result["paths"].items():
            print(f"   - {agent_id}: {len(path)} 點")
    else:
        print("   ❌ 多智能體規劃失敗")

    # 9. 演算法對比
    print("\n[9] 演算法對比...")
    compare_result = await plugin._tool_plan_compare_algorithms(
        start_x=2, start_y=25, goal_x=45, goal_y=25
    )
    print(f"   A* 代價: {compare_result['astar']['cost']:.2f}")
    print(f"   RRT 代價: {compare_result['rrt']['cost']:.2f}")
    print(f"   RRT* 代價: {compare_result['rrt_star']['cost']:.2f}")
    print(f"   對比圖像 (base64): {len(compare_result['comparison_image'])} 字符")

    # 10. 可視化
    print("\n[10] 可視化...")
    if astar_result["success"]:
        vis_result = await plugin._tool_plan_visualize_path(
            path=astar_result["path"],
            start=[2, 25],
            goal=[45, 25],
            title="A* 路徑規劃結果",
        )
        print(f"   可視化圖像 (base64): {len(vis_result['image_base64'])} 字符")

    # 11. 清除障礙物
    print("\n[11] 清除障礙物...")
    clear_result = await plugin._tool_plan_clear_obstacles()
    print(f"   已清除 {clear_result['cleared']} 個障礙物")

    # 12. 關閉插件
    print("\n[12] 關閉插件...")
    await plugin.shutdown()
    print("   插件已關閉")

    print("\n" + "=" * 60)
    print("範例執行完畢!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
