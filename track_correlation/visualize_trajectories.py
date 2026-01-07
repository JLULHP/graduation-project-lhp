import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_trajectories(vehicle_file, roadside_file):
    """
    可视化车端和路端轨迹数据，清晰标注每个轨迹的ID
    """
    vehicle_df = pd.read_csv(vehicle_file)
    roadside_df = pd.read_csv(roadside_file)

    # 创建图形
    fig, (ax_main, ax_legend) = plt.subplots(1, 2, figsize=(16, 8),
                                             gridspec_kw={'width_ratios': [3, 1]})

    # 绘制车端轨迹（不同ID用不同颜色）
    vehicle_ids = vehicle_df['id'].unique()
    # 使用tab10颜色映射，更容易区分
    colors_vehicle = plt.cm.tab10(np.linspace(0, 1, len(vehicle_ids)))

    vehicle_legend_info = []
    for i, v_id in enumerate(vehicle_ids):
        v_traj = vehicle_df[vehicle_df['id'] == v_id]
        color = colors_vehicle[i % 10]  # 循环使用颜色

        # 绘制轨迹线
        ax_main.plot(v_traj['x'], v_traj['y'],
                    color=color,
                    linewidth=2,
                    alpha=0.8,
                    marker='o',
                    markersize=3,
                    markevery=max(1, len(v_traj)//20))  # 每20个点标记一个

        # 在轨迹起点标注ID
        start_point = v_traj.iloc[0]
        ax_main.annotate(f'V{v_id}', (start_point['x'], start_point['y']),
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                        fontsize=8, fontweight='bold')

        vehicle_legend_info.append(f'Vehicle {v_id}')

    # 绘制路端轨迹（不同ID用不同颜色和虚线）
    roadside_ids = roadside_df['id'].unique()
    colors_roadside = plt.cm.tab10(np.linspace(0, 1, len(roadside_ids)))

    roadside_legend_info = []
    for i, r_id in enumerate(roadside_ids):
        r_traj = roadside_df[roadside_df['id'] == r_id]
        color = colors_roadside[i % 10]

        # 绘制轨迹线（虚线）
        ax_main.plot(r_traj['x'], r_traj['y'],
                    color=color,
                    linewidth=2,
                    alpha=0.8,
                    linestyle='--',
                    marker='s',
                    markersize=3,
                    markevery=max(1, len(r_traj)//20))

        # 在轨迹起点标注ID
        start_point = r_traj.iloc[0]
        ax_main.annotate(f'R{r_id}', (start_point['x'], start_point['y']),
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8),
                        fontsize=8, fontweight='bold')

        roadside_legend_info.append(f'Roadside {r_id}')

    # 设置主图属性
    ax_main.set_xlabel('X coordinate (m)')
    ax_main.set_ylabel('Y coordinate (m)')
    ax_main.set_title('Vehicle and Roadside Trajectories\n(V: Vehicle, R: Roadside)')
    ax_main.grid(True, alpha=0.3)
    ax_main.axis('equal')

    # 在右侧创建ID映射图例
    ax_legend.axis('off')
    legend_text = "轨迹ID映射表\n\n"

    # 车端轨迹ID
    legend_text += "车端轨迹 (Vehicle):\n"
    for i, v_id in enumerate(vehicle_ids[:20]):  # 只显示前20个
        color = colors_vehicle[i % 10]
        legend_text += f"• V{v_id}\n"
    if len(vehicle_ids) > 20:
        legend_text += f"... 还有 {len(vehicle_ids)-20} 个\n"

    legend_text += "\n路端轨迹 (Roadside):\n"
    for i, r_id in enumerate(roadside_ids[:20]):  # 只显示前20个
        color = colors_roadside[i % 10]
        legend_text += f"• R{r_id}\n"
    if len(roadside_ids) > 20:
        legend_text += f"... 还有 {len(roadside_ids)-20} 个\n"

    ax_legend.text(0.05, 0.95, legend_text, transform=ax_legend.transAxes,
                  fontsize=9, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # 添加统计信息
    stats_text = f'统计信息:\n车端轨迹: {len(vehicle_ids)} 条\n路端轨迹: {len(roadside_ids)} 条'
    ax_main.text(0.02, 0.98, stats_text, transform=ax_main.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('trajectories_visualization_with_ids.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    vehicle_file = "demo/extract/14_vehicle_ext.csv"
    roadside_file = "demo/extract/14_inf_ext.csv"
    visualize_trajectories(vehicle_file, roadside_file)