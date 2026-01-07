import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from collections import defaultdict

class MatchConfig:
    """轨迹匹配算法配置类"""

    def __init__(self):
        # 空间距离阈值：越小匹配越严格 
        self.spatial_threshold = 1.5
        # 轨迹相似度阈值：越小匹配越严格 
        self.trajectory_threshold = 2.5
        # 观测一致性最低阈值
        self.consistency_threshold = 0.35
        # 共同观测点数量最低要求
        self.min_common_points = 10
        # 最终综合匹配阈值 
        self.final_match_threshold = 0.63
        # 时间偏移最大允许值（秒）
        self.time_offset_threshold = 10.0
        # 权重分配 
        self.spatial_weight = 0.45
        self.trajectory_weight = 0.45
        self.consistency_weight = 0.1

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    # 按照id和timestamp排序
    df = df.sort_values(['id', 'timestamp'])
    return df

def calculate_spatial_membership(distance, threshold):
    """
    计算空间距离的模糊隶属度
    使用高斯隶属度函数: exp(-(d/threshold)^2)
    距离越近,隶属度越接近1;距离越远,隶属度越接近0
    """
    return np.exp(-(distance/threshold)**2)

def calculate_trajectory_membership(similarity, threshold):
    """
    计算轨迹相似度的模糊隶属度
    使用S型隶属度函数,相似度越高(值越小),隶属度越接近1;相似度越低,隶属度越接近0
    """
    if similarity >= threshold:
        return 0
    elif similarity <= 0:
        return 1
    else:
        return 0.5 * (1 + np.cos(np.pi * similarity / threshold))

def calculate_time_alignment_score(traj1_df, traj2_df, max_time_offset):
    """计算时间对齐得分，考虑观测延迟
    Args:
        traj1_df: 轨迹1 DataFrame
        traj2_df: 轨迹2 DataFrame
        max_time_offset: 最大允许时间偏移(秒)
    Returns:
        dict: 时间对齐信息
    """
    timestamps1 = sorted(traj1_df['timestamp'].unique())
    timestamps2 = sorted(traj2_df['timestamp'].unique())

    # 计算时间范围重叠
    start1, end1 = timestamps1[0], timestamps1[-1]
    start2, end2 = timestamps2[0], timestamps2[-1]
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_end <= overlap_start:
        return {'alignment_score': 0, 'common_timestamps': []}
    # 在重叠时间范围内寻找最相近的时间戳对
    common_timestamps = []
    time_offset_threshold = max_time_offset
    # 对轨迹1的每个时间戳，找轨迹2中最接近的时间戳
    for ts1 in timestamps1:
        if overlap_start <= ts1 <= overlap_end:
            closest_ts2 = min(timestamps2, key=lambda ts2: abs(ts2 - ts1))
            if abs(closest_ts2 - ts1) <= time_offset_threshold:
                common_timestamps.append((ts1, closest_ts2))
    # 计算时间对齐质量
    if len(common_timestamps) < 3:
        return {'alignment_score': 0, 'common_timestamps': []}
    # 时间偏移的标准差（越小越好）
    time_offsets = [abs(ts1 - ts2) for ts1, ts2 in common_timestamps]
    avg_offset = np.mean(time_offsets)
    offset_std = np.std(time_offsets)
    # 对齐得分：基于共同点数量和时间偏移稳定性
    alignment_score = len(common_timestamps) / max(len(timestamps1), len(timestamps2))
    alignment_score *= np.exp(-avg_offset / 0.05)  # 50ms平均偏移惩罚
    alignment_score *= np.exp(-offset_std / 0.02)  # 时间抖动惩罚

    return {
        'alignment_score': alignment_score,
        'common_timestamps': common_timestamps,
        'avg_time_offset': avg_offset,
        'time_offset_std': offset_std
    }

def calculate_trajectory_similarity(traj1_df, traj2_df, max_time_offset=0.1):
    """计算两条轨迹的相似度，基于时间戳对齐的点对距离
    Args:
        traj1_df: 包含 timestamp, x, y 列的DataFrame
        traj2_df: 包含 timestamp, x, y 列的DataFrame
        max_time_offset: 最大允许时间偏移(秒)
    Returns:
        float: 平均距离，如果没有共同时间戳则返回inf
    """
    time_alignment = calculate_time_alignment_score(traj1_df, traj2_df, max_time_offset)

    if time_alignment['alignment_score'] < 0.1:  # 时间对齐质量不足
        return float('inf')
    common_timestamps = time_alignment['common_timestamps']
    if len(common_timestamps) < 3:
        return float('inf')
    total_dist = 0
    valid_pairs = 0

    for ts1, ts2 in common_timestamps:
        # 获取对应时间戳的位置
        p1_rows = traj1_df[traj1_df['timestamp'] == ts1]
        p2_rows = traj2_df[traj2_df['timestamp'] == ts2]
        if not p1_rows.empty and not p2_rows.empty:
            p1 = p1_rows[['x', 'y']].values[0]
            p2 = p2_rows[['x', 'y']].values[0]
            total_dist += euclidean(p1, p2)
            valid_pairs += 1
    if valid_pairs == 0:
        return float('inf')

    return total_dist / valid_pairs

def calculate_observation_consistency(vehicle_traj, roadside_traj, max_time_offset=0.1):
    """计算车端和路端观测的一致性
    Args:
        vehicle_traj: 车端轨迹DataFrame
        roadside_traj: 路端轨迹DataFrame
        max_time_offset: 最大允许时间偏移(秒)
    Returns:
        dict: 观测一致性指标
    """
    time_alignment = calculate_time_alignment_score(vehicle_traj, roadside_traj, max_time_offset)
    common_pairs = time_alignment['common_timestamps']

    if len(common_pairs) < 3:
        return {
            'consistency_score': 0,
            'observation_ratio': 0,
            'spatial_consistency': 0,
            'temporal_coverage': 0
        }

    # 1. 观测比例：共同观测时间占各自总观测时间的比例
    v_total_time = vehicle_traj['timestamp'].max() - vehicle_traj['timestamp'].min()
    r_total_time = roadside_traj['timestamp'].max() - roadside_traj['timestamp'].min()
    common_time = max([ts1 for ts1, ts2 in common_pairs]) - min([ts1 for ts1, ts2 in common_pairs])

    v_obs_ratio = common_time / v_total_time if v_total_time > 0 else 0
    r_obs_ratio = common_time / r_total_time if r_total_time > 0 else 0
    observation_ratio = (v_obs_ratio + r_obs_ratio) / 2

    # 2. 空间一致性：共同观测点的空间距离分布
    distances = []
    for ts1, ts2 in common_pairs:
        v_point = vehicle_traj[vehicle_traj['timestamp'] == ts1][['x', 'y']].values
        r_point = roadside_traj[roadside_traj['timestamp'] == ts2][['x', 'y']].values

        if len(v_point) > 0 and len(r_point) > 0:
            dist = euclidean(v_point[0], r_point[0])
            distances.append(dist)

    if distances:
        avg_distance = np.mean(distances)
        distance_std = np.std(distances)
        # 空间一致性：距离越小且越稳定，一致性越高
        spatial_consistency = np.exp(-avg_distance / 2.0) * np.exp(-distance_std / 1.0)
    else:
        spatial_consistency = 0

    # 3. 时间覆盖率：考虑观测的连续性和完整性
    v_timestamps = set(vehicle_traj['timestamp'])
    r_timestamps = set(roadside_traj['timestamp'])
    common_timestamps = set([ts1 for ts1, ts2 in common_pairs])

    v_coverage = len(common_timestamps) / len(v_timestamps) if v_timestamps else 0
    r_coverage = len(common_timestamps) / len(r_timestamps) if r_timestamps else 0
    temporal_coverage = (v_coverage + r_coverage) / 2

    # 4. 综合一致性得分
    consistency_score = (
        0.3 * observation_ratio +
        0.4 * spatial_consistency +
        0.3 * temporal_coverage
    )

    return {
        'consistency_score': consistency_score,
        'observation_ratio': observation_ratio,
        'spatial_consistency': spatial_consistency,
        'temporal_coverage': temporal_coverage,
        'avg_distance': avg_distance if distances else float('inf'),
        'distance_std': distance_std if distances else float('inf')
    }

def match_trajectories(vehicle_df, roadside_df, config=None):
    # 使用默认配置或传入配置
    if config is None:
        config = MatchConfig()

    matches = {}
    matched_roadside_ids = set()

    vehicle_groups = vehicle_df.groupby('id')
    roadside_groups = roadside_df.groupby('id')

    # 对每个车端ID
    for v_id, v_group in vehicle_groups:
        best_match = None
        best_match_score = float('-inf')
        best_match_details = {}

        # 对每个路端ID
        for r_id, r_group in roadside_groups:
            if r_id in matched_roadside_ids:
                continue

            # 计算观测一致性
            consistency = calculate_observation_consistency(v_group, r_group, config.time_offset_threshold)
            # 如果观测一致性太低，跳过
            if consistency['consistency_score'] < config.consistency_threshold:
                continue
            # 计算空间距离的模糊隶属度（基于共同观测点）
            time_alignment = calculate_time_alignment_score(v_group, r_group, config.time_offset_threshold)
            common_pairs = time_alignment['common_timestamps']
            if len(common_pairs) < config.min_common_points:
                continue
            spatial_distances = []
            for ts1, ts2 in common_pairs:
                v_point = v_group[v_group['timestamp'] == ts1][['x', 'y']].values
                r_point = r_group[r_group['timestamp'] == ts2][['x', 'y']].values
                if len(v_point) > 0 and len(r_point) > 0:
                    dist = euclidean(v_point[0], r_point[0])
                    spatial_distances.append(dist)

            if not spatial_distances:
                continue

            avg_spatial_distance = np.mean(spatial_distances)
            spatial_membership = calculate_spatial_membership(avg_spatial_distance, config.spatial_threshold)
            # 计算轨迹相似度的模糊隶属度
            trajectory_similarity = calculate_trajectory_similarity(v_group, r_group, config.time_offset_threshold)
            if trajectory_similarity == float('inf'):
                continue
            trajectory_membership = calculate_trajectory_membership(trajectory_similarity, config.trajectory_threshold)
            # 计算综合隶属度
            total_membership = (
                config.spatial_weight * spatial_membership +
                config.trajectory_weight * trajectory_membership +
                config.consistency_weight * consistency['consistency_score']
            )

            if total_membership > best_match_score:
                best_match_score = total_membership
                best_match = r_id
                best_match_details = {
                    'score': best_match_score,
                    'spatial_score': spatial_membership,
                    'trajectory_score': trajectory_membership,
                    'consistency_score': consistency['consistency_score'],
                    'avg_distance': consistency['avg_distance'],
                    'observation_ratio': consistency['observation_ratio']
                }

        # 只有当综合隶属度超过阈值时才认为匹配成功
        if best_match is not None and best_match_score > config.final_match_threshold:
            matches[v_id] = best_match
            matched_roadside_ids.add(best_match)
            print(f"Matched vehicle {v_id} with roadside {best_match} "
                  ".3f"
                  ".3f")
    return matches

def merge_matched_data(vehicle_df, roadside_df, matches):
    merged_data = []
    
    # 处理匹配的数据
    for v_id, r_id in matches.items():
        v_group = vehicle_df[vehicle_df['id'] == v_id]
        r_group = roadside_df[roadside_df['id'] == r_id]
        # 获取两个轨迹的所有时间戳
        all_timestamps = sorted(set(v_group['timestamp']) | set(r_group['timestamp']))
        common_timestamps = set(v_group['timestamp']) & set(r_group['timestamp'])
        # 创建组合ID
        combined_id = f"{v_id}_{r_id}"
        for ts in all_timestamps:
            if ts in common_timestamps:
                # 如果两端都有数据，取平均位置
                v_row = v_group[v_group['timestamp'] == ts].iloc[0]
                r_row = r_group[r_group['timestamp'] == ts].iloc[0]
                avg_x = (v_row['x'] + r_row['x']) / 2
                avg_y = (v_row['y'] + r_row['y']) / 2
            elif ts in v_group['timestamp'].values:
                # 如果只有车端数据
                v_row = v_group[v_group['timestamp'] == ts].iloc[0]
                avg_x = v_row['x']
                avg_y = v_row['y']
            else:
                # 如果只有路端数据
                r_row = r_group[r_group['timestamp'] == ts].iloc[0]
                avg_x = r_row['x']
                avg_y = r_row['y']
            
            # 创建新的行，使用组合ID
            new_row = {
                'timestamp': ts,
                'id': combined_id,
                'type': v_group.iloc[0]['type'],  # 使用车端类型
                'x': avg_x,
                'y': avg_y
            }
            merged_data.append(new_row)
    
    # 添加未匹配的车端和路端数据（保持不变）
    unmatched_v_ids = set(vehicle_df['id']) - set(matches.keys())
    for v_id in unmatched_v_ids:
        v_group = vehicle_df[vehicle_df['id'] == v_id]
        for _, row in v_group.iterrows():
            merged_data.append({
                'timestamp': row['timestamp'],
                'id': str(row['id']),
                'type': row['type'],
                'x': row['x'],
                'y': row['y']
            })
    
    # 添加未匹配的路端数据
    unmatched_r_ids = set(roadside_df['id']) - set(matches.values())
    for r_id in unmatched_r_ids:
        r_group = roadside_df[roadside_df['id'] == r_id]
        for _, row in r_group.iterrows():
            merged_data.append({
                'timestamp': row['timestamp'],
                'id': str(row['id']),
                'type': row['type'],
                'x': row['x'],
                'y': row['y']
            })
    
    # 转换为DataFrame并排序
    result_df = pd.DataFrame(merged_data)
    result_df = result_df.sort_values(['id', 'timestamp'])
    
    return result_df

def main():
    # 读取数据
    vehicle_file = "/media/lhp/LHP_SSD/DAIR-V2X数据集/车路协同轨迹预测v2x-Seq-TFD/V2X-Seq-TFD/demo/extract/14_vehicle_ext.csv"
    roadside_file = "/media/lhp/LHP_SSD/DAIR-V2X数据集/车路协同轨迹预测v2x-Seq-TFD/V2X-Seq-TFD/demo/extract/14_inf_ext.csv"
    
    vehicle_df = load_and_preprocess_data(vehicle_file)
    roadside_df = load_and_preprocess_data(roadside_file)
    
    matches = match_trajectories(vehicle_df, roadside_df)
    
    # 合并数据
    merged_df = merge_matched_data(vehicle_df, roadside_df, matches)
    merged_df.to_csv('merged_data7.csv', index=False)
    print(f"Matched pairs: {len(matches)}")
    print("Data has been merged and saved to merged_data7.csv")

if __name__ == "__main__":
    main()
