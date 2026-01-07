import pandas as pd
import numpy as np
from tools_match7 import match_trajectories, load_and_preprocess_data, MatchConfig
from sklearn.metrics import precision_score, recall_score, f1_score

def load_ground_truth(gt_file, vehicle_file=None):
    """
    加载ground truth匹配关系
    返回格式: {car_id: road_id}
    注意：
    1. 过滤掉road_side_id为-1的无效匹配
    2. 只考虑tag不为OTHERS的目标（因为cooperative数据只记录这些）
    3. 每个car_id只保留第一个有效的road_id
    """
    df = pd.read_csv(gt_file)

    # 过滤掉无效匹配（road_side_id = -1）
    valid_df = df[df['road_side_id'] != -1]

    # 如果提供了vehicle_file，只考虑tag不为OTHERS的车端ID
    target_car_ids = None
    if vehicle_file:
        vehicle_df = pd.read_csv(vehicle_file)
        # 获取tag不为OTHERS的ID集合
        target_ids = vehicle_df[vehicle_df['tag'] != 'OTHERS']['id'].unique()
        target_car_ids = set(target_ids)

    # 按照car_side_id分组，每组只保留第一个有效的匹配
    gt_matches = {}
    for car_id, group in valid_df.groupby('car_side_id'):
        # 如果指定了target_car_ids，只保留tag不为OTHERS的
        if target_car_ids is not None and car_id not in target_car_ids:
            continue
        # 取每组的第一行（按文件顺序）
        first_match = group.iloc[0]
        gt_matches[car_id] = first_match['road_side_id']

    return gt_matches

def evaluate_matching(predictions, ground_truth, target_car_ids=None):
    """
    计算匹配算法的评估指标
    如果提供了target_car_ids，只评估tag不为OTHERS的车端ID
    """
    # 如果指定了target_car_ids，只评估这些ID
    if target_car_ids is not None:
        predictions = {car_id: road_id for car_id, road_id in predictions.items() 
                      if car_id in target_car_ids}
        ground_truth = {car_id: road_id for car_id, road_id in ground_truth.items() 
                       if car_id in target_car_ids}
    
    # 计算精确率：预测正确的匹配 / 预测的总匹配数
    if predictions:
        correct_predictions = sum(1 for car_id, road_id in predictions.items()
                                 if ground_truth.get(car_id) == road_id)
        precision = correct_predictions / len(predictions)
    else:
        precision = 0.0

    # 计算召回率：预测正确的匹配 / ground truth的总匹配数
    total_gt_matches = len(ground_truth)
    correct_matches = sum(1 for car_id, road_id in predictions.items()
                         if ground_truth.get(car_id) == road_id)
    recall = correct_matches / total_gt_matches if total_gt_matches > 0 else 0.0

    # 计算F1分数
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # 计算匹配准确率（同召回率）
    matching_accuracy = recall

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'matching_accuracy': matching_accuracy,
        'correct_matches': correct_matches,
        'total_gt_matches': total_gt_matches
    }

def validate_on_scene(scene_id):
    """
    在指定场景上验证匹配算法
    """
    print(f" 开始验证场景 {scene_id} 的匹配算法...")

    # 文件路径
    vehicle_file = f"cooperative-vehicle-infrastructure/vehicle-trajectories/train/{scene_id}.csv"
    roadside_file = f"cooperative-vehicle-infrastructure/infrastructure-trajectories/train/{scene_id}.csv"
    gt_file = f"cooperative-vehicle-infrastructure/cooperative-trajectories/train/{scene_id}.csv"

    try:
        # 1. 加载并预处理数据
        print("加载轨迹数据...")
        vehicle_df_full = pd.read_csv(vehicle_file) 
        vehicle_df = load_and_preprocess_data(vehicle_file)
        roadside_df = load_and_preprocess_data(roadside_file)

        # 获取tag不为OTHERS的车端ID集合
        target_car_ids = set(vehicle_df_full[vehicle_df_full['tag'] != 'OTHERS']['id'].unique())
        print(f"   Tag不为OTHERS的车端ID数量: {len(target_car_ids)}")

        # 只保留需要的列
        vehicle_df = vehicle_df[['timestamp', 'id', 'type', 'x', 'y']]
        roadside_df = roadside_df[['timestamp', 'id', 'type', 'x', 'y']]

        print(f"   车端轨迹数量: {len(vehicle_df['id'].unique())}")
        print(f"   路端轨迹数量: {len(roadside_df['id'].unique())}")

        # 2. 加载ground truth（只考虑tag不为OTHERS的目标）
        print("加载ground truth...")
        ground_truth = load_ground_truth(gt_file, vehicle_file)
        print(f"   Ground truth匹配对数量: {len(ground_truth)} (只包含tag不为OTHERS的目标)")

        # 3. 运行匹配算法
        predictions = match_trajectories(vehicle_df, roadside_df)
        print(f"  预测匹配对数量: {len(predictions)}")

        # 4. 评估结果（只评估tag不为OTHERS的车端ID）
        results = evaluate_matching(predictions, ground_truth, target_car_ids)

        # 5. 输出结果
        print("\n" + "="*50)
        print("验证结果:")
        print(f"匹配准确率 (Matching Accuracy): {results['matching_accuracy']:.4f}")
        print(f"精确率 (Precision): {results['precision']:.4f}")
        print(f"召回率 (Recall): {results['recall']:.4f}")
        print(f"F1分数: {results['f1_score']:.4f}")
        print(f"正确匹配数: {results['correct_matches']}/{results['total_gt_matches']}")
        print("="*50)

        # 6. 详细分析（只显示tag不为OTHERS的目标）

        # 过滤predictions，只保留target_car_ids中的
        filtered_predictions = {car_id: road_id for car_id, road_id in predictions.items() 
                               if car_id in target_car_ids}

        # 显示匹配成功的例子
        successful_matches = [(car_id, road_id) for car_id, road_id in filtered_predictions.items()
                             if ground_truth.get(car_id) == road_id]
        print(f"✅ 匹配成功的轨迹对 (前5个): {successful_matches[:5]}")

        # 显示匹配失败的例子
        failed_predictions = [(car_id, pred_road_id, ground_truth.get(car_id))
                             for car_id, pred_road_id in filtered_predictions.items()
                             if ground_truth.get(car_id) != pred_road_id]
        print(f"❌ 匹配错误的轨迹对 (前5个): {failed_predictions[:5]}")

        # 显示未匹配的ground truth
        unmatched_gt = [(car_id, road_id) for car_id, road_id in ground_truth.items()
                       if car_id not in filtered_predictions]
        print(f"🤷 未匹配的ground truth (前5个): {unmatched_gt[:5]}")

        return results

    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        return None
    except Exception as e:
        print(f"验证过程中出错: {e}")
        return None

if __name__ == "__main__":
    # 验证场景4
    results = validate_on_scene(scene_id=18)

    if results:
        print("验证完成！")    
    else:
        print("\n 验证失败，请检查文件路径和数据格式。")