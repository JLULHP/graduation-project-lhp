#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monodepth2深度估计准确性评估脚本

使用标签文件直接评估Monodepth2的深度估计性能
"""

import os
import json
import numpy as np
import math
from typing import Dict, List, Tuple
import argparse
from collections import defaultdict
import time
import matplotlib.pyplot as plt

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, current_dir)

from position_transform_deep import PositionTransformDeep


def load_labels(label_path: str) -> List[Dict]:
    """加载标签数据"""
    with open(label_path, 'r') as f:
        labels = json.load(f)
    return labels


def calculate_pixel_position(obj: Dict) -> Tuple[float, float]:
    """
    计算对象的像素位置（使用2D框的底部中心点）
    Args:
        obj: 对象标签数据
    Returns:
        (u, v) 像素坐标
    """
    xmin = float(obj['2d_box']['xmin'])
    ymin = float(obj['2d_box']['ymin'])
    xmax = float(obj['2d_box']['xmax'])
    ymax = float(obj['2d_box']['ymax'])

    # 底部中心点
    u = (xmin + xmax) / 2.0
    v = ymax  # 底部y坐标

    return u, v


def compute_depth_errors(true_depth: float, pred_depth: float) -> Dict:
    """
    计算深度误差指标
    Args:
        true_depth: 真实深度
        pred_depth: 预测深度
    Returns:
        包含各种误差指标的字典
    """
    if true_depth <= 0 or pred_depth <= 0:
        return {
            'abs_rel': np.nan,
            'sq_rel': np.nan,
            'rmse': np.nan,
            'rmse_log': np.nan,
            'delta_1_25': np.nan,
            'delta_1_56': np.nan,
            'delta_1_95': np.nan,
            'true_depth': true_depth,
            'pred_depth': pred_depth,
            'error': np.nan
        }

    # 基本误差
    abs_rel = abs(true_depth - pred_depth) / true_depth
    sq_rel = ((true_depth - pred_depth) ** 2) / true_depth
    rmse = math.sqrt((true_depth - pred_depth) ** 2)
    rmse_log = math.sqrt((math.log(true_depth) - math.log(pred_depth)) ** 2)

    # 准确率指标
    ratio = max(true_depth / pred_depth, pred_depth / true_depth)
    delta_1_25 = ratio < 1.25
    delta_1_56 = ratio < 1.56**2
    delta_1_95 = ratio < 1.95**2

    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'delta_1_25': delta_1_25,
        'delta_1_56': delta_1_56,
        'delta_1_95': delta_1_95,
        'true_depth': true_depth,
        'pred_depth': pred_depth,
        'error': abs(true_depth - pred_depth)
    }


def evaluate_with_labels(image_dir: str, label_dir: str, model_name: str = 'mono+stereo_640x192',
                       max_images: int = None, use_cuda: bool = True) -> Dict:
    """
    使用标签文件评估Monodepth2的深度估计性能
    Args:
        image_dir: 图像文件目录
        label_dir: 标签文件目录
        model_name: Monodepth2模型名称
        max_images: 最大处理图像数量（用于快速测试）
        use_cuda: 是否使用CUDA
    Returns:
        评估结果字典
    """
    print("=== Monodepth2深度估计评估 ===")
    print(f"图像目录: {image_dir}")
    print(f"标签目录: {label_dir}")
    print(f"模型: {model_name}")
    print(f"使用CUDA: {use_cuda}")
    # 获取所有标签文件
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    if max_images:
        label_files = label_files[:max_images]
        print(f"限制处理图像数量: {max_images}")

    print(f"总共需要处理 {len(label_files)} 张图像")

    # 初始化Monodepth2模型
    print("\n初始化Monodepth2模型...")
    transformer = PositionTransformDeep(
        model_name=model_name,
        use_cuda=use_cuda,
        skip_config=True
    )
    transformer.load_monodepth2_model()

    # 存储所有误差结果
    all_errors = []
    errors_by_type = defaultdict(list)
    errors_by_distance = {
        '0-10m': [],
        '10-20m': [],
        '20-50m': [],
        '50m+': []
    }

    start_time = time.time()

    # 处理每张图像
    for idx, label_file in enumerate(label_files):
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"处理进度: {idx}/{len(label_files)} (耗时: {elapsed:.1f}s)")

        # 构建文件路径
        base_name = label_file.replace('.json', '')
        image_path = os.path.join(image_dir, f"{base_name}.jpg")
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(image_path):
            print(f"警告: 图像文件不存在: {image_path}")
            continue
        if not os.path.exists(label_path):
            print(f"警告: 标签文件不存在: {label_path}")
            continue

        try:
            # 加载标签数据
            labels = load_labels(label_path)
            # 处理每个对象
            for obj in labels:
                obj_type = obj['type']

                # 计算真实深度 (从3d_location.z获取)
                true_depth = abs(float(obj['3d_location']['z']))

                # 过滤异常值
                if true_depth < 0.1 or true_depth > 100:
                    continue

                # 计算像素位置 (从2d_box计算底部中心点)
                u, v = calculate_pixel_position(obj)

                # 预测深度
                pred_depth = transformer.estimate_depth_from_image(image_path, u, v)

                # 计算误差
                errors = compute_depth_errors(true_depth, pred_depth)
                errors['type'] = obj_type
                errors['image_id'] = idx

                # 存储结果
                all_errors.append(errors)
                errors_by_type[obj_type].append(errors)

                # 按距离范围分类
                if 0 < true_depth <= 10:
                    errors_by_distance['0-10m'].append(errors)
                elif 10 < true_depth <= 20:
                    errors_by_distance['10-20m'].append(errors)
                elif 20 < true_depth <= 50:
                    errors_by_distance['20-50m'].append(errors)
                else:
                    errors_by_distance['50m+'].append(errors)

        except Exception as e:
            print(f"处理图像 {label_file} 时出错: {e}")
            continue

    # 计算总体统计
    total_time = time.time() - start_time
    results = {
        'total_objects': len(all_errors),
        'total_images': len(label_files),
        'processing_time': total_time,
        'objects_per_second': len(all_errors) / total_time if total_time > 0 else 0,
        'overall_metrics': compute_statistics(all_errors),
        'metrics_by_type': {obj_type: compute_statistics(errors)
                           for obj_type, errors in errors_by_type.items()},
        'metrics_by_distance': {dist_range: compute_statistics(errors)
                               for dist_range, errors in errors_by_distance.items()},
        'raw_errors': all_errors
    }

    return results


def compute_statistics(errors: List[Dict]) -> Dict:
    """
    计算误差统计信息

    Args:
        errors: 误差列表

    Returns:
        统计结果字典
    """
    if not errors:
        return {}

    # 过滤有效的误差
    valid_errors = [e for e in errors if not np.isnan(e['abs_rel'])]

    if not valid_errors:
        return {}

    # 提取各种指标
    abs_rel = [e['abs_rel'] for e in valid_errors]
    sq_rel = [e['sq_rel'] for e in valid_errors]
    rmse = [e['rmse'] for e in valid_errors]
    rmse_log = [e['rmse_log'] for e in valid_errors]
    delta_1_25 = [e['delta_1_25'] for e in valid_errors]
    delta_1_56 = [e['delta_1_56'] for e in valid_errors]
    delta_1_95 = [e['delta_1_95'] for e in valid_errors]

    return {
        'count': len(valid_errors),
        'abs_rel': {
            'mean': np.mean(abs_rel),
            'median': np.median(abs_rel),
            'std': np.std(abs_rel),
            'min': np.min(abs_rel),
            'max': np.max(abs_rel)
        },
        'sq_rel': {
            'mean': np.mean(sq_rel),
            'median': np.median(sq_rel)
        },
        'rmse': {
            'mean': np.mean(rmse),
            'median': np.median(rmse)
        },
        'rmse_log': {
            'mean': np.mean(rmse_log),
            'median': np.median(rmse_log)
        },
        'accuracy': {
            'delta_1_25': np.mean(delta_1_25),  # 准确率
            'delta_1_56': np.mean(delta_1_56),
            'delta_1_95': np.mean(delta_1_95)
        }
    }


def print_results(results: Dict):
    """打印评估结果"""
    print("\n" + "="*60)
    print("MONODEPTH2深度估计评估结果")
    print("="*60)

    print(f"\n数据集统计:")
    print(f"  处理图像数量: {results['total_images']}")
    print(f"  处理对象数量: {results['total_objects']}")
    print(f"  处理时间: {results['processing_time']:.1f}秒")
    print(f"  处理速度: {results['objects_per_second']:.1f} objects/sec")

    # 总体指标
    overall = results['overall_metrics']
    if overall:
        print(f"\n总体性能指标:")
        print(f"  Abs Rel:  {overall['abs_rel']['mean']:.4f} (中位数: {overall['abs_rel']['median']:.4f})")
        print(f"  Sq Rel:   {overall['sq_rel']['mean']:.4f} (中位数: {overall['sq_rel']['median']:.4f})")
        print(f"  RMSE:     {overall['rmse']['mean']:.4f} (中位数: {overall['rmse']['median']:.4f})")
        print(f"  RMSE_log: {overall['rmse_log']['mean']:.4f}")
        print(f"  δ < 1.25: {overall['accuracy']['delta_1_25']:.3f}")
        print(f"  δ < 1.56²: {overall['accuracy']['delta_1_56']:.3f}")
        print(f"  δ < 1.95²: {overall['accuracy']['delta_1_95']:.3f}")
        print(f"  样本数量: {overall['count']}")

    # 按类型统计
    print(f"\n按对象类型统计:")
    for obj_type, metrics in results['metrics_by_type'].items():
        if metrics:
            count = metrics['count']
            abs_rel = metrics['abs_rel']['mean']
            acc = metrics['accuracy']['delta_1_25']
            print(f"  {obj_type:12s}: AbsRel={abs_rel:.4f}, δ<1.25={acc:.3f}, 数量={count}")

    # 按距离统计
    print(f"\n按距离范围统计:")
    for dist_range, metrics in results['metrics_by_distance'].items():
        if metrics:
            count = metrics['count']
            abs_rel = metrics['abs_rel']['mean']
            acc = metrics['accuracy']['delta_1_25']
            print(f"  {dist_range:8s}: AbsRel={abs_rel:.4f}, δ<1.25={acc:.3f}, 数量={count}")


def save_results(results: Dict, output_file: str):
    """保存结果到文件"""
    import json

    # 将numpy类型转换为Python类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Monodepth2深度估计评估工具')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='图像文件目录')
    parser.add_argument('--label_dir', type=str, required=True,
                       help='标签文件目录')
    parser.add_argument('--model_name', type=str,
                       default='mono+stereo_640x192',
                       choices=[
                           "mono_640x192", "stereo_640x192", "mono+stereo_640x192",
                           "mono_no_pt_640x192", "stereo_no_pt_640x192", "mono+stereo_no_pt_640x192",
                           "mono_1024x320", "stereo_1024x320", "mono+stereo_1024x320"],
                       help='Monodepth2模型名称')
    parser.add_argument('--max_images', type=int, default=None,
                       help='最大处理图像数量（用于快速测试）')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='输出目录')
    parser.add_argument('--no_cuda', action='store_true',
                       help='禁用CUDA，使用CPU')
    parser.add_argument('--save_plots', action='store_true',
                       help='保存误差分布图')
    args = parser.parse_args()

    # 运行评估
    results = evaluate_with_labels(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        model_name=args.model_name,
        max_images=args.max_images,
        use_cuda=not args.no_cuda
    )
    print_results(results)

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'evaluation_results.json')
    save_results(results, output_file)


if __name__ == '__main__':
    main()