#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单图像Monodepth2深度估计验证脚本
"""
import os
import json
import numpy as np
import math
from typing import Dict, List, Tuple
import argparse
import time
import matplotlib.pyplot as plt
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, current_dir)
from position_transform_deep import PositionTransformDeep


def load_labels(label_path: str) -> List[Dict]:
    with open(label_path, 'r') as f:
        return json.load(f)


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
    return (xmin + xmax) / 2.0, ymax


def compute_depth_errors(true_depth: float, pred_depth: float) -> Dict:
    """
    计算深度误差指标

    Args:
        true_depth: 真实深度
        pred_depth: 预测深度

    Returns:
        包含各种误差指标的字典
    """
    if true_depth <= 0 or pred_depth <= 0 or not np.isfinite(true_depth) or not np.isfinite(pred_depth):
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
            'error': np.nan,
            'valid': False
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
        'error': abs(true_depth - pred_depth),
        'ratio': ratio,
        'valid': True
    }


def evaluate_single_image(image_path: str, label_path: str, model_name: str = 'mono+stereo_640x192',
                         use_cuda: bool = True, save_visualization: bool = True, disable_median_scaling: bool = False):
    """
    验证单张图像的Monodepth2深度估计
    Args:
        image_path: 图像文件路径
        label_path: 标签文件路径
        model_name: Monodepth2模型名称
        use_cuda: 是否使用CUDA
        save_visualization: 是否保存可视化结果
    Returns:
        验证结果字典
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"标签文件不存在: {label_path}")

    transformer = PositionTransformDeep(
        config_dir=None,
        model_name=model_name,
        use_cuda=use_cuda,
        skip_config=True
    )

    image = Image.open(image_path)
    labels = load_labels(label_path)


    all_raw_predictions = []
    scale_ratios = []
    valid_objects = 0
    start_time = time.time()

    for obj in labels:
        try:
            true_depth = abs(float(obj['3d_location']['x']))
            if true_depth < 0.1 or true_depth > 100:
                continue

            pixel_x, pixel_y = calculate_pixel_position(obj)
            if not (0 <= pixel_x < image.size[0] and 0 <= pixel_y < image.size[1]):
                continue

            pred_depth_raw = transformer.estimate_depth_from_image(image_path, pixel_x, pixel_y)

            if pred_depth_raw > 0:
                ratio = true_depth / pred_depth_raw
                scale_ratios.append(ratio)

            all_raw_predictions.append({
                'obj': obj,
                'true_depth': true_depth,
                'pred_depth_raw': pred_depth_raw,
                'pixel_x': pixel_x,
                'pixel_y': pixel_y
            })

        except Exception:
            continue

    optimal_scale_factor = np.median(scale_ratios) if scale_ratios and not disable_median_scaling else 10.5

    all_errors = []
    for pred_data in all_raw_predictions:
        obj = pred_data['obj']
        true_depth = pred_data['true_depth']
        pred_depth_raw = pred_data['pred_depth_raw']

        pred_depth = pred_depth_raw * optimal_scale_factor

        errors = compute_depth_errors(true_depth, pred_depth)
        if errors['valid']:
            valid_objects += 1
            all_errors.append(errors)


    total_time = time.time() - start_time

    if valid_objects > 0:
        valid_errors = [e for e in all_errors if e['valid']]
        abs_rel_mean = np.mean([e['abs_rel'] for e in valid_errors])
        sq_rel_mean = np.mean([e['sq_rel'] for e in valid_errors])
        rmse_mean = np.mean([e['rmse'] for e in valid_errors])
        rmse_log_mean = np.mean([e['rmse_log'] for e in valid_errors])
        delta_1_25_mean = np.mean([e['delta_1_25'] for e in valid_errors])
        delta_1_56_mean = np.mean([e['delta_1_56'] for e in valid_errors])
        delta_1_95_mean = np.mean([e['delta_1_95'] for e in valid_errors])

        print(f"\nValid objects: {valid_objects}/{len(labels)}")
        print(f"Scale factor: {optimal_scale_factor:.3f}")
        print(".4f")
        print(".4f")
        print(".3f")

    if valid_objects > 0 and save_visualization:
        create_detection_image(image_path, labels, all_errors)

    # 返回结果
    result = {
        'image_path': image_path,
        'label_path': label_path,
        'total_objects': len(labels),
        'valid_objects': valid_objects,
        'errors': all_errors
    }

    if valid_objects > 0:
        result.update({
            'abs_rel_mean': abs_rel_mean,
            'sq_rel_mean': sq_rel_mean,
            'rmse_mean': rmse_mean,
            'rmse_log_mean': rmse_log_mean,
            'delta_1_25_mean': delta_1_25_mean,
            'delta_1_56_mean': delta_1_56_mean,
            'delta_1_95_mean': delta_1_95_mean,
            'optimal_scale_factor': optimal_scale_factor,
            'scale_ratios_count': len(scale_ratios)
        })

    return result


def create_detection_image(image_path: str, labels: List[Dict], errors: List[Dict]):
    """创建原始图像及检测框的可视化（左上角子图）"""
    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.imshow(image)
        ax.set_title('Detection Results', fontsize=16)
        ax.axis('off')

        # 在图像上绘制检测框和深度信息
        for i, obj in enumerate(labels):
            box = obj['2d_box']
            xmin, ymin, xmax, ymax = float(box['xmin']), float(box['ymin']), float(box['xmax']), float(box['ymax'])

            # 计算真实深度 (使用x轴作为深度信息)
            true_depth = abs(float(obj['3d_location']['x']))

            # 查找对应的预测结果
            pred_depth = None
            error_info = None
            for error in errors:
                if abs(error['true_depth'] - true_depth) < 0.001:
                    pred_depth = error['pred_depth']
                    error_info = error
                    break

            # 只有当有预测结果时才绘制
            if pred_depth is not None and error_info and error_info['valid']:
                # 根据误差大小选择颜色
                abs_rel = error_info['abs_rel']
                if abs_rel < 0.1:
                    box_color = (0, 255, 0)  # 绿色 - 优秀
                elif abs_rel < 0.25:
                    box_color = (255, 255, 0)  # 黄色 - 良好
                elif abs_rel < 0.5:
                    box_color = (255, 165, 0)  # 橙色 - 一般
                else:
                    box_color = (255, 0, 0)  # 红色 - 较差

                # 绘制边界框 (转换为RGB颜色)
                rgb_color = tuple(c/255.0 for c in box_color)
                rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                   fill=False, color=rgb_color, linewidth=2)
                ax.add_patch(rect)

                # 添加对象类型和深度信息
                obj_type = obj['type']
                label_text = f'{obj_type}\nTrue: {true_depth:.2f}m\nPred: {pred_depth:.2f}m'

                # 根据框的颜色选择文字颜色
                if abs_rel < 0.1:
                    text_color = 'green'
                elif abs_rel < 0.25:
                    text_color = 'orange'
                elif abs_rel < 0.5:
                    text_color = 'red'
                else:
                    text_color = 'red'

                ax.text(xmin, ymin-5, label_text, color=text_color,
                       fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        plt.tight_layout()

        # 保存图像为evaluation_single_{图像名}.png
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_file = f'evaluation_single_{base_name}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved detection image: {output_file}")

    except Exception:
        pass


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='单图像Monodepth2深度估计验证')
    parser.add_argument('--image', type=str, required=True,
                       help='图像文件路径')
    parser.add_argument('--label', type=str, required=True,
                       help='标签文件路径')
    parser.add_argument('--model_name', type=str,
                       default='mono+stereo_640x192',
                       help='Monodepth2模型名称')
    parser.add_argument('--no_cuda', action='store_true',
                       help='禁用CUDA，使用CPU')
    parser.add_argument('--no_vis', action='store_true',
                       help='不生成可视化结果')
    parser.add_argument('--disable_median_scaling', action='store_true',
                       help='禁用median scaling，使用默认尺度因子')

    args = parser.parse_args()

    # 运行验证
    evaluate_single_image(
        image_path=args.image,
        label_path=args.label,
        model_name=args.model_name,
        use_cuda=not args.no_cuda,
        save_visualization=not args.no_vis,
        disable_median_scaling=args.disable_median_scaling
    )


if __name__ == '__main__':
    main()