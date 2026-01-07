#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标转换模块 - 单目测距实现
将像素坐标转换为世界坐标系坐标
"""

import numpy as np
import yaml
import math
import csv
import time
import os
from typing import List, Tuple, Optional


class PositionTransform:
    """坐标转换类 - 使用单目测距进行像素到世界坐标转换"""
    def __init__(self, config_dir: str = "config", skip_config: bool = False):
        """
        初始化坐标转换器

        Args:
            config_dir: 配置文件目录
            skip_config: 是否跳过加载配置文件（用于使用数据集参数的情况）
        """
        self.config_dir = config_dir
        
        # 机器人位姿
        self.robot_pose = {
            'x': 0.0,      # 机器人x坐标
            'y': 0.0,      # 机器人y坐标
            'z': 0.0,      # 机器人z坐标
            'yaw': 0.0     # 机器人朝向角（弧度）
        }
        
        # 初始化相机参数为None，等待从数据集加载
        self.camera_matrix = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.R = None
        self.t = None
        self.camera_height = None
        self.camera_pitch = None
        
        if not skip_config:
            self.load_camera_parameters()

    def load_camera_parameters(self):
        """加载相机参数"""
        try:
            # 加载相机位置参数
            cam_pos_path = os.path.join(self.config_dir, "cam_pos.yaml")
            with open(cam_pos_path, 'r', encoding='utf-8') as f:
                cam_pos = yaml.safe_load(f)

            self.camera_height = cam_pos['camera']['height']
            self.camera_pitch = np.radians(cam_pos['camera']['pitch'])

            # 加载相机标定参数
            calib_path = os.path.join(self.config_dir, "calibration_parameter.yaml")
            with open(calib_path, 'r', encoding='utf-8') as f:
                calib = yaml.safe_load(f)

            # 相机内参
            self.camera_matrix = np.array(calib['camera_matrix']['data']).reshape(3, 3)
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]

            # 相机外参
            self.R = np.array(calib['extrinsic_matrix']['rotation']).reshape(3, 3)
            self.t = np.array(calib['extrinsic_matrix']['translation'])

            print("相机参数加载成功")
            print(f"相机高度: {self.camera_height}m")
            print(f"相机俯仰角: {np.degrees(self.camera_pitch)}°")

        except FileNotFoundError as e:
            print(f"配置文件未找到: {e}")
            raise
        except Exception as e:
            print(f"加载相机参数失败: {e}")
            raise

    def update_robot_pose(self, x: float, y: float, z: float, yaw: float):
        """更新机器人位姿"""
        self.robot_pose['x'] = x
        self.robot_pose['y'] = y
        self.robot_pose['z'] = z
        self.robot_pose['yaw'] = yaw

    def estimate_depth(self, v: float) -> float:
        """
        使用地面假设估计深度
        Args:
            v: 图像坐标系中的y坐标（目标底部中心点）
        Returns:
            相机坐标系下的深度值Z_c
        """
        return self.camera_height / (math.tan(self.camera_pitch) + (v - self.cy) / self.fy)

    def pixel_to_world(self, u: float, v: float, z_c: float) -> Optional[np.ndarray]:
        """
        像素坐标到世界坐标转换
        Args:
            u, v: 像素坐标
            z_c: 相机坐标系下的深度
        Returns:
            世界坐标系坐标 [x, y, z] 或 None（如果机器人位姿无效）
        """
        # 相机坐标系计算
        x_c = (u - self.cx) * z_c / self.fx
        y_c = (v - self.cy) * z_c / self.fy
        # 转换到机器人坐标系
        camera_coords = np.array([x_c, y_c, z_c])
        robot_coords = np.dot(self.R, camera_coords) + self.t
        # 转换到世界坐标系
        yaw = self.robot_pose['yaw']
        rotation_matrix = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])
        # 世界坐标系变换
        translated = rotation_matrix.dot(robot_coords)
        world_coords = translated + np.array([
            self.robot_pose['x'],
            self.robot_pose['y'],
            self.robot_pose['z']
        ])
        return world_coords

    def transform_tracking_data(self, tracking_csv: str, output_csv: str):
        """
        处理跟踪数据并进行坐标转换
        Args:
            tracking_csv: 输入跟踪数据CSV文件路径
            output_csv: 输出世界坐标CSV文件路径
        """
        print(f"开始处理跟踪数据: {tracking_csv}")
        # 读取跟踪数据
        tracking_data = []
        with open(tracking_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tracking_data.append({
                    'track_id': int(row['track_id']),
                    'class': row['class'],
                    'x1': float(row['x1']),
                    'y1': float(row['y1']),
                    'x2': float(row['x2']),
                    'y2': float(row['y2']),
                    'center_x': float(row['center_x']),
                    'center_y': float(row['center_y']),
                    'frame_id': int(row['frame_id'])
                })

        print(f"共读取 {len(tracking_data)} 条跟踪记录")

        # 处理坐标转换
        world_positions = []
        for i, track in enumerate(tracking_data):
            if i % 1000 == 0:
                print(f"处理进度: {i}/{len(tracking_data)}")

            # 计算底部中心点（用于深度估计）
            bottom_center_x = (track['x1'] + track['x2']) / 2.0
            bottom_center_y = track['y2']  # 使用底部y坐标

            # 深度估计
            depth = self.estimate_depth(bottom_center_y)

            # 坐标转换
            world_coords = self.pixel_to_world(bottom_center_x, bottom_center_y, depth)

            if world_coords is not None:
                world_positions.append({
                    'track_id': track['track_id'],
                    'class': track['class'],
                    'frame_id': track['frame_id'],
                    'pixel_x': track['center_x'],
                    'pixel_y': track['center_y'],
                    'world_x': world_coords[0],
                    'world_y': world_coords[1],
                    'world_z': world_coords[2],
                    'depth': depth,
                    'timestamp': time.time()
                })

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['track_id', 'class', 'frame_id', 'pixel_x', 'pixel_y',
                         'world_x', 'world_y', 'world_z', 'depth', 'timestamp']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(world_positions)
        print(f"坐标转换完成，结果已保存到: {output_csv}")



def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='坐标转换工具')
    parser.add_argument('--tracking_csv', type=str,
                       default='ultralytics/runs/detect/track/tracking_results.csv',
                       help='输入跟踪数据CSV文件')
    parser.add_argument('--output_csv', type=str,
                       default='world_positions.csv',
                       help='输出世界坐标CSV文件')
    parser.add_argument('--config_dir', type=str,
                       default='config',
                       help='配置文件目录')
    parser.add_argument('--robot_x', type=float, default=0.0,
                       help='机器人x坐标')
    parser.add_argument('--robot_y', type=float, default=0.0,
                       help='机器人y坐标')
    parser.add_argument('--robot_yaw', type=float, default=0.0,
                       help='机器人朝向角（度）')
    parser.add_argument('--create_config', action='store_true',
                       help='创建默认配置文件')

    args = parser.parse_args()


    # 初始化坐标转换器
    try:
        transformer = PositionTransform(args.config_dir)
    except FileNotFoundError:
        print("配置文件不存在，请先运行 --create_config 创建默认配置")
        return

    # 设置机器人位姿
    transformer.update_robot_pose(
        x=args.robot_x,
        y=args.robot_y,
        z=0.0,
        yaw=np.radians(args.robot_yaw)
    )
    # 执行坐标转换
    transformer.transform_tracking_data(args.tracking_csv, args.output_csv)


if __name__ == '__main__':
    main()
