#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坐标转换模块 - 使用Monodepth2进行深度估计
将像素坐标转换为世界坐标系坐标
"""

import numpy as np
import yaml
import math
import csv
import time
import os
import sys
from typing import List, Tuple, Optional
import PIL.Image as pil
import torch
from torchvision import transforms

# 添加monodepth2路径到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
monodepth2_path = os.path.join(current_dir, "monodepth2")
if monodepth2_path not in sys.path:
    sys.path.insert(0, monodepth2_path)

# 导入monodepth2模块
import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

# 从networks模块获取类
ResnetEncoder = networks.ResnetEncoder
DepthDecoder = networks.DepthDecoder


class PositionTransformDeep:
    """坐标转换类 - 使用Monodepth2进行深度估计的像素到世界坐标转换"""

    def __init__(self, config_dir: str = "config",
                 model_name: str = "mono+stereo_640x192",
                 model_path: str = None,
                 use_cuda: bool = True,
                 skip_config: bool = False,
                 calib_dir: str = None):
        """
        初始化坐标转换器

        Args:
            config_dir: 配置文件目录
            model_name: Monodepth2模型名称
            model_path: 模型路径（如果为None，则使用models目录下的模型）
            use_cuda: 是否使用CUDA
            skip_config: 是否跳过加载配置文件（用于使用数据集参数的情况）
            calib_dir: 数据集标定文件目录（如果提供，将从数据集文件加载相机参数）
        """
        self.config_dir = config_dir
        self.calib_dir = calib_dir
        self.model_name = model_name
        self.model_path = model_path or os.path.join(current_dir, "models", model_name)
        self.use_cuda = use_cuda and torch.cuda.is_available()

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

        # Monodepth2相关参数
        self.encoder = None
        self.depth_decoder = None
        self.feed_height = None
        self.feed_width = None
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        if not skip_config:
            self.load_camera_parameters()

        # 总是加载Monodepth2模型（深度估计不需要相机参数）
        self.load_monodepth2_model()

    def load_camera_parameters(self):
        """加载相机参数"""
        try:
            # 如果指定了calib_dir，从数据集标定文件中加载
            if self.calib_dir:
                self._load_camera_parameters_from_dataset()
            else:
                # 否则从默认config目录加载
                self._load_camera_parameters_from_config()

        except Exception as e:
            print(f"加载相机参数失败: {e}")
            raise

    def _load_camera_parameters_from_dataset(self):
        """从数据集标定文件中加载相机参数"""
        try:
            # 加载相机内参
            calib_json_path = os.path.join(self.calib_dir, "camera_intrinsic", "000036.json")
            if not os.path.exists(calib_json_path):
                # 对于infrastructure数据集，尝试000000.json
                calib_json_path = os.path.join(self.calib_dir, "camera_intrinsic", "000000.json")

            if os.path.exists(calib_json_path):
                with open(calib_json_path, 'r') as f:
                    calib_data = json.load(f)
                self.camera_matrix = np.array(calib_data['cam_K']).reshape(3, 3)
            else:
                raise FileNotFoundError(f"找不到相机内参文件: {calib_json_path}")

            # 加载相机姿态信息（从lidar_to_camera或virtuallidar_to_camera）
            pose_json_path = os.path.join(self.calib_dir, "lidar_to_camera", "000036.json")
            if not os.path.exists(pose_json_path):
                # 对于infrastructure数据集，尝试virtuallidar_to_camera
                pose_json_path = os.path.join(self.calib_dir, "virtuallidar_to_camera", "000000.json")

            if os.path.exists(pose_json_path):
                with open(pose_json_path, 'r') as f:
                    pose_data = json.load(f)

                # 从平移向量提取相机高度 (Z分量)
                translation = np.array(pose_data['translation']).flatten()
                self.camera_height = abs(translation[2])

                # 从旋转矩阵提取相机俯仰角
                rotation = np.array(pose_data['rotation'])
                # 俯仰角是从旋转矩阵的特定元素计算得出的
                # 对于相机坐标系，俯仰角可以通过arcsin(-R[2,0])获得
                pitch_rad = -np.arcsin(rotation[2, 0])
                self.camera_pitch = pitch_rad
            else:
                # 如果没有姿态信息，使用默认值
                self.camera_height = 1.2
                self.camera_pitch = np.radians(-15.0)

            # 设置相机内参
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]

            # 外参暂时设为空（如果需要的话）
            self.R = np.eye(3)
            self.t = np.zeros(3)

            print("从数据集标定文件加载相机参数成功")
            print(f"相机高度: {self.camera_height:.3f}m")
            print(f"相机俯仰角: {np.degrees(self.camera_pitch):.1f}°")

        except Exception as e:
            print(f"从数据集加载相机参数失败: {e}")
            raise

    def _load_camera_parameters_from_config(self):
        """从默认config目录加载相机参数"""
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

            print("从默认配置加载相机参数成功")
            print(f"相机高度: {self.camera_height}m")
            print(f"相机俯仰角: {np.degrees(self.camera_pitch)}°")

        except FileNotFoundError as e:
            print(f"配置文件未找到: {e}")
            raise
        except Exception as e:
            print(f"从配置加载相机参数失败: {e}")
            raise

    def load_monodepth2_model(self):
        """加载Monodepth2模型"""
        try:
            print(f"正在加载Monodepth2模型: {self.model_name}")

            # 下载模型（如果不存在）
            download_model_if_doesnt_exist(self.model_name)

            # 模型路径
            encoder_path = os.path.join(self.model_path, "encoder.pth")
            depth_decoder_path = os.path.join(self.model_path, "depth.pth")

            if not os.path.exists(encoder_path) or not os.path.exists(depth_decoder_path):
                raise FileNotFoundError(f"模型文件不存在: {encoder_path} 或 {depth_decoder_path}")

            # 加载编码器
            print("  加载编码器...")
            self.encoder = ResnetEncoder(18, False)
            loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

            # 获取训练时使用的图像尺寸
            self.feed_height = loaded_dict_enc['height']
            self.feed_width = loaded_dict_enc['width']

            # 过滤字典，只加载匹配的参数
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(filtered_dict_enc)
            self.encoder.to(self.device)
            self.encoder.eval()

            # 加载解码器
            print("  加载解码器...")
            self.depth_decoder = DepthDecoder(
                num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

            loaded_dict = torch.load(depth_decoder_path, map_location=self.device)
            self.depth_decoder.load_state_dict(loaded_dict)
            self.depth_decoder.to(self.device)
            self.depth_decoder.eval()

            print(f"Monodepth2模型加载成功 (分辨率: {self.feed_width}x{self.feed_height})")
            print(f"使用设备: {self.device}")

        except Exception as e:
            print(f"加载Monodepth2模型失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def estimate_depth_from_image(self, image_path: str, u: float, v: float) -> float:
        """
        从图像和像素坐标估计深度

        Args:
            image_path: 图像文件路径
            u, v: 像素坐标
        Returns:
            深度值（米）
        """
        if self.encoder is None or self.depth_decoder is None:
            raise RuntimeError("Monodepth2模型未加载")

        try:
            # 加载和预处理图像
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size

            # 调整图像尺寸
            input_image_resized = input_image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
            input_tensor = self.image_transform(input_image_resized).unsqueeze(0).to(self.device)

            # 推理深度
            with torch.no_grad():
                features = self.encoder(input_tensor)
                outputs = self.depth_decoder(features)

                # 获取视差图
                disp = outputs[("disp", 0)]

                # 转换为深度（在原始视差图尺寸上计算）
                scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

                # 将深度图调整回原始图像尺寸
                depth_resized = torch.nn.functional.interpolate(
                    depth, (original_height, original_width), mode="bilinear", align_corners=False)

                # 获取指定像素的深度值
                depth_map = depth_resized.squeeze().cpu().numpy()

                # 保存调试信息
                self._debug_depth_map = depth_resized.squeeze().cpu().numpy()

                # 确保坐标在有效范围内
                u_int = max(0, min(int(u), original_width - 1))
                v_int = max(0, min(int(v), original_height - 1))

                depth_value = depth_map[v_int, u_int]

                return float(depth_value)

        except Exception as e:
            print(f"深度估计失败: {e}")
            # 返回默认深度作为fallback
            return self.camera_height / math.tan(abs(self.camera_pitch))

    def update_robot_pose(self, x: float, y: float, z: float, yaw: float):
        """更新机器人位姿"""
        self.robot_pose['x'] = x
        self.robot_pose['y'] = y
        self.robot_pose['z'] = z
        self.robot_pose['yaw'] = yaw

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

    def transform_tracking_data(self, tracking_csv: str, image_dir: str, output_csv: str):
        """
        处理跟踪数据并进行坐标转换
        Args:
            tracking_csv: 输入跟踪数据CSV文件路径
            image_dir: 图像文件目录
            output_csv: 输出世界坐标CSV文件路径
        """
        print(f"开始处理跟踪数据: {tracking_csv}")
        print(f"图像目录: {image_dir}")

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
            if i % 100 == 0:  # 更频繁的进度报告，因为深度估计更耗时
                print(f"处理进度: {i}/{len(tracking_data)}")

            # 计算底部中心点（用于深度估计）
            bottom_center_x = (track['x1'] + track['x2']) / 2.0
            bottom_center_y = track['y2']  # 使用底部y坐标

            # 构建图像路径
            image_name = f"{track['frame_id']:06d}.jpg"  # 假设图像命名格式
            image_path = os.path.join(image_dir, image_name)

            if not os.path.exists(image_path):
                print(f"警告: 图像文件不存在: {image_path}，跳过此条记录")
                continue

            # 使用Monodepth2估计深度
            depth = self.estimate_depth_from_image(image_path, bottom_center_x, bottom_center_y)

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

        # 保存结果
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['track_id', 'class', 'frame_id', 'pixel_x', 'pixel_y',
                         'world_x', 'world_y', 'world_z', 'depth', 'timestamp']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(world_positions)

        print(f"坐标转换完成，结果已保存到: {output_csv}")
        print(f"成功转换 {len(world_positions)} 条记录")


def create_default_config():
    """创建默认配置文件"""
    config_dir = "config"
    os.makedirs(config_dir, exist_ok=True)

    # 创建相机位置配置文件
    cam_pos_config = {
        'camera': {
            'height': 1.2,  # 相机高度（米）
            'pitch': -15    # 俯仰角（度）
        }
    }

    with open(os.path.join(config_dir, "cam_pos.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(cam_pos_config, f, default_flow_style=False)

    # 创建相机标定参数文件（示例参数）
    calib_config = {
        'camera_matrix': {
            'data': [800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0]
        },
        'extrinsic_matrix': {
            'rotation': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            'translation': [0.0, 0.0, 0.0]
        }
    }

    with open(os.path.join(config_dir, "calibration_parameter.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(calib_config, f, default_flow_style=False)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='深度坐标转换工具 (使用Monodepth2)')
    parser.add_argument('--tracking_csv', type=str,
                       default='ultralytics/runs/detect/track/tracking_results.csv',
                       help='输入跟踪数据CSV文件')
    parser.add_argument('--image_dir', type=str,
                       default='images',
                       help='图像文件目录')
    parser.add_argument('--output_csv', type=str,
                       default='world_positions_deep.csv',
                       help='输出世界坐标CSV文件')
    parser.add_argument('--config_dir', type=str,
                       default='config',
                       help='配置文件目录')
    parser.add_argument('--model_name', type=str,
                       default='mono+stereo_640x192',
                       choices=[
                           "mono_640x192",
                           "stereo_640x192",
                           "mono+stereo_640x192",
                           "mono_no_pt_640x192",
                           "stereo_no_pt_640x192",
                           "mono+stereo_no_pt_640x192",
                           "mono_1024x320",
                           "stereo_1024x320",
                           "mono+stereo_1024x320"],
                       help='Monodepth2模型名称')
    parser.add_argument('--robot_x', type=float, default=0.0,
                       help='机器人x坐标')
    parser.add_argument('--robot_y', type=float, default=0.0,
                       help='机器人y坐标')
    parser.add_argument('--robot_yaw', type=float, default=0.0,
                       help='机器人朝向角（度）')
    parser.add_argument('--no_cuda', action='store_true',
                       help='禁用CUDA，使用CPU')
    parser.add_argument('--create_config', action='store_true',
                       help='创建默认配置文件')

    args = parser.parse_args()

    if args.create_config:
        create_default_config()
        return

    # 初始化坐标转换器
    try:
        transformer = PositionTransformDeep(
            config_dir=args.config_dir,
            model_name=args.model_name,
            use_cuda=not args.no_cuda
        )
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
    transformer.transform_tracking_data(args.tracking_csv, args.image_dir, args.output_csv)


if __name__ == '__main__':
    main()
