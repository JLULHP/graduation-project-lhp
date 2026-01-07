#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import rospy
import numpy as np
import yaml
import math
from nav_msgs.msg import Odometry
from yolo_tracking.msg import TrackingArray
from position_transform.msg import WorldPositionArray, WorldPosition
import tf.transformations as tf_trans

class PositionTransform(object):
    def __init__(self):
        rospy.init_node('position_transform_node', anonymous=True)
        
        # 加载相机参数‌:ml-citation{ref="6" data="citationList"}
        self.load_camera_parameters()

        # 初始化订阅者和发布者
        self.tracking_sub = rospy.Subscriber("/yolo_tracking/tracks", TrackingArray, self.tracking_callback)
        self.odom_sub = rospy.Subscriber("/udp_odom", Odometry, self.odom_callback)
        self.world_pos_pub = rospy.Publisher("/world_positions", WorldPositionArray, queue_size=10)

        # 存储器人位姿和世界位置消息
        self.current_robot_pose = None
        self.latest_world_pos_msg = None

        # 创建定时器，指定发布频率‌:ml-citation{ref="4" data="citationList"}
        publish_rate = rospy.get_param('~publish_rate', 10) # 10Hz
        self.timer = rospy.Timer(rospy.Duration(1.0/publish_rate), self.timer_callback)

        rospy.loginfo("Position Transform Node initialized")

    def load_camera_parameters(self):
        # 读取相机位置参数‌:ml-citation{ref="6" data="citationList"}
        cam_pos_path = rospy.get_param('~cam_pos_path', 'src/position_transform/config/cam_pos.yaml')
        with open(cam_pos_path, 'r') as f:
            cam_pos = yaml.safe_load(f)
        self.camera_height = cam_pos['camera']['height']
        self.camera_pitch = np.radians(cam_pos['camera']['pitch'])

        # 读取相机标定参数‌:ml-citation{ref="2" data="citationList"}
        calib_path = rospy.get_param('~calib_path', 'src/position_transform/config/calibration_parameter.yaml')
        with open(calib_path, 'r') as f:
            calib = yaml.safe_load(f)

        # 相机内参‌:ml-citation{ref="6" data="citationList"}
        self.camera_matrix = np.array(calib['camera_matrix']['data']).reshape(3, 3)
        self.fx = self.camera_matrix[0, 0]
        self.fy = self.camera_matrix[1, 1]
        self.cx = self.camera_matrix[0, 2]
        self.cy = self.camera_matrix[1, 2]

        # 相机外参‌:ml-citation{ref="2" data="citationList"}
        self.R = np.array(calib['extrinsic_matrix']['rotation']).reshape(3, 3)
        self.t = np.array(calib['extrinsic_matrix']['translation'])

    def odom_callback(self, msg):
        # 更新机器人位姿‌:ml-citation{ref="4" data="citationList"}
        self.current_robot_pose = msg

    def estimate_depth(self, v):
        """使用地面假设估计深度‌:ml-citation{ref="6" data="citationList"}
        v: 图像坐标系中的y坐标（目标底部中心点）
        返回：相机坐标系下的深度值Z_c
        """
        return self.camera_height / (math.tan(self.camera_pitch) + (v - self.cy)/self.fy)

    def pixel_to_world(self, u, v, z_c):
        """像素坐标到世界坐标转换‌:ml-citation{ref="2,6" data="citationList"}
        参数：
            u,v: 像素坐标
            z_c: 相机坐标系下的深度
        返回：
            world_coords: 世界坐标系坐标
        """
        # 相机坐标系计算
        x_c = (u - self.cx) * z_c / self.fx
        y_c = (v - self.cy) * z_c / self.fy

        # 转换到机器人坐标系
        camera_coords = np.array([x_c, y_c, z_c])
        robot_coords = np.dot(self.R, camera_coords) + self.t

        # 转换到世界坐标系‌:ml-citation{ref="4" data="citationList"}
        if self.current_robot_pose:
            q = self.current_robot_pose.pose.pose.orientation
            _, _, yaw = tf_trans.euler_from_quaternion([q.x, q.y, q.z, q.w])
            
            rotation_matrix = np.array([
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1]
            ])
            
            # 世界坐标系变换‌:ml-citation{ref="2" data="citationList"}
            translated = rotation_matrix.dot(robot_coords)
            world_coords = translated + np.array([
                self.current_robot_pose.pose.pose.position.x,
                self.current_robot_pose.pose.pose.position.y,
                self.current_robot_pose.pose.pose.position.z
            ])
            return world_coords
        else:
            return None

    def tracking_callback(self, msg):
        world_positions = []
        for track in msg.tracks:
            # 计算底部中心点‌:ml-citation{ref="6" data="citationList"}
            bottom_center_x = (track.x1 + track.x2) / 2.0
            bottom_center_y = track.y2
            
            # 深度估计
            depth = self.estimate_depth(bottom_center_y)
            
            # 坐标转换
            world_coords = self.pixel_to_world(bottom_center_x, bottom_center_y, depth)
            
            if world_coords is not None:
                wp = WorldPosition()
                wp.id = track.id
                wp.x = world_coords
                wp.y = world_coords‌:ml-citation{ref="1" data="citationList"}
                wp.z = world_coords‌:ml-citation{ref="2" data="citationList"}
                world_positions.append(wp)
        
        self.latest_world_pos_msg = WorldPositionArray(positions=world_positions)

    def timer_callback(self, event):
        if self.latest_world_pos_msg:
            self.world_pos_pub.publish(self.latest_world_pos_msg)

if __name__ == '__main__':
    try:
        pt = PositionTransform()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
