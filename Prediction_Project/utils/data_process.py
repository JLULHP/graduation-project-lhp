import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from collections import defaultdict
from tqdm import tqdm
import pickle


class MultiAgentTrajectoryDataset(Dataset):
    """多智能体轨迹数据集类"""
    
    def __init__(self, data_dir, obs_len=20, pred_len=80, min_agents=1, max_agents=50, 
                 delim=',', agent_types=None, time_step=0.1, min_seq_len=60, cache_file=None):
        """
        初始化多智能体轨迹数据集
        
        参数:
            data_dir: 数据目录
            obs_len: 观测序列长度
            pred_len: 预测序列长度
            min_agents: 每个场景最少智能体数量
            max_agents: 每个场景最多智能体数量
            delim: CSV文件分隔符
            agent_types: 要包含的智能体类型列表，如['VEHICLE', 'BICYCLE']，None表示包含所有类型
            time_step: 时间步长，用于处理不连续的时间戳
            min_seq_len: 最小可接受的序列长度，至少要有观测+一些预测
            cache_file: 缓存文件路径，如果提供且文件存在，将从缓存加载数据
        """
        super(MultiAgentTrajectoryDataset, self).__init__()
        
        self.data_dir = data_dir
        self.default_obs_len = obs_len
        self.default_pred_len = pred_len
        self.min_seq_len = min_seq_len  # 最小可接受的序列长度
        self.min_agents = min_agents
        self.max_agents = max_agents
        self.delim = delim
        self.agent_types = agent_types
        self.time_step = time_step
        
        # 尝试从缓存加载数据
        if cache_file and os.path.exists(cache_file):
            print(f"从缓存文件加载数据: {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.scenes = cache_data['scenes']
                self.scene_agent_counts = cache_data['scene_agent_counts']
            print(f"共加载了 {len(self.scenes)} 个场景，包含 {sum(self.scene_agent_counts)} 个轨迹")
            return
        
        # 加载所有数据
        self.scenes = []  # 存储所有场景数据
        self.scene_agent_counts = []  # 每个场景中的智能体数量
        
        # 获取所有CSV文件
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        
        # 处理每个文件
        print(f"正在加载数据，共 {len(all_files)} 个文件...")
        for file_path in tqdm(all_files, desc="处理数据文件", mininterval=5.0):
            self._process_file(file_path)
        
        print(f"共加载了 {len(self.scenes)} 个场景，包含 {sum(self.scene_agent_counts)} 个轨迹")
        
        # 保存到缓存
        if cache_file:
            print(f"保存数据到缓存文件: {cache_file}")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'scenes': self.scenes,
                    'scene_agent_counts': self.scene_agent_counts
                }, f)
    
    def _process_file(self, path):
        """处理单个CSV文件并提取场景"""
        try:
            # 读取CSV文件
            df = pd.read_csv(path, delimiter=self.delim)
            
            required_columns = ['timestamp', 'id', 'x', 'y']
            if not all(col in df.columns for col in required_columns):
                return
            
            # 如果指定了智能体类型，则过滤数据
            if self.agent_types is not None and 'type' in df.columns:
                df = df[df['type'].isin(self.agent_types)]
            
            # 获取所有唯一的时间戳并排序
            timestamps = sorted(df['timestamp'].unique())
            
            # 如果时间戳数量小于最小序列长度，跳过此文件
            if len(timestamps) < self.min_seq_len:
                return
            
            # 动态确定观测和预测长度
            total_len = len(timestamps)
            if total_len >= self.default_obs_len + self.default_pred_len:
                obs_len = self.default_obs_len
                pred_len = self.default_pred_len
            else:
                obs_len = min(self.default_obs_len, total_len // 2)
                pred_len = total_len - obs_len
            
            # 对于每个可能的起始时间戳，创建一个场景
            for start_idx in range(len(timestamps) - (obs_len + pred_len) + 1):
                scene_timestamps = timestamps[start_idx:start_idx + obs_len + pred_len]
                
                # 按智能体ID组织数据
                agent_trajectories = defaultdict(list)
                agent_types = {}
                
                # 收集每个智能体在这些时间戳的轨迹
                for ts in scene_timestamps:
                    ts_data = df[df['timestamp'] == ts]
                    
                    for _, row in ts_data.iterrows():
                        agent_id = row['id']
                        agent_trajectories[agent_id].append([row['x'], row['y']])
                        if 'type' in row and agent_id not in agent_types:
                            agent_types[agent_id] = row['type']
                
                # 过滤掉观测序列不足的智能体
                complete_agents = {}
                for agent_id, traj in agent_trajectories.items():
                    traj_len = len(traj)
                    if traj_len >= obs_len:
                        # 只要有完整的观测序列就保留
                        if traj_len >= obs_len + pred_len:
                            # 有完整观测+预测序列
                            complete_agents[agent_id] = traj[:obs_len + pred_len]
                        else:
                            obs_part = traj[:obs_len]
                            pred_part = traj[obs_len:]
                            while len(pred_part) < pred_len:
                                pred_part.append(traj[-1])  
                            complete_agents[agent_id] = obs_part + pred_part
                    # 如果观测序列长度不足obs_len，直接跳过
                
                # 检查智能体数量是否在范围内
                if len(complete_agents) < self.min_agents or len(complete_agents) > self.max_agents:
                    continue
                
                # 创建场景数据
                scene_data = {
                    'obs': [],
                    'pred': [],
                    'agent_ids': [],
                    'agent_types': [],
                    'obs_len': obs_len,
                    'pred_len': pred_len
                }
                
                for agent_id, traj in complete_agents.items():
                    traj = np.array(traj)
                    scene_data['obs'].append(traj[:obs_len])
                    scene_data['pred'].append(traj[obs_len:])
                    scene_data['agent_ids'].append(agent_id)
                    scene_data['agent_types'].append(agent_types.get(agent_id, 'UNKNOWN'))
                
                # 将场景数据转换为numpy数组
                scene_data['obs'] = np.array(scene_data['obs'])
                scene_data['pred'] = np.array(scene_data['pred'])
                
                # 添加到场景列表
                self.scenes.append(scene_data)
                self.scene_agent_counts.append(len(complete_agents))
            
            # 释放内存
            del df
            
        except Exception as e:
            print(f"处理文件 {path} 时出错: {e}")
    
    def __len__(self):
        """返回场景数量"""
        return len(self.scenes)
    
    def __getitem__(self, idx):
        """获取指定索引的场景数据"""
        scene = self.scenes[idx]
        
        # 转换为张量
        obs_traj = torch.FloatTensor(scene['obs'])
        pred_traj = torch.FloatTensor(scene['pred'])
        
        # 创建掩码，标记有效的智能体
        mask = torch.ones(len(scene['agent_ids']), dtype=torch.bool)
        
        # 返回观测轨迹、预测轨迹、掩码、智能体ID、智能体类型、观测长度和预测长度
        return obs_traj, pred_traj, mask, scene['agent_ids'], scene['agent_types'], scene['obs_len'], scene['pred_len']


def collate_fn(batch):
    """
    自定义批处理函数，处理不同场景中智能体数量不同的情况
    """
    # 解包批次数据
    batch_obs_traj, batch_pred_traj, batch_mask, batch_agent_ids, batch_agent_types, batch_obs_lens, batch_pred_lens = zip(*batch)
    
    # 找到最大智能体数量
    max_agents = max(len(obs) for obs in batch_obs_traj)
    
    # 创建填充后的张量
    batch_size = len(batch_obs_traj)
    obs_len = batch_obs_traj[0].shape[1]
    pred_len = batch_pred_traj[0].shape[1]
    feature_dim = batch_obs_traj[0].shape[2]
    
    # 初始化填充后的张量
    padded_batch_obs_traj = torch.zeros(batch_size, max_agents, obs_len, feature_dim)
    padded_batch_pred_traj = torch.zeros(batch_size, max_agents, pred_len, feature_dim)
    padded_batch_mask = torch.zeros(batch_size, max_agents, dtype=torch.bool)
    padded_batch_agent_types = torch.zeros(batch_size, max_agents, dtype=torch.long)

    # 填充数据
    for i, (obs, pred, mask, agent_types) in enumerate(zip(batch_obs_traj, batch_pred_traj, batch_mask, batch_agent_types)):
        num_agents = obs.shape[0]
        padded_batch_obs_traj[i, :num_agents] = obs
        padded_batch_pred_traj[i, :num_agents] = pred
        padded_batch_mask[i, :num_agents] = mask  # 使用原始掩码填充

        # 处理agent_types，如果是列表则转换为tensor
        if isinstance(agent_types, list):
            # 将字符串类型映射为数字
            type_mapping = {'VEHICLE': 0, 'BICYCLE': 1, 'PEDESTRIAN': 2, 'vehicle': 0, 'bicycle': 1, 'pedestrian': 2}
            converted_types = []
            for item in agent_types:
                if isinstance(item, str):
                    converted_types.append(type_mapping.get(item.lower(), 0))
                elif isinstance(item, (int, float)):
                    converted_types.append(int(item))
                else:
                    converted_types.append(0)  # 默认值
            agent_types_tensor = torch.tensor(converted_types, dtype=torch.long)
        else:
            agent_types_tensor = torch.tensor(agent_types, dtype=torch.long)

        padded_batch_agent_types[i, :num_agents] = agent_types_tensor

    # 返回批次数据
    return padded_batch_obs_traj, padded_batch_pred_traj, padded_batch_mask, batch_agent_ids, padded_batch_agent_types, batch_obs_lens, batch_pred_lens


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """
    创建数据加载器
    参数:
        dataset: 数据集实例
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作线程数
    返回:
        数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True  # 对CUDA更友好
    )


def normalize_trajectories(trajectories, scale_factor=1000.0):
    """
    对轨迹进行相对位置归一化处理 + 全局缩放
    参数:
        trajectories: 形状为(N, T, 2)的轨迹数组
        scale_factor: 全局缩放因子，默认1000.0
    返回:
        归一化后的轨迹和原始起点位置
    """
    # 计算每个智能体第一个时间步的位置
    first_pos = trajectories[:, 0:1, :]

    # 相对于第一个位置的偏移量 + 全局缩放
    rel_trajectories = (trajectories - first_pos) / scale_factor
    return rel_trajectories, first_pos


def denormalize_trajectories(rel_trajectories, first_pos, scale_factor=1000.0):
    """
    将归一化的轨迹转换回原始坐标

    参数:
        rel_trajectories: 相对轨迹
        first_pos: 原始起点位置
        scale_factor: 全局缩放因子，默认1000.0

    返回:
        原始坐标下的轨迹
    """
    # 先乘回缩放因子，再加原始起点
    rel_trajectories = rel_trajectories * scale_factor
    return rel_trajectories + first_pos



class PreprocessedDataset(Dataset):
    """预处理数据集类，直接从保存的文件加载数据"""
    
    def __init__(self, data_file):
        """
        初始化预处理数据集
        
        参数:
            data_file: 预处理数据文件路径
        """
        super(PreprocessedDataset, self).__init__()
        print(f"从预处理文件加载数据: {data_file}")
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            self.scenes = data['scenes']
            self.scene_agent_counts = data['scene_agent_counts']
        
        print(f"共加载了 {len(self.scenes)} 个场景，包含 {sum(self.scene_agent_counts)} 个轨迹")
    
    def __len__(self):
        """返回场景数量"""
        return len(self.scenes)
    
    def __getitem__(self, idx):
        """获取指定索引的场景数据"""
        scene = self.scenes[idx]
        
        # 转换为张量
        obs_traj = torch.FloatTensor(scene['obs'])
        pred_traj = torch.FloatTensor(scene['pred'])
        
        # 创建掩码，标记有效的智能体
        mask = torch.ones(len(scene['agent_ids']), dtype=torch.bool)
        
        # 返回观测轨迹、预测轨迹、掩码、智能体ID、智能体类型、观测长度和预测长度
        return obs_traj, pred_traj, mask, scene['agent_ids'], scene['agent_types'], scene['obs_len'], scene['pred_len']