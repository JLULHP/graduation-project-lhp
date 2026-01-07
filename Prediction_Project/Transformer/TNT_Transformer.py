import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Transformer.Models import Transformer, PositionalEncoding, Encoder


class TargetPredictor(nn.Module):
    """目标预测模块 - 第一阶段"""
    def __init__(self, d_feature_vec, d_hidden, num_targets=6, dropout=0.1):
        super(TargetPredictor, self).__init__()
        self.d_feature_vec = d_feature_vec
        self.num_targets = num_targets

        # 输入维度：聚合编码特征 + 观测终点 + 平均速度 + 平均加速度
        motion_feature_dim = d_feature_vec + 2 + 2 + 2  # 128 + 2 + 2 + 2 = 134
        self.target_net = nn.Sequential(
            nn.Linear(motion_feature_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),  
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_targets * 2)
        )
        self.prob_net = nn.Sequential(
            nn.Linear(motion_feature_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_targets)
        )
        
    def forward(self, motion_features, mask=None):
        """
        Args:
            motion_features: (batch_size, N, motion_feature_dim) 包含完整运动信息的特征
                           包括：聚合编码特征 + 观测终点 + 平均速度 + 平均加速度
            mask: (batch_size, N) 智能体掩码
        Returns:
            targets: (batch_size, N, num_targets, 2) 预测的目标位置（相对观测终点的位移）
            target_probs: (batch_size, N, num_targets) 目标概率
        """
        batch_size, N, feature_dim = motion_features.shape

        # 重塑为 (batch_size * N, feature_dim)
        features_flat = motion_features.view(batch_size * N, feature_dim)

        # 预测相对位移（相对于观测终点）
        target_outputs = self.target_net(features_flat)  # (batch_size * N, num_targets * 2)
        relative_targets = target_outputs.view(batch_size * N, self.num_targets, 2)

        # 预测目标概率
        target_probs = self.prob_net(features_flat)  # (batch_size * N, num_targets)

        # 处理非 Tensor 或空情况
        if not isinstance(target_probs, torch.Tensor) or target_probs.numel() == 0:
            target_probs = torch.zeros(batch_size, N, self.num_targets, device=motion_features.device)
        else:
            target_probs = torch.nn.functional.softmax(target_probs, dim=-1)

        # 重塑回原始维度
        target_probs = target_probs.view(batch_size, N, self.num_targets)
        targets = relative_targets.view(batch_size, N, self.num_targets, 2)

        # 掩码
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # (batch_size, N, 1, 1)
            targets = targets * mask_expanded
            target_probs = target_probs * mask.unsqueeze(-1)
        return targets, target_probs


class TrajectoryGenerator(nn.Module):
    """轨迹生成模块 - 第二阶段"""
    def __init__(self, d_feature_vec, d_hidden, pred_len, dropout=0.1, num_paths_per_target=2, d_latent=16):
        super(TrajectoryGenerator, self).__init__()
        self.num_paths_per_target = num_paths_per_target
        self.d_latent = d_latent
        self.pred_len = pred_len

        # 编码观测特征、观测终点、目标位置和真实轨迹
        # 输入维度: d_feature_vec + 2(观测终点) + 2(目标位置) + pred_len*2(真实轨迹)
        encoder_input_dim = d_feature_vec + 2 + 2 + pred_len * 2
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU()
        )
        self.mu_net = nn.Linear(d_hidden // 2, d_latent)  # 潜在空间均值
        self.logvar_net = nn.Linear(d_hidden // 2, d_latent)  # 潜在空间方差
        # 将观测特征、观测终点、目标位置和潜在变量解码为轨迹
        # 输入维度: d_feature_vec + 2(观测终点) + 2(目标位置) + d_latent
        decoder_input_dim = d_feature_vec + 2 + 2 + d_latent
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Linear(d_hidden // 2, pred_len * 2)
        )

    def _adjust_trajectory_endpoints(self, trajectories, obs_endpoints, targets):
        """
        调整轨迹的起点和终点，确保：
        1. 轨迹从观测终点开始
        2. 轨迹终点在目标位置
        保持轨迹的相对形状，通过线性插值调整
        """
        batch_size, N, num_targets, num_paths, pred_len, _ = trajectories.shape

        # 扩展观测终点和目标位置以匹配轨迹维度
        obs_endpoints_expanded = obs_endpoints.unsqueeze(2).unsqueeze(2).expand(
            -1, -1, num_targets, num_paths, -1)  # (B, N, num_targets, num_paths, 2)
        targets_expanded = targets.unsqueeze(3).expand(
            -1, -1, -1, num_paths, -1)  # (B, N, num_targets, num_paths, 2)

        # 获取轨迹的起点和终点
        trajectory_starts = trajectories[:, :, :, :, 0, :]  # (B, N, num_targets, num_paths, 2)
        trajectory_ends = trajectories[:, :, :, :, -1, :]  # (B, N, num_targets, num_paths, 2)

        # 计算起点和终点的偏移
        start_offset = obs_endpoints_expanded - trajectory_starts  # (B, N, num_targets, num_paths, 2)
        end_offset = targets_expanded - trajectory_ends  # (B, N, num_targets, num_paths, 2)

        # 创建时间步权重 (0 到 1)，形状: (pred_len,)
        time_weights = torch.linspace(0, 1, pred_len, device=trajectories.device)  # (pred_len,)
        
        # 扩展偏移到所有时间步，形状: (B, N, num_targets, num_paths, pred_len, 2)
        start_offset_expanded = start_offset.unsqueeze(4).expand(-1, -1, -1, -1, pred_len, -1)  # (B, N, num_targets, num_paths, pred_len, 2)
        end_offset_expanded = end_offset.unsqueeze(4).expand(-1, -1, -1, -1, pred_len, -1)  # (B, N, num_targets, num_paths, pred_len, 2)
        
        # 扩展时间权重以匹配形状
        time_weights_expanded = time_weights.view(1, 1, 1, 1, pred_len, 1)  # (1, 1, 1, 1, pred_len, 1)
        
        # 应用线性插值：起点偏移在 t=0 时完全应用，终点偏移在 t=1 时完全应用
        # offset(t) = start_offset * (1 - t) + end_offset * t
        combined_offset = start_offset_expanded * (1 - time_weights_expanded) + end_offset_expanded * time_weights_expanded
        
        # 应用偏移到整个轨迹
        adjusted_trajectories = trajectories + combined_offset

        return adjusted_trajectories

    def sample_latent(self, mu, logvar, batch_size, N, num_targets, num_paths):
        # 从潜在空间分布采样
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(batch_size * N * num_targets * num_paths, self.d_latent).to(mu.device)
        z = mu + eps * std
        return z
    
    def forward(self, obs_features, targets, obs_endpoints=None, gt_trajectories=None, training=True):
        """
        生成从观测终点到目标位置的多条轨迹

        Args:
            obs_features: 观测特征 (batch_size, N, d_feat)
            targets: 目标位置 (batch_size, N, num_targets, 2)
            obs_endpoints: 观测轨迹终点位置 (batch_size, N, 2)
            gt_trajectories: 真实轨迹 (batch_size, N, pred_len, 2)
            training: 是否训练模式
        """
        batch_size, N, d_feat = obs_features.shape
        _, _, num_targets, _ = targets.shape

        # 扩展观测特征和目标位置以匹配 num_paths_per_target
        obs_features_expanded = obs_features.unsqueeze(2).unsqueeze(2).expand(
            -1, -1, num_targets, self.num_paths_per_target, -1)
        obs_features_flat = obs_features_expanded.reshape(
            batch_size * N * num_targets * self.num_paths_per_target, d_feat)

        # 扩展观测终点位置
        if obs_endpoints is not None:
            obs_endpoints_expanded = obs_endpoints.unsqueeze(2).unsqueeze(2).expand(
                -1, -1, num_targets, self.num_paths_per_target, -1)
            obs_endpoints_flat = obs_endpoints_expanded.reshape(
                batch_size * N * num_targets * self.num_paths_per_target, 2)
        else:
            obs_endpoints_flat = torch.zeros(
                batch_size * N * num_targets * self.num_paths_per_target, 2).to(obs_features.device)

        targets_expanded = targets.unsqueeze(3).expand(-1, -1, -1, self.num_paths_per_target, -1)
        targets_flat = targets_expanded.reshape(
            batch_size * N * num_targets * self.num_paths_per_target, 2)
        
        if training and gt_trajectories is not None:
            # 训练时：使用真实轨迹学习分布
            gt_traj_expanded = gt_trajectories.unsqueeze(2).unsqueeze(2).expand(
                -1, -1, num_targets, self.num_paths_per_target, -1, -1)
            gt_traj_flat = gt_traj_expanded.reshape(
                batch_size * N * num_targets * self.num_paths_per_target, self.pred_len * 2)

            # 编码器输入：观测特征 + 观测终点 + 目标位置 + 真实轨迹
            encoder_input = torch.cat([obs_features_flat, obs_endpoints_flat, targets_flat, gt_traj_flat], dim=-1)
            encoder_output = self.encoder(encoder_input)
            mu = self.mu_net(encoder_output)
            logvar = self.logvar_net(encoder_output)
        else:
            # 预测时：生成新轨迹
            dummy_gt = torch.zeros(
                batch_size * N * num_targets * self.num_paths_per_target, self.pred_len * 2).to(obs_features.device)
            encoder_input = torch.cat([obs_features_flat, obs_endpoints_flat, targets_flat, dummy_gt], dim=-1)
            encoder_output = self.encoder(encoder_input)
            mu = self.mu_net(encoder_output)
            logvar = self.logvar_net(encoder_output)

        # 采样潜在变量
        z = self.sample_latent(mu, logvar, batch_size, N, num_targets, self.num_paths_per_target)

        # 解码生成轨迹
        decoder_input = torch.cat([obs_features_flat, obs_endpoints_flat, targets_flat, z], dim=-1)
        traj_outputs = self.decoder(decoder_input)
        trajectories = traj_outputs.reshape(
            batch_size, N, num_targets, self.num_paths_per_target, self.pred_len, 2)

        # 后处理：确保轨迹从观测终点开始，终点在目标位置
        if obs_endpoints is not None:
            trajectories = self._adjust_trajectory_endpoints(trajectories, obs_endpoints, targets)

        if training:
            return trajectories, mu, logvar
        else:
            return trajectories


class TrajectoryScorer(nn.Module):
    """轨迹评分模块 - 第三阶段"""
    def __init__(self, d_feature_vec, d_hidden, pred_len=80, dropout=0.1):
        super(TrajectoryScorer, self).__init__()
        input_dim = d_feature_vec + pred_len * 2  # 观测特征 + 轨迹特征
        self.score_net = nn.Sequential(
            nn.Linear(input_dim, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.ReLU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, obs_features, trajectories):
        """
        Args:
            obs_features: (batch_size, N, d_feat) 观测特征
            trajectories: (batch_size, N, num_targets, num_paths, pred_len, 2) 预测轨迹
        Returns:
            scores: (batch_size, N, num_targets, num_paths) 轨迹评分
        """
        batch_size, N, num_targets, num_paths, pred_len, _ = trajectories.shape
        d_feat = obs_features.shape[-1]

        # 扩展观测特征以匹配 num_targets 和 num_paths
        obs_features_expanded = obs_features.unsqueeze(2).unsqueeze(2).expand(
            -1, -1, num_targets, num_paths, -1)
        obs_features_flat = obs_features_expanded.reshape(
            batch_size * N * num_targets * num_paths, d_feat)
        
        # 将轨迹展平为特征向量
        trajectories_flat = trajectories.reshape(
            batch_size * N * num_targets * num_paths, pred_len * 2)
        
        # 拼接特征
        score_input = torch.cat([obs_features_flat, trajectories_flat], dim=-1)
        
        # 计算评分
        scores = self.score_net(score_input)
        scores = scores.reshape(batch_size, N, num_targets, num_paths)
        return scores


class TNT_Transformer(nn.Module):
    def __init__(self, n_features=2, d_feature_vec=128, n_layers=6, n_head=8, d_k=64, d_v=64, d_inner=2048, dropout=0.1, d_latent=16, obs_len=20, pred_len=80, num_targets=3, num_paths_per_target=2, scale_emb=False, num_agent_types=3):
        super(TNT_Transformer, self).__init__()
        self.pred_len = pred_len
        self.num_targets = num_targets
        self.num_paths_per_target = num_paths_per_target
        self.num_agent_types = num_agent_types
        self.type_embedding = nn.Embedding(num_agent_types, d_feature_vec)
        # 输入投影
        # self.input_proj = nn.Linear(n_features, d_feature_vec)
        # Transformer编码器
        self.encoder = Encoder(
            n_features=n_features,
            seq_len=obs_len,
            d_feature_vec=d_feature_vec,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            d_inner=d_inner,
            dropout=dropout,
            scale_emb=scale_emb
        )
        
        # 目标预测器
        self.target_predictor = TargetPredictor(
            d_feature_vec=d_feature_vec,
            d_hidden=d_inner,
            num_targets=num_targets,
            dropout=dropout
        )
        
        # 轨迹生成器 
        self.trajectory_generator = TrajectoryGenerator(
            d_feature_vec=d_feature_vec,
            d_hidden=d_inner,
            pred_len=pred_len,
            dropout=dropout,
            d_latent=16,
            num_paths_per_target=num_paths_per_target
        )
        
        # 轨迹评分器
        self.trajectory_scorer = TrajectoryScorer(
            d_feature_vec=d_feature_vec,
            d_hidden=d_inner // 2,
            pred_len=pred_len,
            dropout=dropout
        )
    
    def forward(self, obs_traj, mask=None, agent_types=None, gt_trajectories=None, return_multimodal=False, training=True):
        batch_size, N, seq_len, _ = obs_traj.shape

        # 提取观测轨迹的终点位置（归一化后的相对坐标）
        obs_endpoints = obs_traj[:, :, -1, :]  # (batch_size, N, 2)

        # 获取智能体类型嵌入
        if agent_types is not None:
            type_emb = self.type_embedding(agent_types)  # (batch_size, N, d_feature_vec)
            # 确保type_emb_expanded的形状与obs_traj完全匹配
            type_emb_expanded = type_emb.unsqueeze(2).repeat(1, 1, seq_len, 1)  # (batch_size, N, seq_len, d_feature_vec)
            obs_traj_emb = torch.cat([obs_traj, type_emb_expanded], dim=-1)  # (batch_size, N, seq_len, 2 + d_feature_vec)
        else:
            obs_traj_emb = obs_traj
        enc_output = self.encoder(obs_traj)

        # 使用平均池化聚合所有时间步的特征，捕捉完整运动趋势
        enc_aggregated = enc_output.mean(dim=2)  # (batch_size, N, d_feature_vec)

        # 计算速度：相邻时间步的位置差
        velocities = obs_traj[:, :, 1:, :] - obs_traj[:, :, :-1, :]  # (batch_size, N, obs_len-1, 2)
        mean_velocity = velocities.mean(dim=2)  # (batch_size, N, 2) 平均速度

        # 计算加速度：相邻速度的差
        accelerations = velocities[:, :, 1:, :] - velocities[:, :, :-1, :]  # (batch_size, N, obs_len-2, 2)
        mean_acceleration = accelerations.mean(dim=2)  # (batch_size, N, 2) 平均加速度

        # 拼接所有特征：聚合编码特征 + 观测终点 + 平均速度 + 平均加速度
        motion_features = torch.cat([
            enc_aggregated,      # 聚合的序列特征
            obs_endpoints,       # 当前位置
            mean_velocity,       # 运动趋势（速度）
            mean_acceleration    # 运动变化趋势（加速度）
        ], dim=-1)  # (batch_size, N, d_feature_vec + 2 + 2 + 2)

        # 预测目标 
        targets, target_probs = self.target_predictor(motion_features, mask)

        # 生成轨迹 (使用VAE)
        if training and gt_trajectories is not None:
            trajectories, mu, logvar = self.trajectory_generator(
                enc_aggregated, targets, obs_endpoints=obs_endpoints,
                gt_trajectories=gt_trajectories, training=True
            )
        else:
            trajectories = self.trajectory_generator(
                enc_aggregated, targets, obs_endpoints=obs_endpoints, training=False
            )
            mu, logvar = None, None

        # 评分轨迹
        trajectory_scores = self.trajectory_scorer(enc_aggregated, trajectories)# (batch_size, N, num_targets, num_paths)
        
        if return_multimodal:
            if training:
                return trajectories, mu, logvar, target_probs, trajectory_scores, enc_aggregated
            else:
                return trajectories, target_probs, trajectory_scores, enc_aggregated
        else:
            # 选择最佳轨迹
            # 扩展 target_probs 以匹配 num_paths 维度
            target_probs_expanded = target_probs.unsqueeze(-1).expand(
                -1, -1, -1, self.num_paths_per_target)  # (batch, N, num_targets, num_paths)
            
            # 计算综合评分
            total_scores = target_probs_expanded * trajectory_scores  # (batch, N, num_targets, num_paths)
            # 展平以找到最佳轨迹索引
            total_scores_flat = total_scores.view(batch_size, N, -1)  # (batch, N, num_targets * num_paths)
            best_traj_indices = torch.argmax(total_scores_flat, dim=-1)  # (batch, N)
            # 计算最佳目标索引和路径索引
            best_target_indices = best_traj_indices // self.num_paths_per_target
            best_path_indices = best_traj_indices % self.num_paths_per_target
            
            # 选择最佳轨迹
            final_trajectories = torch.zeros(
                batch_size, N, self.pred_len, 2, device=trajectories.device)
            for b in range(batch_size):
                for n in range(N):
                    t = best_target_indices[b, n]
                    p = best_path_indices[b, n]
                    final_trajectories[b, n] = trajectories[b, n, t, p, :, :]
            
            if training:
                return final_trajectories, mu, logvar
            else:
                return final_trajectories
    
    def predict_multimodal(self, obs_traj, mask=None, agent_types=None, num_samples=3):
        """
        生成多模态轨迹预测
        Args:
            obs_traj: (batch_size, N, T, F) 观测轨迹
            mask: (batch_size, N) 智能体掩码
            agent_types: (batch_size, N) 智能体类型标识
            num_samples: 返回的轨迹数量 (最大为num_targets * num_paths_per_target，即6条)
        Returns:
            trajectories: (batch_size, N, num_samples, pred_len, 2) 多模态轨迹
            probabilities: (batch_size, N, num_samples) 轨迹概率
        """
        max_samples = self.num_targets * self.num_paths_per_target
        num_samples = min(num_samples, max_samples)

        batch_size, N, T, n_feat = obs_traj.shape
        # 获取所有候选轨迹和评分
        trajectories, target_probs, trajectory_scores, _ = self.forward(
            obs_traj, mask=mask, agent_types=agent_types, return_multimodal=True, training=False
        )

        # trajectories shape: (batch_size, N, num_targets, num_paths, pred_len, 2)
        # target_probs shape: (batch_size, N, num_targets)
        # trajectory_scores shape: (batch_size, N, num_targets, num_paths)

        # 扩展 target_probs 以匹配 num_paths 维度
        target_probs_expanded = target_probs.unsqueeze(-1).expand(
            -1, -1, -1, self.num_paths_per_target)  # (batch_size, N, num_targets, num_paths)

        # 计算综合评分
        combined_scores = target_probs_expanded * trajectory_scores  # (batch_size, N, num_targets, num_paths)
        
        # 展平以选择 top-k 轨迹
        combined_scores_flat = combined_scores.view(batch_size, N, -1)  # (batch_size, N, num_targets * num_paths)
        total_candidates = combined_scores_flat.shape[-1]

        # 确保num_samples不超过可用候选数量且至少为1
        num_samples = max(1, min(num_samples, total_candidates))

        top_k_scores, top_k_indices = torch.topk(combined_scores_flat, k=num_samples, dim=-1)

        # 确保top_k_scores是正确的tensor
        if not isinstance(top_k_scores, torch.Tensor):
            top_k_scores = torch.tensor(top_k_scores, device=combined_scores_flat.device, dtype=combined_scores_flat.dtype)
        
        # torch.topk 当 k=1 时返回 (batch_size, N)，当 k>1 时返回 (batch_size, N, k)
        if top_k_scores.dim() == 2:
            # 如果是2维，需要添加最后一个维度
            top_k_scores = top_k_scores.unsqueeze(-1)  # (batch_size, N) -> (batch_size, N, 1)
        
        # 确保是3维tensor
        assert top_k_scores.dim() == 3, f"top_k_scores should be 3D, got {top_k_scores.dim()}D with shape {top_k_scores.shape}"

        # 计算目标索引和路径索引
        top_k_target_indices = top_k_indices // self.num_paths_per_target  # (batch_size, N, num_samples)
        top_k_path_indices = top_k_indices % self.num_paths_per_target  # (batch_size, N, num_samples)

        # 提取对应的轨迹
        selected_trajectories = torch.zeros(
            batch_size, N, num_samples, self.pred_len, 2, device=trajectories.device)
        for b in range(batch_size):
            for n in range(N):
                for s in range(num_samples):
                    t = top_k_target_indices[b, n, s]
                    p = top_k_path_indices[b, n, s]
                    selected_trajectories[b, n, s] = trajectories[b, n, t, p, :, :]

        # 归一化
        probabilities = F.softmax(top_k_scores, dim=-1)
        return selected_trajectories, probabilities 