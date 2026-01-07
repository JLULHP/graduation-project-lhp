import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用通用字体
plt.rcParams['axes.unicode_minus'] = False    
from tqdm import tqdm
import json
import yaml

from Transformer.TNT_Transformer import TNT_Transformer
from utils.data_process import MultiAgentTrajectoryDataset, create_dataloader, normalize_trajectories, denormalize_trajectories, PreprocessedDataset


class TNTLoss(nn.Module):
    """TNT模型的损失函数"""
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, vae_weight=1.0, kl_beta=0.1, max_target_distance=0.005, delta=0.01):
        super(TNTLoss, self).__init__()
        self.alpha = alpha  # 轨迹损失权重
        self.beta = beta    # 目标损失权重
        self.gamma = gamma  # 评分损失权重
        self.vae_weight = vae_weight  # VAE 损失权重
        self.kl_beta = kl_beta  # KL 散度损失权重
        self.max_target_distance = max_target_distance  # 目标点最大允许距离（归一化后的单位，0.005对应5米）
        self.delta = delta  # 平滑损失权重
        
    def forward(self, pred_trajectories, gt_trajectories, pred_targets, gt_targets, 
                target_probs, trajectory_scores, mask=None, mu=None, logvar=None):
        """
        Args:
            pred_trajectories: (batch_size, N, num_targets, num_paths, pred_len, 2) 预测轨迹
            gt_trajectories: (batch_size, N, pred_len, 2) 真实轨迹
            pred_targets: (batch_size, N, num_targets, 2) 预测目标
            gt_targets: (batch_size, N, 2) 真实目标（轨迹终点）
            target_probs: (batch_size, N, num_targets) 目标概率
            trajectory_scores: (batch_size, N, num_targets, num_paths) 轨迹评分
            mask: (batch_size, N) 智能体掩码
            mu: (batch_size * N * num_targets * num_paths, d_latent) 潜在空间均值
            logvar: (batch_size * N * num_targets * num_paths, d_latent) 潜在空间方差
        """
        batch_size, N, num_targets, num_paths, pred_len, _ = pred_trajectories.shape
        
        # 1. 轨迹损失 - 找到最接近真实轨迹的预测轨迹
        gt_trajectories_expanded = gt_trajectories.unsqueeze(2).unsqueeze(2).expand(-1, -1, num_targets, num_paths, -1, -1)
        trajectory_losses = torch.norm(pred_trajectories - gt_trajectories_expanded, dim=-1).mean(dim=-1)  # (batch_size, N, num_targets, num_paths)
        
        # 找到最佳匹配的轨迹
        trajectory_losses_flat = trajectory_losses.view(batch_size, N, -1)
        best_traj_indices = torch.argmin(trajectory_losses_flat, dim=-1)  # (batch_size, N)
        best_target_indices = best_traj_indices // num_paths
        best_path_indices = best_traj_indices % num_paths
        
        # 计算最佳轨迹的损失
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, N)
        agent_indices = torch.arange(N).unsqueeze(0).expand(batch_size, -1)
        best_trajectory_loss = trajectory_losses_flat[batch_indices, agent_indices, best_traj_indices]
        
        # 2. 目标损失 - 计算预测目标中与真实目标最接近的距离
        gt_targets_expanded = gt_targets.unsqueeze(2).expand(-1, -1, num_targets, -1)
        target_distances = torch.norm(pred_targets - gt_targets_expanded, dim=-1)  # (batch_size, N, num_targets)
        target_losses = torch.min(target_distances, dim=-1)[0]  # (batch_size, N) 取最小距离
        
        # 2.5. 目标距离约束损失 - 限制所有目标点不能离真实目标太远
        # 如果距离超过max_target_distance，则增加惩罚
        max_distance_penalty = torch.clamp(target_distances - self.max_target_distance, min=0.0)  # (batch_size, N, num_targets)
        # 应用掩码
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_targets)  # (batch_size, N, num_targets)
            max_distance_penalty = max_distance_penalty * mask_expanded
            valid_targets = mask_expanded.sum()
        else:
            valid_targets = batch_size * N * num_targets
        # 只对超过最大距离的目标点进行惩罚
        target_range_loss = max_distance_penalty.sum() / valid_targets if valid_targets > 0 else torch.tensor(0.0, device=target_distances.device)

        # 3. 评分损失,鼓励最佳轨迹有更高的评分
        best_traj_scores = trajectory_scores.view(batch_size, N, -1)[batch_indices, agent_indices, best_traj_indices]
        score_loss = -torch.log(best_traj_scores + 1e-8).mean()
        
        # 4. 多样性损失
        diversity_loss = 0
        for i in range(num_targets):
            for j in range(i + 1, num_targets):
                target_diff = torch.norm(pred_targets[:, :, i, :] - pred_targets[:, :, j, :], dim=-1)
                diversity_loss += torch.exp(-target_diff).mean()
        diversity_loss = diversity_loss / (num_targets * (num_targets - 1) / 2)
        
        # VAE多样性损失
        vae_loss = 0
        kl_loss = 0
        if mu is not None and logvar is not None:
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (batch_size * N * num_targets * num_paths)
            vae_loss = self.kl_beta * kl_loss
        
        # 应用掩码
        if mask is not None:
            best_trajectory_loss = best_trajectory_loss * mask
            target_losses = target_losses * mask  # target_losses 现在是 (batch_size, N)
            valid_samples = mask.sum()
        else:
            valid_samples = batch_size * N
        
        # 平滑损失 - 对所有轨迹的速度和加速度变化进行惩罚
        pred_trajectories_all = pred_trajectories.view(batch_size, N, num_targets * num_paths, pred_len, 2)
        pred_velocities = pred_trajectories_all[:, :, :, 1:, :] - pred_trajectories_all[:, :, :, :-1, :]
        pred_accelerations = pred_velocities[:, :, :, 1:, :] - pred_velocities[:, :, :, :-1, :]
        velocity_loss = torch.mean(pred_velocities.pow(2))
        acceleration_loss = torch.mean(pred_accelerations.pow(2))
        smoothness_loss = velocity_loss + acceleration_loss
        if mask is not None:
            mask_expanded_vel = mask.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, num_targets * num_paths, pred_len - 1, 2)
            mask_expanded_acc = mask.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, num_targets * num_paths, pred_len - 2, 2)
            velocity_loss = torch.mean(pred_velocities.pow(2) * mask_expanded_vel)
            acceleration_loss = torch.mean(pred_accelerations.pow(2) * mask_expanded_acc)
            smoothness_loss = velocity_loss + acceleration_loss

        # 计算总损失
        total_loss = (self.alpha * best_trajectory_loss.sum() / valid_samples +
                     self.beta * target_losses.sum() / valid_samples +
                     self.gamma * score_loss +
                     0.1 * diversity_loss +
                     self.vae_weight * vae_loss +
                     0.5 * target_range_loss +
                     self.delta * smoothness_loss)
        
        return total_loss, {
            'trajectory_loss': best_trajectory_loss.mean().item(),
            'target_loss': target_losses.mean().item(),
            'target_range_loss': target_range_loss.item(),
            'score_loss': score_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'vae_loss': vae_loss.item() if vae_loss != 0 else 0,
            'kl_loss': kl_loss.item() if kl_loss != 0 else 0,
            'smoothness_loss': smoothness_loss.item()
        }


def train_tnt(args):
    """训练TNT-Transformer模型"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备: {device}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建数据集
    if args.preprocessed_train:
        print(f"从预处理文件加载训练数据: {args.preprocessed_train}")
        train_dataset = PreprocessedDataset(args.preprocessed_train)
    else:
        print("正在加载训练数据集...")
        train_dataset = MultiAgentTrajectoryDataset(
            data_dir=args.train_data,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            min_agents=args.min_agents,
            max_agents=args.max_agents,
            min_seq_len=args.min_seq_len,
            cache_file=os.path.join(args.save_dir, 'train_cache.pkl') if args.use_cache else None
        )
    
    if args.preprocessed_val:
        print(f"从预处理文件加载验证数据: {args.preprocessed_val}")
        val_dataset = PreprocessedDataset(args.preprocessed_val)
    else:
        print("正在加载验证数据集...")
        val_dataset = MultiAgentTrajectoryDataset(
            data_dir=args.val_data,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            min_agents=args.min_agents,
            max_agents=args.max_agents,
            min_seq_len=args.min_seq_len,
            cache_file=os.path.join(args.save_dir, 'val_cache.pkl') if args.use_cache else None
        )
    
    # 创建数据加载器
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 创建TNT-Transformer模型
    model = TNT_Transformer(
        n_features=args.n_features,
        d_feature_vec=args.d_feature_vec,
        n_layers=args.n_layers,
        n_head=args.n_head,
        d_k=args.d_k,
        d_v=args.d_v,
        d_inner=args.d_inner,
        dropout=args.dropout,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        num_targets=args.num_targets,
        num_paths_per_target=args.num_paths_per_target,
        d_latent=args.d_latent,
        scale_emb=args.scale_emb
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = TNTLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, 
                       vae_weight=args.vae_weight, kl_beta=args.kl_beta,
                       max_target_distance=getattr(args, 'max_target_distance', 0.005))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    epoch_list = []
    
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and torch.cuda.is_available() else None
    
    for epoch in range(1, args.epochs + 1):
        # 训练模式
        model.train()
        train_loss = 0
        loss_components = {'trajectory_loss': 0, 'target_loss': 0, 'target_range_loss': 0, 'score_loss': 0, 'diversity_loss': 0, 'vae_loss': 0, 'kl_loss': 0, 'smoothness_loss': 0}
        
        train_pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch}/{args.epochs}", mininterval=5.0)
        
        for batch_obs_traj, batch_pred_traj, batch_mask, batch_agent_ids, batch_agent_types, batch_obs_lens, batch_pred_lens in train_pbar:
            batch_obs_traj = batch_obs_traj.to(device)
            batch_pred_traj = batch_pred_traj.to(device)
            batch_mask = batch_mask.to(device)
            batch_agent_types = batch_agent_types.to(device)
            
            # 归一化数据
            batch_size = batch_obs_traj.size(0)
            first_pos_list = []
            for i in range(batch_size):
                valid_agents = torch.nonzero(batch_mask[i], as_tuple=True)[0]
                if valid_agents.numel() > 0:
                    norm_obs, first_pos = normalize_trajectories(batch_obs_traj[i, valid_agents], scale_factor=1000.0)
                    batch_obs_traj[i, valid_agents] = norm_obs
                    norm_pred = (batch_pred_traj[i, valid_agents] - first_pos) / 1000.0
                    batch_pred_traj[i, valid_agents] = norm_pred
                    first_pos_list.append(first_pos)
                else:
                    first_pos_list.append(None)
            
            gt_targets = batch_pred_traj[:, :, -1, :2]

            trajectories, mu, logvar, target_probs, trajectory_scores, enc_aggregated = model(
                batch_obs_traj, mask=batch_mask, agent_types=batch_agent_types, gt_trajectories=batch_pred_traj, return_multimodal=True, training=True
            )

            obs_endpoints = batch_obs_traj[:, :, -1, :]
            velocities = batch_obs_traj[:, :, 1:, :] - batch_obs_traj[:, :, :-1, :]
            mean_velocity = velocities.mean(dim=2)
            accelerations = velocities[:, :, 1:, :] - velocities[:, :, :-1, :]
            mean_acceleration = accelerations.mean(dim=2)

            motion_features = torch.cat([
                enc_aggregated, obs_endpoints, mean_velocity, mean_acceleration
            ], dim=-1)

            targets, _ = model.target_predictor(motion_features, batch_mask)

            loss, loss_dict = criterion(
                trajectories, batch_pred_traj[:, :, :, :2], targets, gt_targets,
                target_probs, trajectory_scores, batch_mask, mu, logvar
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            for key, value in loss_dict.items():
                loss_components[key] += value
            
            train_pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "traj_loss": f"{loss_dict['trajectory_loss']:.6f}",
                "target_loss": f"{loss_dict['target_loss']:.6f}",
                "smooth_loss": f"{loss_dict['smoothness_loss']:.6f}",
                "vae_loss": f"{loss_dict['vae_loss']:.6f}"
            })
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        for key in loss_components:
            loss_components[key] /= len(train_loader)

        # 验证模式
        model.eval()
        val_loss = 0
        
        val_pbar = tqdm(val_loader, desc=f"验证 Epoch {epoch}/{args.epochs}", mininterval=5.0)
        
        with torch.no_grad():
            for obs_traj, pred_traj_gt, mask, agent_ids, agent_types, obs_lens, pred_lens in val_pbar:
                obs_traj = obs_traj.to(device)
                pred_traj_gt = pred_traj_gt.to(device)
                mask = mask.to(device)
                agent_types = agent_types.to(device)
                
                # 归一化数据
                batch_size = obs_traj.size(0)
                for i in range(batch_size):
                    valid_agents = torch.nonzero(mask[i], as_tuple=True)[0]
                    if valid_agents.numel() > 0:
                        norm_obs, first_pos = normalize_trajectories(obs_traj[i, valid_agents], scale_factor=1000.0)
                        obs_traj[i, valid_agents] = norm_obs
                        norm_pred = (pred_traj_gt[i, valid_agents] - first_pos) / 1000.0
                        pred_traj_gt[i, valid_agents] = norm_pred
                
                gt_targets = pred_traj_gt[:, :, -1, :2]

                trajectories, target_probs, trajectory_scores, enc_aggregated = model(
                    obs_traj, mask=mask, agent_types=agent_types, return_multimodal=True, training=False
                )

                obs_endpoints = obs_traj[:, :, -1, :]
                velocities = obs_traj[:, :, 1:, :] - obs_traj[:, :, :-1, :]
                mean_velocity = velocities.mean(dim=2)
                accelerations = velocities[:, :, 1:, :] - velocities[:, :, :-1, :]
                mean_acceleration = accelerations.mean(dim=2)

                motion_features = torch.cat([
                    enc_aggregated, obs_endpoints, mean_velocity, mean_acceleration
                ], dim=-1)

                targets, _ = model.target_predictor(motion_features, mask)

                loss, _ = criterion(
                    trajectories, pred_traj_gt[:, :, :, :2], targets, gt_targets,
                    target_probs, trajectory_scores, mask, None, None
                )

                val_loss += loss.item()

                val_pbar.set_postfix({"val_loss": f"{loss.item():.6f}"})
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        epoch_list.append(epoch)

        scheduler.step()
        
        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  训练损失: {train_loss:.6f}")
        print(f"  验证损失: {val_loss:.6f}")
        print(f"  学习率: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'args': args
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  保存最佳模型 (验证损失: {best_val_loss:.6f})")

        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'args': args
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, train_losses, label='Train Loss')
    plt.plot(epoch_list, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, [abs(t - v) for t, v in zip(train_losses, val_losses)], label='Loss Difference')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.title('Training and Validation Loss Difference')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_history.png'))
    plt.close()
    
    with open(os.path.join(args.save_dir, 'training_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"训练完成，最佳验证损失: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练TNT-Transformer轨迹预测模型')
    
    # 加载配置文件
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 数据参数
    parser.add_argument('--train-data', type=str, default=config['data']['train_data'], help='训练数据目录')
    parser.add_argument('--val-data', type=str, default=config['data']['val_data'], help='验证数据目录')
    parser.add_argument('--preprocessed-train', type=str, default=config['data']['preprocessed_train'], help='预处理训练数据文件')
    parser.add_argument('--preprocessed-val', type=str, default=config['data']['preprocessed_val'], help='预处理验证数据文件')
    parser.add_argument('--obs-len', type=int, default=config['data']['obs_len'], help='观测序列长度')
    parser.add_argument('--pred-len', type=int, default=config['data']['pred_len'], help='预测序列长度')
    parser.add_argument('--min-agents', type=int, default=config['data']['min_agents'], help='每个场景最少智能体数量')
    parser.add_argument('--max-agents', type=int, default=config['data']['max_agents'], help='每个场景最多智能体数量')
    parser.add_argument('--min-seq-len', type=int, default=config['data']['min_seq_len'], help='最小可接受的序列长度')
    
    # 模型参数
    parser.add_argument('--n-features', type=int, default=config['model']['n_features'], help='特征数量')
    parser.add_argument('--d-feature-vec', type=int, default=config['model']['d_feature_vec'], help='特征向量维度')
    parser.add_argument('--n-layers', type=int, default=config['model']['n_layers'], help='Transformer层数')
    parser.add_argument('--n-head', type=int, default=config['model']['n_head'], help='多头注意力头数')
    parser.add_argument('--d-k', type=int, default=config['model']['d_k'], help='键维度')
    parser.add_argument('--d-v', type=int, default=config['model']['d_v'], help='值维度')
    parser.add_argument('--d-inner', type=int, default=config['model']['d_inner'], help='前馈网络隐藏层维度')
    parser.add_argument('--dropout', type=float, default=config['model']['dropout'], help='Dropout率')
    parser.add_argument('--scale-emb', action='store_true', default=config['model']['scale_emb'], help='是否缩放嵌入')
    parser.add_argument('--num-targets', type=int, default=config['model']['num_targets'], help='目标数量')
    parser.add_argument('--num-paths-per-target', type=int, default=config['model']['num_paths_per_target'], help='每个目标的轨迹数量')
    parser.add_argument('--d-latent', type=int, default=config['model']['d_latent'], help='VAE潜在空间维度')
    
    # 损失函数参数
    parser.add_argument('--alpha', type=float, default=config['loss']['alpha'], help='轨迹损失权重')
    parser.add_argument('--beta', type=float, default=config['loss']['beta'], help='目标损失权重')
    parser.add_argument('--gamma', type=float, default=config['loss']['gamma'], help='评分损失权重')
    parser.add_argument('--vae-weight', type=float, default=config['loss']['vae_weight'], help='VAE损失权重')
    parser.add_argument('--kl-beta', type=float, default=config['loss']['kl_beta'], help='KL散度损失权重')
    parser.add_argument('--max-target-distance', type=float, default=config['loss'].get('max_target_distance', 0.005), help='目标点最大允许距离（归一化单位，0.005对应5米，因为scale_factor=1000）')
    parser.add_argument('--delta', type=float, default=config['loss'].get('delta', 0.01), help='平滑损失权重')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=config['training']['batch_size'], help='批次大小')
    parser.add_argument('--epochs', type=int, default=config['training']['epochs'], help='训练轮数')
    parser.add_argument('--lr', type=float, default=config['training']['lr'], help='学习率')
    parser.add_argument('--lr-step', type=int, default=config['training']['lr_step'], help='学习率衰减步长')
    parser.add_argument('--lr-gamma', type=float, default=config['training']['lr_gamma'], help='学习率衰减因子')
    parser.add_argument('--num-workers', type=int, default=config['training']['num_workers'], help='数据加载器工作进程数')
    parser.add_argument('--save-dir', type=str, default=config['training']['save_dir'], help='模型保存目录')
    parser.add_argument('--save-interval', type=int, default=config['training']['save_interval'], help='保存检查点间隔')
    parser.add_argument('--use-cache', action='store_true', default=config['training']['use_cache'], help='是否使用数据缓存')
    parser.add_argument('--use-amp', action='store_true', default=config['training']['use_amp'], help='是否使用混合精度训练')
    parser.add_argument('--no-cuda', action='store_true', default=config['training']['no_cuda'], help='不使用CUDA')
    
    args = parser.parse_args()
    
    train_tnt(args) 