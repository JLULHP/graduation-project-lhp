import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import yaml
import argparse

from Transformer.TNT_Transformer import TNT_Transformer
from utils.data_process import MultiAgentTrajectoryDataset, PreprocessedDataset, create_dataloader, normalize_trajectories, denormalize_trajectories

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_model(config_path, model_path):
    """加载训练好的TNT模型"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() and not config['training']['no_cuda'] else 'cpu')
    print(f"使用设备: {device}")

    # 创建TNT模型
    model = TNT_Transformer(
        n_features=config['model']['n_features'],
        d_feature_vec=config['model']['d_feature_vec'],
        n_layers=config['model']['n_layers'],
        n_head=config['model']['n_head'],
        d_k=config['model']['d_k'],
        d_v=config['model']['d_v'],
        d_inner=config['model']['d_inner'],
        dropout=config['model']['dropout'],
        obs_len=config['data']['obs_len'],
        pred_len=config['data']['pred_len'],
        num_targets=config['model']['num_targets'],
        num_paths_per_target=config['model']['num_paths_per_target'],
        d_latent=config['model']['d_latent'],
        scale_emb=config['model']['scale_emb']
    )

    # 加载模型权重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载模型权重: {model_path}")
        if 'epoch' in checkpoint:
            print(f"模型训练轮数: {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            print(f"最佳验证损失: {checkpoint['best_val_loss']:.6f}")
    else:
        print(f"警告: 模型文件不存在 {model_path}，将使用随机初始化的模型")

    # 评估模式
    model = model.to(device)
    model.eval()

    return model, device, config


def calculate_metrics(pred_trajectories, gt_trajectories, pred_probs=None):
    """
    计算预测指标

    Args:
        pred_trajectories: 预测轨迹 (batch_size, N, num_samples, pred_len, 2) 或 (batch_size, N, pred_len, 2)
        gt_trajectories: 真实轨迹 (batch_size, N, pred_len, 2)
        pred_probs: 预测概率 (batch_size, N, num_samples) 用于多模态预测

    Returns:
        各种评估指标
    """
    # 处理多模态预测：如果是多模态，选择最接近真实轨迹的预测
    if len(pred_trajectories.shape) == 5:  # 多模态预测 (batch_size, N, num_samples, pred_len, 2)
        batch_size, N, num_samples, pred_len, _ = pred_trajectories.shape
        # 扩展真实轨迹以匹配预测轨迹维度
        gt_expanded = gt_trajectories.unsqueeze(2).expand(-1, -1, num_samples, -1, -1)  # (batch_size, N, num_samples, pred_len, 2)
        # 计算每个样本的ADE
        ade_per_sample = torch.norm(pred_trajectories - gt_expanded, dim=-1).mean(dim=-1)  # (batch_size, N, num_samples)
        # 选择最接近真实轨迹的预测（最小ADE）
        best_indices = torch.argmin(ade_per_sample, dim=-1)  # (batch_size, N)
        # 提取最佳预测轨迹
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, N)
        agent_indices = torch.arange(N).unsqueeze(0).expand(batch_size, -1)
        best_trajectories = pred_trajectories[batch_indices, agent_indices, best_indices]  # (batch_size, N, pred_len, 2)
        pred_trajectories = best_trajectories
    
    # ADE
    ade = torch.norm(pred_trajectories - gt_trajectories, dim=-1).mean(dim=-1)  # (batch_size, N)

    # FDE
    fde = torch.norm(pred_trajectories[:, :, -1, :] - gt_trajectories[:, :, -1, :], dim=-1)  # (batch_size, N)

    return {
        'ADE': ade.mean().item(),
        'FDE': fde.mean().item(),
        'ADE_std': ade.std().item(),
        'FDE_std': fde.std().item()
    }


def visualize_predictions(obs_traj, pred_trajectories, gt_traj, pred_targets=None,
                         save_path=None, agent_ids=None, scenario_idx=0, target_agent_id=None, valid_agents=None):
    """
    可视化预测结果

    Args:
        obs_traj: 观测轨迹 (N, obs_len, 2)
        pred_trajectories: 预测轨迹 (N, num_samples, pred_len, 2)
        gt_traj: 真实轨迹 (N, pred_len, 2)
        pred_targets: 预测目标位置 (N, num_targets, 2)
        save_path: 保存路径
        agent_ids: 智能体ID列表
        scenario_idx: 场景索引
        target_agent_id: 要预测的智能体ID
        valid_agents: 有效智能体的索引数组
    """
    N = obs_traj.shape[0]
    
    # 选择要可视化的智能体
    if target_agent_id is not None and agent_ids is not None:
        try:
            i = agent_ids.index(target_agent_id)
            if valid_agents is not None and i not in valid_agents:
                print(f"Warning: Agent ID {target_agent_id} not in valid agents. Using first valid agent.")
                i = valid_agents[0] if len(valid_agents) > 0 else 0
        except ValueError:
            print(f"Warning: Agent ID {target_agent_id} not found. Using displacement-based selection.")
            i = None
    else:
        i = None

    if i is None:
        # 自动选择位移最大的智能体
        candidates = valid_agents if valid_agents is not None and len(valid_agents) > 0 else range(N)
        displacements = []
        for n in candidates:
            displacement = np.linalg.norm(obs_traj[n, -1, :] - obs_traj[n, 0, :])
            displacements.append(displacement)
        i = list(candidates)[np.argmax(displacements)]
    
    plt.figure(figsize=(12, 10))

    # 1. 绘制20帧历史轨迹：蓝实线
    plt.plot(obs_traj[i, :, 0], obs_traj[i, :, 1], 'b-',
            linewidth=2.5, label='Observed Trajectory (20 frames)', zorder=3, marker='o', markersize=4)

    # 2. 绘制80帧真实轨迹：蓝虚线
    plt.plot(gt_traj[i, :, 0], gt_traj[i, :, 1], 'b--',
            linewidth=2, label='Ground Truth (80 frames)', zorder=2, marker='s', markersize=4)


    # 3. 绘制3个预测目标点：红点
    if pred_targets is not None:
        targets = pred_targets[i]  # (num_targets, 2)
        plt.scatter(targets[:, 0], targets[:, 1], c='red', s=100, 
                   marker='o', label='Predicted Targets', zorder=5, edgecolors='darkred', linewidths=1.5)

    # 绘制预测轨迹
    if len(pred_trajectories.shape) == 4:  # 多模态预测
        num_samples = pred_trajectories.shape[1]
        for j in range(num_samples):
            plt.plot(pred_trajectories[i, j, :, 0], pred_trajectories[i, j, :, 1],
                    'r--', linewidth=1.5, alpha=0.7, zorder=1)
        if num_samples > 0:
            plt.plot([], [], 'r--', linewidth=1.5, label=f'Predicted Trajectories ({num_samples} paths)')
    else:  # 单模态预测
        plt.plot(pred_trajectories[i, :, 0], pred_trajectories[i, :, 1],
                'r--', linewidth=1.5, label='Predicted Trajectory', zorder=1)

    agent_id = agent_ids[i] if agent_ids else i

    plt.title(f'Trajectory Prediction - Agent {agent_id} (Scenario {scenario_idx})', fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate (m)', fontsize=12)
    plt.ylabel('Y Coordinate (m)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.close()


def predict_and_evaluate(model, dataloader, device, config, output_dir, num_visualizations=10, target_agent_id=None):
    """进行预测并评估模型性能"""

    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []
    scenario_count = 0

    prediction_results = {
        'scenarios': [],
        'agent_mapping': {}
    }

    print("开始预测和评估...")

    with torch.no_grad():
        for batch_obs_traj, batch_pred_traj, batch_mask, batch_agent_ids, batch_agent_types, batch_obs_lens, batch_pred_lens in tqdm(dataloader, desc="预测中"):

            batch_size = batch_obs_traj.size(0)

            # 数据预处理和归一化
            first_pos_list = []
            for b in range(batch_size):
                valid_agents = torch.nonzero(batch_mask[b], as_tuple=True)[0]
                if valid_agents.numel() > 0:
                    norm_obs, first_pos = normalize_trajectories(batch_obs_traj[b, valid_agents], scale_factor=1000.0)
                    batch_obs_traj[b, valid_agents] = norm_obs
                    norm_pred = (batch_pred_traj[b, valid_agents] - first_pos) / 1000.0
                    batch_pred_traj[b, valid_agents] = norm_pred
                    first_pos_list.append(first_pos)
                else:
                    first_pos_list.append(None)

            # 移动数据到设备
            batch_obs_traj = batch_obs_traj.to(device)
            batch_pred_traj = batch_pred_traj.to(device)
            batch_mask = batch_mask.to(device)

            # 多模态预测
            num_total_trajectories = config['model']['num_targets'] * config['model']['num_paths_per_target']
            pred_trajectories, pred_probs = model.predict_multimodal(
                batch_obs_traj,
                mask=batch_mask,
                num_samples=num_total_trajectories
            )

            # 获取预测的目标位置和编码特征
            with torch.no_grad():
                _, _, _, enc_aggregated = model(
                    batch_obs_traj, mask=batch_mask, return_multimodal=True, training=False
                )

                # 提取运动特征
                obs_endpoints = batch_obs_traj[:, :, -1, :]
                velocities = batch_obs_traj[:, :, 1:, :] - batch_obs_traj[:, :, :-1, :]
                mean_velocity = velocities.mean(dim=2)
                accelerations = velocities[:, :, 1:, :] - velocities[:, :, :-1, :]
                mean_acceleration = accelerations.mean(dim=2)

                motion_features = torch.cat([
                    enc_aggregated, obs_endpoints, mean_velocity, mean_acceleration
                ], dim=-1)

                pred_targets, _ = model.target_predictor(motion_features, batch_mask)

            # 计算指标
            metrics = calculate_metrics(pred_trajectories, batch_pred_traj, pred_probs)
            all_metrics.append(metrics)

            # 保存预测结果
            scene_result = {
                'scenario_idx': scenario_count,
                'agent_ids': batch_agent_ids[b] if batch_agent_ids is not None else list(range(len(batch_pred_traj[b]))),
                'agent_types': batch_agent_types[b].cpu().numpy() if batch_agent_types is not None else ['UNKNOWN'] * len(batch_pred_traj[b]),
                'predicted_trajectories': pred_trajectories[b].cpu().numpy(),
                'ground_truth_trajectories': batch_pred_traj[b].cpu().numpy(),
                'predicted_targets': pred_targets[b].cpu().numpy(),
                'metrics': metrics
            }
            prediction_results['scenarios'].append(scene_result)

            # 反归一化数据用于可视化
            for b in range(batch_size):
                valid_agents = torch.nonzero(batch_mask[b], as_tuple=True)[0]
                if valid_agents.numel() > 0 and first_pos_list[b] is not None:
                    first_pos = first_pos_list[b].to(device)

                    # 反归一化所有数据
                    batch_obs_traj[b, valid_agents] = batch_obs_traj[b, valid_agents] * 1000.0 + first_pos
                    batch_pred_traj[b, valid_agents] = batch_pred_traj[b, valid_agents] * 1000.0 + first_pos

                    # 反归一化预测轨迹 
                    pred_trajectories[b, valid_agents] = pred_trajectories[b, valid_agents] * 1000.0 + first_pos.unsqueeze(1)

                    # 反归一化预测目标
                    num_targets = pred_targets.shape[2]
                    first_pos_expanded = first_pos.unsqueeze(1).expand(-1, num_targets, -1, -1).squeeze(2)
                    pred_targets[b, valid_agents] = pred_targets[b, valid_agents] * 1000.0 + first_pos_expanded

                # 可视化前N个场景（可选择特定ID的智能体）
                if scenario_count < num_visualizations:
                    save_path = os.path.join(output_dir, f'prediction_batch_{scenario_count:02d}.png')
                    visualize_predictions(
                        batch_obs_traj[b].cpu().numpy(),
                        pred_trajectories[b].cpu().numpy(),
                        batch_pred_traj[b].cpu().numpy(),
                        pred_targets[b].cpu().numpy(),  # 传入目标位置
                        save_path=save_path,
                        agent_ids=batch_agent_ids[b] if batch_agent_ids else None,
                        scenario_idx=scenario_count,
                        target_agent_id=target_agent_id,  # 传入目标智能体ID
                        valid_agents=valid_agents.cpu().numpy() if valid_agents.numel() > 0 else None  # 传入有效智能体索引
                    )

            scenario_count += batch_size

    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    # 保存详细的预测结果
    results_file = os.path.join(output_dir, 'detailed_predictions.pkl')
    with open(results_file, 'wb') as f:
        import pickle
        pickle.dump(prediction_results, f)

    # 打印预测结果摘要
    print_prediction_summary(prediction_results, output_dir)

    return avg_metrics


def print_prediction_summary(prediction_results, output_dir):
    """打印预测结果摘要"""
    print("\n" + "="*60)
    print("🎯 TNT轨迹预测结果摘要")
    print("="*60)

    total_scenarios = len(prediction_results['scenarios'])
    total_agents = sum(len(scene['agent_ids']) for scene in prediction_results['scenarios'])

    print(f"📊 总场景数: {total_scenarios}")
    print(f"👥 总智能体数: {total_agents}")
    print(f"📁 结果保存路径: {output_dir}")

    print("\n🔍 各场景预测详情:")
    print("-" * 40)

    for scene in prediction_results['scenarios'][:5]:  # 只显示前5个场景
        print(f"场景 {scene['scenario_idx']}:")
        for i, agent_id in enumerate(scene['agent_ids']):
            agent_type = scene['agent_types'][i] if i < len(scene['agent_types']) else 'UNKNOWN'
            ade = scene['metrics'].get('ADE', 0)
            fde = scene['metrics'].get('FDE', 0)
            print(f"  🤖 智能体 {agent_id} ({agent_type}): ADE={ade:.3f}, FDE={fde:.3f}")

    if total_scenarios > 5:
        print(f"  ... 还有 {total_scenarios - 5} 个场景")

    print("\n💾 保存的文件:")
    print("  📄 evaluation_results.txt - 评估指标")
    print("  🖼️  prediction_batch_*.png - 可视化结果")
    print("  📦 detailed_predictions.pkl - 详细预测数据")
    print("\n💡 使用提示:")
    print("  • 使用 --target-agent-id 参数查看特定智能体的预测")
    print("  • 详细预测数据包含每个智能体的完整轨迹信息")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='TNT轨迹预测模型推理')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--model-path', type=str, default='checkpoints/tnt_transformer/best_model.pth', help='模型权重文件路径')
    parser.add_argument('--test-data', type=str, default='data/test', help='测试数据目录')
    parser.add_argument('--output-dir', type=str, default='predictions', help='输出目录')
    parser.add_argument('--num-visualizations', type=int, default=10, help='可视化场景数量')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--target-agent-id', type=int, default=None, help='要预测的特定智能体ID（例如：100100）')

    args = parser.parse_args()

    # 加载模型
    model, device, config = load_model(args.config, args.model_path)

    # 加载预处理后的测试数据集
    if os.path.exists(config['data']['preprocessed_test']):
        print(f"从预处理文件加载测试数据: {config['data']['preprocessed_test']}")
        test_dataset = PreprocessedDataset(config['data']['preprocessed_test'])
    else:
        print("预处理测试数据文件不存在，将使用原始数据目录")
        test_dataset = MultiAgentTrajectoryDataset(
            data_dir=args.test_data,
            obs_len=config['data']['obs_len'],
            pred_len=config['data']['pred_len'],
            min_agents=config['data']['min_agents'],
            max_agents=config['data']['max_agents'],
            min_seq_len=config['data']['min_seq_len']
        )

    # 创建数据加载器
    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 进行预测和评估
    metrics = predict_and_evaluate(model, test_loader, device, config, args.output_dir, 
                                  args.num_visualizations, target_agent_id=args.target_agent_id)

    # 保存评估结果
    results_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("TNT Transformer 预测评估结果\n")
        f.write("=" * 40 + "\n")
        f.write(f"测试场景数量: {len(test_dataset)}\n")
        f.write(f"模型路径: {args.model_path}\n")
        f.write("\n评估指标:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\n")
        f.write("ADE: Average Displacement Error (平均位移误差)\n")
        f.write("FDE: Final Displacement Error (最终位移误差)\n")

    print("\n评估完成!")
    print("结果已保存到:", results_file)
    print("\n主要指标:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == '__main__':
    main()
