import os
import argparse
import torch
import pickle
from tqdm import tqdm
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from utils.data_process import MultiAgentTrajectoryDataset


def process_single_file(file_path, obs_len=20, pred_len=80, min_agents=1, max_agents=50, min_seq_len=60):
    scenes = []
    scene_agent_counts = []

    try:
        df = pd.read_csv(file_path, delimiter=',')
        required_columns = ['timestamp', 'id', 'x', 'y']
        if not all(col in df.columns for col in required_columns):
            return scenes, scene_agent_counts

        timestamps = sorted(df['timestamp'].unique())
        if len(timestamps) < min_seq_len:
            return scenes, scene_agent_counts

        total_len = len(timestamps)
        if total_len >= obs_len + pred_len:
            actual_obs_len = obs_len
            actual_pred_len = pred_len
        else:
            actual_obs_len = min(obs_len, total_len // 2)
            actual_pred_len = total_len - actual_obs_len

        max_scenes = 3
        scene_count = 0
        step_size = max(1, actual_obs_len // 2)

        for start_idx in range(0, len(timestamps) - (actual_obs_len + actual_pred_len) + 1, step_size):
            if scene_count >= max_scenes:
                break

            scene_timestamps = timestamps[start_idx:start_idx + actual_obs_len + actual_pred_len]
            agent_trajectories = defaultdict(list)
            agent_types = {}

            for ts in scene_timestamps:
                ts_data = df[df['timestamp'] == ts]
                for _, row in ts_data.iterrows():
                    agent_id = row['id']
                    agent_trajectories[agent_id].append([row['x'], row['y']])
                    if 'type' in row and agent_id not in agent_types:
                        agent_types[agent_id] = row['type']

            complete_agents = {}
            for agent_id, traj in agent_trajectories.items():
                traj_len = len(traj)
                if traj_len >= actual_obs_len:
                    if traj_len >= actual_obs_len + actual_pred_len:
                        complete_agents[agent_id] = traj[:actual_obs_len + actual_pred_len]
                    else:
                        obs_part = traj[:actual_obs_len]
                        pred_part = traj[actual_obs_len:]
                        while len(pred_part) < actual_pred_len:
                            pred_part.append(traj[-1])
                        complete_agents[agent_id] = obs_part + pred_part

            if len(complete_agents) < min_agents or len(complete_agents) > max_agents:
                continue

            scene_data = {
                'obs': [],
                'pred': [],
                'agent_ids': [],
                'agent_types': [],
                'obs_len': actual_obs_len,
                'pred_len': actual_pred_len
            }

            for agent_id, traj in complete_agents.items():
                traj = np.array(traj)
                scene_data['obs'].append(traj[:actual_obs_len])
                scene_data['pred'].append(traj[actual_obs_len:])
                scene_data['agent_ids'].append(agent_id)
                scene_data['agent_types'].append(agent_types.get(agent_id, 'UNKNOWN'))

            scene_data['obs'] = np.array(scene_data['obs'])
            scene_data['pred'] = np.array(scene_data['pred'])

            scenes.append(scene_data)
            scene_agent_counts.append(len(complete_agents))
            scene_count += 1

    except Exception:
        pass

    return scenes, scene_agent_counts


def process_file_batch(batch_files, obs_len, pred_len, min_agents, max_agents, min_seq_len):
    batch_scenes = []
    batch_counts = []

    for file_path in batch_files:
        file_scenes, file_counts = process_single_file(
            file_path, obs_len, pred_len, min_agents, max_agents, min_seq_len
        )
        batch_scenes.extend(file_scenes)
        batch_counts.extend(file_counts)

    return batch_scenes, batch_counts


def process_dataset_parallel(data_dir, args):
    if not data_dir:
        return None, None

    all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not all_files:
        return None, None

    batch_size = 20
    file_batches = [all_files[i:i + batch_size] for i in range(0, len(all_files), batch_size)]

    all_scenes = []
    all_counts = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        from functools import partial
        process_func = partial(
            process_file_batch,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            min_agents=args.min_agents,
            max_agents=args.max_agents,
            min_seq_len=args.min_seq_len
        )

        results = list(tqdm(
            executor.map(process_func, file_batches),
            total=len(file_batches),
            desc="Processing batches"
        ))

        for scenes, counts in results:
            all_scenes.extend(scenes)
            all_counts.extend(counts)

    return all_scenes, all_counts


def preprocess_data(args):
    os.makedirs(args.output_dir, exist_ok=True)

    if args.train_data:
        train_scenes, train_counts = process_dataset_parallel(args.train_data, args)
        if train_scenes is not None:
            with open(os.path.join(args.output_dir, 'train_data.pkl'), 'wb') as f:
                pickle.dump({'scenes': train_scenes, 'scene_agent_counts': train_counts}, f)

    if args.val_data:
        val_scenes, val_counts = process_dataset_parallel(args.val_data, args)
        if val_scenes is not None:
            with open(os.path.join(args.output_dir, 'val_data.pkl'), 'wb') as f:
                pickle.dump({'scenes': val_scenes, 'scene_agent_counts': val_counts}, f)

    if args.test_data:
        test_scenes, test_counts = process_dataset_parallel(args.test_data, args)
        if test_scenes is not None:
            with open(os.path.join(args.output_dir, 'test_data.pkl'), 'wb') as f:
                pickle.dump({'scenes': test_scenes, 'scene_agent_counts': test_counts}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='预处理轨迹数据')
    
    # 数据参数
    parser.add_argument('--train-data', type=str, default='data/train', help='训练数据目录')
    parser.add_argument('--val-data', type=str, default='data/val', help='验证数据目录')
    parser.add_argument('--test-data', type=str, default='data/predict', help='测试数据目录')
    parser.add_argument('--output-dir', type=str, default='processed_data', help='处理后数据保存目录')
    parser.add_argument('--obs-len', type=int, default=20, help='观测序列长度')
    parser.add_argument('--pred-len', type=int, default=80, help='预测序列长度')
    parser.add_argument('--min-agents', type=int, default=1, help='每个场景最少智能体数量')
    parser.add_argument('--max-agents', type=int, default=50, help='每个场景最多智能体数量')
    parser.add_argument('--min-seq-len', type=int, default=60, help='最小可接受的序列长度')
    args = parser.parse_args()

    preprocess_data(args) 