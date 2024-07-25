import torch
import numpy as np
import os
import pickle
import argparse
from flask import Flask, request, jsonify
from einops import rearrange

from constants import SIM_TASK_CONFIGS
from utils import set_seed
from policy import ACTPolicy

app = Flask(__name__)

def load_model(ckpt_dir, ckpt_name, policy_config, camera_names):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = ACTPolicy(policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(f'Loaded: {ckpt_path}')
    print(f'Loading status: {loading_status}')
    policy.cuda()
    policy.eval()

    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    return policy, stats

def pre_process(s_qpos, stats):
    # print(s_qpos)
    # print(stats['qpos_mean'])
    # print(stats['qpos_std'])
    return (s_qpos - stats['qpos_mean']) / stats['qpos_std']

def post_process(a, stats):
    return a * stats['action_std'] + stats['action_mean']

def get_image(image_data, camera_names):
    curr_images = []
    for cam_name in camera_names:
        _img = np.array(image_data[cam_name])
        curr_image = rearrange(_img, 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    qpos = data['qpos']
    image_data = data['image_data']

    qpos = pre_process(np.array(qpos), stats)
    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
    curr_image = get_image(image_data, camera_names)

    with torch.inference_mode():
        all_actions = policy(qpos, curr_image)
        # raw_action = all_actions[:, 0]  # We're only interested in the first action
        raw_action = all_actions # We're only interested in the first action

    # raw_action = raw_action.squeeze(0).cpu().numpy()
    raw_action = raw_action.cpu().numpy()
    action = post_process(raw_action, stats)

    return jsonify({'action': action.tolist()})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    args = parser.parse_args()

    set_seed(args.seed)

    task_config = SIM_TASK_CONFIGS[args.task_name]
    camera_names = task_config['camera_names']

    policy_config = {
        'lr': 1e-4,  # You might want to make this configurable
        'num_queries': args.chunk_size,  # You might want to make this configurable
        'kl_weight': args.kl_weight,  # You might want to make this configurable
        'hidden_dim': args.hidden_dim,  # You might want to make this configurable
        'dim_feedforward': args.dim_feedforward,  # You might want to make this configurable
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': camera_names,
        'policy_class' : "ACT",
        "seed": args.seed
    }

    policy, stats = load_model(args.ckpt_dir, 'policy_best.ckpt', policy_config, camera_names)

    app.run(host='0.0.0.0', port=8080)