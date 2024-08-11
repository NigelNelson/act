import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import json
from types import SimpleNamespace

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos
from constants import SIM_TASK_CONFIGS
import wandb
from grokfast import gradfilter_ma, gradfilter_ema

# from sim_env import BOX_POSE

import IPython
e = IPython.embed

def main(json_config):
    wandb_id = f"{json_config.alpha}_{json_config.lamb}_{json_config.learning_rate}_{json_config.kl_weight}_{json_config.chunk_size}_{json_config.batch_size}_act_needle-lift"
    wandb.init(project="ACT-training", config=json_config, entity="nigelnel", id=wandb_id, resume="allow")
    set_seed(0)

    task_config = {
        'dataset_dir': '/data',
        'num_episodes': 150,
        'episode_len': 522,
        'camera_names': ['image', 'wrist_image']
    }
    camera_names = task_config['camera_names']

    checkpoint_dir = f"/lustre/fsw/portfolios/healthcareeng/users/nigeln/vr_act_needle_lift_weights/lr_{json_config.learning_rate}_kl_{json_config.kl_weight}_chunk_{json_config.chunk_size}_b{json_config.batch_size}_alpha{json_config.alpha}_lamb{json_config.lamb}"
    # checkpoint_dir = "./tmppp"
    args = {
        'lr': json_config.learning_rate,  # You might want to make this configurable
        'num_queries': json_config.chunk_size,  # You might want to make this configurable
        'chunk_size': json_config.chunk_size,       
        'kl_weight': json_config.kl_weight,  # You might want to make this configurable
        'hidden_dim': 512,  # You might want to make this configurable
        "batch_size": json_config.batch_size,
        "num_epochs": 10000,
        'dim_feedforward': 3200,  # You might want to make this configurable
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': camera_names,
        'policy_class' : "ACT",
        "seed": 0,
        "eval": False,
        "ckpt_dir": checkpoint_dir,
        "onscreen_render": False,
        "task_name": "needle_lift2",
        "temporal_agg": True
    }

    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if True: # TODO clean up
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 4 # x_pos, y_pos, z_pos, x_rot, y_rot, z_rot, gripper_bool
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                        'num_queries': args['chunk_size'],
                        'kl_weight': args['kl_weight'],
                        'hidden_dim': args['hidden_dim'],
                        'dim_feedforward': args['dim_feedforward'],
                        'lr_backbone': lr_backbone,
                        'backbone': backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'camera_names': camera_names,
                        }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                        'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }


    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    # Training loop evals
    start_epoch = 0
    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    epoch_val_loss = 0.0
    epoch_train_loss = 0.0

    # Check if there's a checkpoint to resume from
    latest_ckpt = find_latest_checkpoint(ckpt_dir, seed)
    grads = None
    if latest_ckpt:
        start_epoch, train_history, validation_history, best_ckpt_info, grads = load_checkpoint(latest_ckpt, policy, optimizer)
        print(f'Resuming from epoch {start_epoch}')
    else:
        print('Starting from scratch..\n\n')

    # GrokFast
    alpha = json_config.alpha
    lamb = json_config.lamb

    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f'\nEpoch {epoch}')
        wandb_summary = {}
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        wandb_summary = {}
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
            wandb_summary[k] = v.item()
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            if json_config.alpha > 0:
                grads = gradfilter_ema(policy, grads=grads, alpha=alpha, lamb=lamb)
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))


        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)
        wandb_summary['train_loss'] = epoch_train_loss
        wandb_summary['epoch'] = epoch
        wandb_summary['val_loss'] = epoch_val_loss
        wandb.log(wandb_summary)

        # Save checkpoint every 250 steps
        if epoch % 250 == 0:
            save_checkpoint(epoch, policy, optimizer, train_history, validation_history, best_ckpt_info, ckpt_dir, seed, grads)

        if epoch % 500 == 0 and epoch > 3000:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)


    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def save_checkpoint(epoch, policy, optimizer, train_history, validation_history, best_ckpt_info, ckpt_dir, seed, grads):
    checkpoint = {
        'epoch': epoch,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_history': train_history,
        'validation_history': validation_history,
        'best_ckpt_info': best_ckpt_info,
        'grads': grads
    }
    ckpt_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch}_seed_{seed}.pth')
    torch.save(checkpoint, ckpt_path)
    print(f'Checkpoint saved at epoch {epoch}')


def load_checkpoint(ckpt_path, policy, optimizer):
    checkpoint = torch.load(ckpt_path)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return (checkpoint['epoch'], checkpoint['train_history'], 
            checkpoint['validation_history'], checkpoint['best_ckpt_info'], checkpoint['grads'])

def find_latest_checkpoint(ckpt_dir, seed):
    checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith(f'checkpoint_epoch_') and f.endswith(f'_seed_{seed}.pth')]
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2]))
    return os.path.join(ckpt_dir, latest_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run main function with JSON config")
    parser.add_argument('config_file', type=str, help='Path to JSON config file')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config_dict = json.load(f)
    
    config = SimpleNamespace(**config_dict)
    main(config)