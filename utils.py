import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import pickle
import random

import IPython
e = IPython.embed

MAX_EPISODE_LENGTH = 522
class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = random.choice([True, False])

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'data_{episode_id}.hdf5')
        start_ts = 0
        unique_idx = 0
        with h5py.File(dataset_path, 'r') as root:
            base_path = f'data/demo_0'
            is_sim = False

            # Find the last unique action
            _original_action = root[f'{base_path}/action'][()]
            num_original_actions = len(_original_action)
            last_unique_action = _original_action[-1]
            unique_idx = num_original_actions
            for i in range(MAX_EPISODE_LENGTH - 2, -1, -1):
                if not np.array_equal(_original_action[i], last_unique_action):
                    break
                else:
                    unique_idx = i

            original_action_shape = root[f'{base_path}/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(unique_idx)
            
            # Get observations and actions
            qpos = root[f'{base_path}/observations/qpos'][start_ts:]
            qvel = root[f'{base_path}/observations/qvel'][start_ts]
            image_dict = {cam_name: root[f'{base_path}/observations/images/{cam_name}'][start_ts:] for cam_name in self.camera_names}
            if is_sim:
                action = root[f'{base_path}/action'][start_ts:]
            else:
                action = root[f'{base_path}/action'][max(0, start_ts - 1):]

        self.is_sim = is_sim
        
        # Pad or trim action, qpos, and image_dict to MAX_EPISODE_LENGTH
        action_len = min(len(action), MAX_EPISODE_LENGTH)
        padded_action = np.zeros((MAX_EPISODE_LENGTH, *original_action_shape[1:]), dtype=np.float32)
        padded_action[:action_len] = action[:action_len]
        if action_len < MAX_EPISODE_LENGTH:
            padded_action[action_len:] = action[-1]  # Repeat last action

        padded_qpos = np.zeros((MAX_EPISODE_LENGTH, *qpos.shape[1:]), dtype=np.float32)
        qpos_len = min(len(qpos), MAX_EPISODE_LENGTH)
        padded_qpos[:qpos_len] = qpos[:qpos_len]
        if qpos_len < MAX_EPISODE_LENGTH:
            padded_qpos[qpos_len:] = qpos[-1]  # Repeat last qpos

        padded_image_dict = {}
        for cam_name in self.camera_names:
            cam_images = image_dict[cam_name]
            padded_cam_images = np.zeros((MAX_EPISODE_LENGTH, *cam_images.shape[1:]), dtype=np.uint8)
            img_len = min(len(cam_images), MAX_EPISODE_LENGTH)
            padded_cam_images[:img_len] = cam_images[:img_len]
            if img_len < MAX_EPISODE_LENGTH:
                padded_cam_images[img_len:] = cam_images[-1]  # Repeat last image
            padded_image_dict[cam_name] = padded_cam_images

        # Update is_pad logic
        is_pad = np.ones(MAX_EPISODE_LENGTH, dtype=bool)
        is_pad[:action_len] = False
        pad_idx = min(unique_idx - start_ts, action_len)
        is_pad[pad_idx:] = True

        # Prepare image data
        all_cam_images = np.stack([padded_image_dict[cam_name] for cam_name in self.camera_names], axis=1)

        # Construct observations
        image_data = torch.from_numpy(all_cam_images).float()
        qpos_data = torch.from_numpy(padded_qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad)

        # Channel last to channel first for images
        image_data = torch.einsum('t k h w c -> t k c h w', image_data)

        # Normalize data
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):

    # Lists to store the means and stds of each episode
    action_means = []
    action_stds = []
    qpos_means = []
    qpos_stds = []

    example_qpos = None

    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'data_{episode_idx}.hdf5')
        # print(f"Trying to load: {dataset_path}")
        with h5py.File(dataset_path, 'r') as root:
            base_path = f'data/demo_0'

            qpos = root[f'{base_path}/observations/qpos'][()]
            action = root[f'{base_path}/action'][()]
            # print(f"Episode {episode_idx} action length: {len(action)}")

            # Calculate mean and std for this episode's action data
            action_mean = np.mean(action, axis=0)
            action_std = np.std(action, axis=0)

            # Calculate mean and std for this episode's qpos data
            qpos_mean = np.mean(qpos, axis=0)
            qpos_std = np.std(qpos, axis=0)

            # Store the means and stds for this episode
            action_means.append(action_mean)
            action_stds.append(action_std)
            qpos_means.append(qpos_mean)
            qpos_stds.append(qpos_std)

            if example_qpos is None:
                example_qpos = qpos

    # Convert the lists to numpy arrays
    action_means = np.array(action_means)
    action_stds = np.array(action_stds)
    qpos_means = np.array(qpos_means)
    qpos_stds = np.array(qpos_stds)

    # Calculate the overall mean and std across all episodes
    action_mean = torch.tensor(np.mean(action_means, axis=0))
    action_std = torch.tensor(np.mean(action_stds, axis=0))
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping to avoid very small std

    qpos_mean = torch.tensor(np.mean(qpos_means, axis=0))
    qpos_std = torch.tensor(np.mean(qpos_stds, axis=0))
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping to avoid very small std

    stats = {
        "action_mean": action_mean.numpy(),
        "action_std": action_std.numpy(),
        "qpos_mean": qpos_mean.numpy(),
        "qpos_std": qpos_std.numpy(),
        "example_qpos": example_qpos
    }

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    # print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    # norm_stats = get_norm_stats(dataset_dir, num_episodes)
    # TODO undo hardcode
    # Specify the file path to load the stats
    file_path = "dataset_stats_250.pkl"

    # Load the stats dictionary from the file
    with open(file_path, "rb") as file:
        norm_stats = pickle.load(file)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# dataset_dir = "C:/Users/NVIDIA/holoscan-dev/orbit-surgical-nv/logs/robomimic/Isaac-Lift-Needle-PSM-IK-Rel-v0-old"
# stats = get_norm_stats(dataset_dir, 250)

# stats_path = 'dataset_stats_250.pkl'
# with open(stats_path, 'wb') as f:
#     pickle.dump(stats, f)