import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import pickle
import random

import IPython
e = IPython.embed

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
        sample_full_episode = False

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'data_{episode_id}.hdf5')
        start_ts = 0
        unique_idx = 0
        with h5py.File(dataset_path, 'r') as root:
            base_path = f'data/demo_0'
            is_sim = True

            ##### Find the last unique action so we don't make start_ts too large #########
            _original_action = root[f'{base_path}/action'][()]
            num_original_actions = len(_original_action)
            last_unique_action = _original_action[-1]
            unique_idx = num_original_actions
            for i in range(270 - 2, -1, -1):
                if not np.array_equal(_original_action[i], last_unique_action):
                    break
                else:
                    # print(f"Action at index {i} is the same as the last unique action")
                    unique_idx = i
            ################################################################################


            original_action_shape = root[f'{base_path}/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(unique_idx)
            # get observation at start_ts only
            qpos = root[f'{base_path}/observations/qpos'][start_ts]
            qvel = root[f'{base_path}/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'{base_path}/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root[f'{base_path}/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root[f'{base_path}/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action

        # Updated is_pad logic
        is_pad = np.ones(episode_len)
        is_pad[:action_len] = 0
        # 0 0 0 0 1 1 1
        
        pad_idx = unique_idx - start_ts
        pad_idx = min(pad_idx, action_len)
        is_pad[pad_idx:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images.astype(np.uint8))
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        assert image_data.dtype == torch.uint8

        # normalize image and change dtype to float
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
    train_ratio = 0.88
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