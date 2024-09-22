import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import pickle
import random
import zarr

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, noise_std=0.1, use_augmentation=False):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.use_augmentation = use_augmentation
        self.__getitem__(0)  # initialize self.is_sim
    def __len__(self):
        return len(self.episode_ids)

    def add_adaptive_noise(self, image, noise_level=0.05):
        # Perform histogram equalization
        img_eq = exposure.equalize_hist(image)
        
        # Generate noise
        noise = np.random.normal(0, noise_level, image.shape)
        
        # Scale noise based on local image intensity
        scaled_noise = noise * img_eq
        
        # Add scaled noise to the original image
        noisy_img = image + scaled_noise
        
        # Rescale the image to [0, 1] range
        noisy_img = exposure.rescale_intensity(noisy_img, out_range=(0, 1))
        
        return noisy_img

    def __getitem__(self, index):
        sample_full_episode = False

        episode_id = self.episode_ids[index]
        dataset_path = self.dataset_dir
        start_ts = 0
        unique_idx = 0

        with zarr.open(dataset_path, 'r') as root:
            episode_ends = root['meta/episode_ends'][:]
            end = episode_ends[episode_id]
            start = 0 if episode_id == 0 else episode_ends[episode_id - 1]

            is_sim = True

            _original_action = root['/data/actions'][start:end]
            num_original_actions = len(_original_action)
            last_unique_action = _original_action[-1]
            unique_idx = num_original_actions
            for i in range(num_original_actions - 2, -1, -1):
                if not np.array_equal(_original_action[i], last_unique_action):
                    break
                else:
                    # print(f"Action at index {i} is the same as the last unique action")
                    unique_idx = i
            ################################################################################


            original_action_shape = _original_action.shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(unique_idx)
            # get observation at start_ts only
            # User joint pos
            qpos = root['/data/joint_pos'][start:end][start_ts]
            image_dict = dict()

            for cam_name in self.camera_names:
                img = root[f'/data/{cam_name}'][start:end][start_ts]
                image_dict[cam_name] = img
            
            # get all actions after and including start_ts
            if is_sim:
                action = _original_action[start_ts:]
                action_len = episode_len - start_ts
            else:
                action = _original_action[max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action

        padded_action = padded_action[:15]

        # Updated is_pad logic
        is_pad = np.ones(episode_len)
        is_pad[:action_len] = 0
        # 0 0 0 0 1 1 1
        
        pad_idx = unique_idx - start_ts
        pad_idx = min(pad_idx, action_len)
        is_pad[pad_idx:] = 1

        is_pad = is_pad[:15]

        # new axis for different cameras
        # Process image data
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0).squeeze(0)
        image_data = torch.from_numpy(all_cam_images.astype(np.uint8))
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        assert image_data.dtype == torch.uint8
        image_data = image_data / 255.0

        # construct observations
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # normalize image and change dtype to float
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

    with zarr.open(dataset_dir, 'r') as root:
        episode_ends = root['meta/episode_ends'][:]
        start = 0
        for episode_idx in range(num_episodes):
            qpos = root['/data/joint_pos'][start:episode_ends[episode_idx]]
            action = root['/data/actions'][start:episode_ends[episode_idx]]

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

            start = episode_ends[episode_idx]

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


def load_data(dataset_dir, num_episodes, total_episodes, camera_names, batch_size_train, batch_size_val, use_augmentation=False):
    # print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.92
    shuffled_indices = np.random.permutation(num_episodes)
    shuffled_indices = shuffled_indices[:total_episodes]
    train_indices = shuffled_indices[:int(train_ratio * total_episodes)]
    val_indices = shuffled_indices[int(train_ratio * total_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    # TODO undo hardcode
    # Specify the file path to load the stats
    # file_path = "dataset_stats_250.pkl"

    # # Load the stats dictionary from the file
    # with open(file_path, "rb") as file:
    #     norm_stats = pickle.load(file)
    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, use_augmentation=use_augmentation)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, use_augmentation=use_augmentation)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=16, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=16, prefetch_factor=2)

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