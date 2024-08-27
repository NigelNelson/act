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
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, use_pointcloud=False):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.use_pointcloud = use_pointcloud
        self.__getitem__(0)  # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir)
        start_ts = 0
        unique_idx = 0

        with zarr.open(dataset_path, 'r') as root:
            episode_ends = root['meta/episode_ends']
            end = episode_ends[episode_id]
            start = 0 if episode_id == 0 else episode_ends[episode_id - 1]

            is_sim = True

            ##### Find the last unique action so we don't make start_ts too large #########
            _original_action = root['/data/actions'][start:end]

            num_original_actions = len(_original_action)
            last_unique_action = _original_action[-1]
            unique_idx = num_original_actions
            for i in range(109 - 2, -1, -1):
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
            qpos = root['/data/joint_pos'][start:end][start_ts]
            qvel = root['/data/joint_vel'][start:end][start_ts]
            image_dict = dict()
            # TODO: uncomment this
            # for cam_name in self.camera_names:
            #     image_dict[cam_name] = root[f'{base_path}/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = _original_action[start_ts:]
                action_len = episode_len - start_ts
            else:
                action =_original_action[max(0, start_ts - 1):] # hack, to make timesteps more aligned
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

        if self.use_pointcloud:
            starting_cloud = root['/data/point_cloud']
            print(f'starting_cloud: {starting_cloud.shape}')
            # Load and process point cloud data
            point_cloud = root['/data/point_cloud'][start:end][start_ts]

            print(f'point_cloud: {point_cloud.shape}')

            # Apply unit sphere normalization
            normalized_point_cloud_xyz = apply_unit_sphere_normalization(
                point_cloud,
                self.norm_stats["point_cloud_mean"],
                self.norm_stats["max_distance"]
            )

            input_data = torch.from_numpy(normalized_point_cloud_xyz).float()
        else:
            # Process image data as before
            all_cam_images = []
            for cam_name in self.camera_names:
                all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=0)
            input_data = torch.from_numpy(all_cam_images.astype(np.uint8))
            input_data = torch.einsum('k h w c -> k c h w', input_data)
            input_data = input_data / 255.0

        # construct observations
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        #print shapes for all data
        print(f'input_data: {input_data.shape}')
        print(f'qpos_data: {qpos_data.shape}')
        print(f'action_data: {action_data.shape}')
        print(f'is_pad: {is_pad.shape}')

        return input_data, qpos_data, action_data, is_pad


def apply_unit_sphere_normalization(point_cloud, point_cloud_mean, max_distance):
    """
    Apply unit sphere normalization to a given point cloud.

    Args:
    - point_cloud: The point cloud data to normalize (numpy array of shape [num_points, num_dimensions]).
    - point_cloud_mean: The mean of the point cloud (numpy array of shape [num_dimensions]).
    - max_distance: The maximum distance used for scaling (a scalar).

    Returns:
    - normalized_point_cloud: The normalized point cloud (numpy array of shape [num_points, num_dimensions]).
    """
    # Step 1: Center the point cloud by subtracting the mean
    point_cloud_centered = point_cloud - point_cloud_mean

    # Step 2: Scale by the max distance
    normalized_point_cloud = point_cloud_centered / max_distance

    return normalized_point_cloud


def get_norm_stats(dataset_dir, num_episodes, use_pointcloud=False):

    # Lists to store the means and stds of each episode
    action_means = []
    action_stds = []
    qpos_means = []
    qpos_stds = []
    point_cloud_means = []
    point_cloud_stds = []
    max_distances = []

    example_qpos = None

    dataset_path = os.path.join(dataset_dir)
    with zarr.open(dataset_path, 'r') as root:
        episode_ends = root['meta/episode_ends']
        start = 0
        for episode_idx in range(num_episodes):
            base_path = f'data/demo_0'

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

            if use_pointcloud:
                point_cloud = root['/data/point_cloud'][start:episode_ends[episode_idx]]
                point_cloud = point_cloud[:, :3]

                # Center the point cloud
                point_cloud_centered = point_cloud - np.mean(point_cloud, axis=(0, 1), keepdims=True)

                # Calculate the distance of each point from the origin
                distances = np.linalg.norm(point_cloud_centered, axis=-1)

                # Find the maximum distance (radius of the smallest sphere containing the point cloud)
                max_distance = np.max(distances)
                max_distances.append(max_distance)

                # Store mean and std of the original point cloud (before normalization)
                point_cloud_means.append(np.mean(point_cloud, axis=(0, 1)))
                point_cloud_stds.append(np.std(point_cloud, axis=(0, 1)))

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

    if use_pointcloud:
        point_cloud_mean = np.mean(point_cloud_means, axis=0)
        point_cloud_std = np.mean(point_cloud_stds, axis=0)
        max_distance = np.mean(max_distances)

        stats.update({
            "point_cloud_mean": point_cloud_mean,
            "point_cloud_std": point_cloud_std,
            "max_distance": max_distance  # This is the scale factor for unit sphere normalization
        })

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, use_pointcloud=False):
    # print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.88
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes, use_pointcloud=use_pointcloud)
    # TODO undo hardcode
    # Specify the file path to load the stats
    # file_path = "dataset_stats_250.pkl"

    # Load the stats dictionary from the file
    # with open(file_path, "rb") as file:
    #     norm_stats = pickle.load(file)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, use_pointcloud)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, use_pointcloud)
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