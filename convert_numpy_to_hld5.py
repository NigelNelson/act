import os
import numpy as np
import h5py
import time
from scipy.spatial.transform import Rotation as R
import numpy as np

MAX_TIMESTEPS = 650

def convert_numpy_to_hdf5(input_dir, output_dir, num_episodes, camera_names):
    os.makedirs(output_dir, exist_ok=True)

    for episode_idx in range(num_episodes):
        input_file = os.path.join(input_dir, f"episode_{episode_idx}.npz.npy")
        
        # Load numpy data
        episode_data = np.load(input_file, allow_pickle=True)
        # episode_data = np_data['arr_0']

        # Prepare data dictionary
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # Fill data dictionary
        for step in episode_data:
            _state = step['state']
            rotation = R.from_quat(_state[3:7])
            rotation_vector = rotation.as_rotvec()
            state = np.concatenate([_state[:3], rotation_vector, np.array([_state[-1]])])
            
            data_dict['/observations/qpos'].append(state)
            data_dict['/observations/qvel'].append(np.zeros_like(state))  # Assuming qvel is not available
            data_dict['/action'].append(step['action'])
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(step['image'])

        # Pad up to MAX_TIMESTEPS
        difference = MAX_TIMESTEPS - len(data_dict['/action'])
        if difference > 0:
            _default_state = episode_data[-1]['state']
            _default_rotation = R.from_quat(_default_state[3:7])
            _default_rotation_vector = _default_rotation.as_rotvec()
            default_state = np.concatenate([_default_state[:3], _default_rotation_vector, np.array([_default_state[-1]])])\
            
            default_img = episode_data[-1]['image']
            for _ in range(difference):
                data_dict['/observations/qpos'].append(default_state)
                data_dict['/observations/qvel'].append(np.zeros_like(default_state))
                data_dict['/action'].append(np.array([0, 0, 0, 0, 0, 0, -1]))
                for cam_name in camera_names:
                    data_dict[f'/observations/images/{cam_name}'].append(default_img)

        # Convert lists to numpy arrays
        for key in data_dict:
            data_dict[key] = np.array(data_dict[key])

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(output_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = False  # Assuming this is real data, not simulation
            obs = root.create_group('observations')
            image = obs.create_group('images')
            
            max_timesteps = len(data_dict['/action'])
            
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 512, 512, 3), dtype='uint8',
                                         chunks=(1, 512, 512, 3))
            
            qpos = obs.create_dataset('qpos', (max_timesteps, 7))
            qvel = obs.create_dataset('qvel', (max_timesteps, 7))
            action = root.create_dataset('action', (max_timesteps, 7))

            for name, array in data_dict.items():
                root[name][...] = array
        
        print(f'Saving episode {episode_idx}: {time.time() - t0:.1f} secs')

    print(f'Saved to {output_dir}')

if __name__ == "__main__":
    config = {
        'dataset_dir': '/media/m2/holoscan-dev/holoscan-ml/robots/data',
        'num_episodes': 50,
        'camera_names': ['top']
    }

    input_dir = "/media/m2/holoscan-dev/holoscan-ml/orbit-surgical/data/needle_lift_dataset/rlds_dataset_builder/needle_dataset2/data"
    output_dir = config['dataset_dir']

    convert_numpy_to_hdf5(
        input_dir,
        output_dir,
        config['num_episodes'],
        config['camera_names']
    )

    print("Conversion complete!")