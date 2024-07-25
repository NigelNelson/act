import os
import numpy as np
import h5py
import time
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2

MAX_TIMESTEPS = 290

def convert_numpy_to_hdf5(input_dir, output_dir, num_episodes, camera_names):
    os.makedirs(output_dir, exist_ok=True)

    for episode_idx in range(num_episodes):
        input_file = os.path.join(input_dir, f"episode_{episode_idx}.npz.npy")
        
        # Load numpy data
        episode_data = np.load(input_file, allow_pickle=True)
        # episode_data = np_data['arr_0']
        print(len(episode_data))
        # continue

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
            state = step['state']
            
            data_dict['/observations/qpos'].append(state)
            data_dict['/observations/qvel'].append(np.zeros_like(state))  # Assuming qvel is not available
            data_dict['/action'].append(step['action'])
            for cam_name in camera_names:
                if cam_name == 'image':
                    data_dict[f'/observations/images/{cam_name}'].append(step['image'])
                else:
                    pixels_to_remove = (640 - 512) // 2
                    wrist_image = step[cam_name][:, pixels_to_remove:-pixels_to_remove]
                    wrist_image = cv2.resize(wrist_image, (512, 512), interpolation=cv2.INTER_LINEAR)
                    assert wrist_image.shape == (512, 512, 3)
                    data_dict[f'/observations/images/{cam_name}'].append(wrist_image)

        # Pad up to MAX_TIMESTEPS
        difference = MAX_TIMESTEPS - len(data_dict['/action'])
        if difference > 0:
            default_state = episode_data[-1]['state']

            default_img = episode_data[-1]['image']
            default_wrist_img = episode_data[-1]['wrist_image']
            pixels_to_remove = (640 - 512) // 2
            default_wrist_img = default_wrist_img[:, pixels_to_remove:-pixels_to_remove]
            default_wrist_img = cv2.resize(default_wrist_img, (512, 512), interpolation=cv2.INTER_LINEAR)
            assert default_wrist_img.shape == (512, 512, 3)
            for _ in range(difference):
                data_dict['/observations/qpos'].append(default_state)
                data_dict['/observations/qvel'].append(np.zeros_like(default_state))
                data_dict['/action'].append(np.array([0, 0, 0, -1]))

                data_dict[f'/observations/images/image'].append(default_img)
                data_dict[f'/observations/images/wrist_image'].append(default_wrist_img)

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
            
            qpos = obs.create_dataset('qpos', (max_timesteps, 4))
            qvel = obs.create_dataset('qvel', (max_timesteps, 4))
            action = root.create_dataset('action', (max_timesteps, 4))

            for name, array in data_dict.items():
                root[name][...] = array
        
        print(f'Saving episode {episode_idx}: {time.time() - t0:.1f} secs')

    print(f'Saved to {output_dir}')

if __name__ == "__main__":
    config = {
        'dataset_dir': '/media/m2/holoscan-dev/holoscan-ml/robots/data',
        'num_episodes': 70,
        'camera_names': ['image', 'wrist_image']
    }

    input_dir = "/media/m2/holoscan-dev/holoscan-ml/orbit-surgical/data/needle_lift_take3"
    output_dir = config['dataset_dir']

    convert_numpy_to_hdf5(
        input_dir,
        output_dir,
        config['num_episodes'],
        config['camera_names']
    )

    print("Conversion complete!")