import pathlib

### Task parameters
DATA_DIR = '<put your data dir here>'
SIM_TASK_CONFIGS = {
    'pc_transfer_block': {
        'dataset_dir': '/media/m2/holoscan-dev/holoscan-ml/robots/orbit-surgical-nv/logs/dp3/Isaac-Transfer-Block-PSM-IK-Rel-v0/d3_2024-09-06-cleaned.zarr',
        'num_episodes': 52,
        'episode_len': 300,
        'camera_names': ['image'],
        'use_pointcloud': True,
        'dual_arm': False,
        'backbone': 'pointnet'
    },
    'pc_needle_lift_back_muscle': {
        'dataset_dir': '/media/m2/holoscan-dev/holoscan-ml/robots/orbit-surgical-nv/logs/dp3/Isaac-Lift-Needle-PSM-Back-Muscle-IK-Rel-v0/d3_2024-09-05_no-padding.zarr',
        'num_episodes': 52,
        'episode_len': 300,
        'camera_names': ['image'],
        'use_pointcloud': True,
        'dual_arm': False,
        'backbone': 'pointnet'
    },
    'pc_needle_handover_back_muscle': {
        'dataset_dir': '/media/m2/holoscan-dev/holoscan-ml/robots/orbit-surgical-nv/logs/dp3/Isaac-Handover-Needle-Dual-PSM-Back-Muscle-IK-Rel-v0/d3_2024-09-05_cleaned-even.zarr',
        'num_episodes': 52,
        'episode_len': 262,
        'camera_names': ['image'],
        'use_pointcloud': True,
        'dual_arm': True,
        'backbone': 'pointnet'
    },
    'pc_suture_pad': {
        'dataset_dir': '/media/m2/holoscan-dev/holoscan-ml/robots/orbit-surgical-nv/logs/dp3/Isaac-Handover-Needle-Suture_Pad-Dual-PSM-IK-Rel-v0/d3_2024-08-28-cleaned.zarr',
        'num_episodes': 44,
        'episode_len': 529,
        'camera_names': ['image'],
        'use_pointcloud': True,
        'dual_arm': True,
        'backbone': 'pointnet'
    },
    'pc_needle_lift': {
        'dataset_dir': '/media/m2/holoscan-dev/holoscan-ml/robots/orbit-surgical-nv/logs/dp3/Isaac-Transfer-Block-PSM-IK-Rel-v0/d3_2024-08-24-cleaned-even.zarr',
        'num_episodes': 52,
        'episode_len': 272,
        'camera_names': ['image'],
        'use_pointcloud': True,
        'dual_arm': True,
        'backbone': 'pointnet'
    },
    'needle_handover': {
        'dataset_dir': '/data',
        'num_episodes': 51,
        'episode_len': 442,
        'camera_names': ['image', 'wrist_image_right', 'wrist_image_left']
    },
    'needle_lift2':{
        'dataset_dir': '/data',
        'num_episodes': 51,
        'episode_len': 108,
        'camera_names': ['image', 'wrist_image']
    },
    'needle_lift':{
        'dataset_dir': '/media/m2/holoscan-dev/holoscan-ml/robots/data',
        'num_episodes': 50,
        'episode_len': 650,
        'camera_names': ['top']
    },

    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top']
    },
}

### Simulation envs fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
