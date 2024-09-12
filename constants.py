import pathlib
### Task parameters
DATA_DIR = '<put your data dir here>'
SIM_TASK_CONFIGS = {
    'rgb_tissue_lift_10': {
        'dataset_dir': '/ACT_Training/lift_tissue/rgb/2024-09-10-Isaac-Lift-Tissue-PSM-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'episode_len': 600,
        'total_episodes': 10,
        'dual_arm': False,
        'camera_names': ['image']
    },
    'rgb_tissue_lift_20': {
        'dataset_dir': '/ACT_Training/lift_tissue/rgb/2024-09-10-Isaac-Lift-Tissue-PSM-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'episode_len': 600,
        'total_episodes': 20,
        'dual_arm': False,
        'camera_names': ['image']
    },
    'rgb_tissue_lift_30': {
        'dataset_dir': '/ACT_Training/lift_tissue/rgb/2024-09-10-Isaac-Lift-Tissue-PSM-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'episode_len': 600,
        'total_episodes': 30,
        'dual_arm': False,
        'camera_names': ['image']
    },
    'rgb_tissue_lift_40': {
        'dataset_dir': '/ACT_Training/lift_tissue/rgb/2024-09-10-Isaac-Lift-Tissue-PSM-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'episode_len': 600,
        'total_episodes': 40,
        'dual_arm': False,
        'camera_names': ['image']
    },
    'rgb_tissue_lift_50': {
        'dataset_dir': '/ACT_Training/lift_tissue/rgb/2024-09-10-Isaac-Lift-Tissue-PSM-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'episode_len': 600,
        'total_episodes': 52,
        'dual_arm': False,
        'camera_names': ['image']
    },
    'rgb_liver_needle_lift_10': {
        'dataset_dir': '/ACT_Training/liver_needle_lift/2024-09-05-Isaac-Lift-Needle-PSM-Back-Muscle-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 10,
        'episode_len': 600,
        'dual_arm': False,
        'camera_names': ['image']
    },
    'rgb_liver_needle_lift_20': {
        'dataset_dir': '/ACT_Training/liver_needle_lift/2024-09-05-Isaac-Lift-Needle-PSM-Back-Muscle-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 20,
        'episode_len': 600,
        'dual_arm': False,
        'camera_names': ['image']
    },
    'rgb_liver_needle_lift_30': {
        'dataset_dir': '/ACT_Training/liver_needle_lift/2024-09-05-Isaac-Lift-Needle-PSM-Back-Muscle-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 30,
        'episode_len': 600,
        'dual_arm': False,
        'camera_names': ['image']
    },
    'rgb_liver_needle_lift_40': {
        'dataset_dir': '/ACT_Training/liver_needle_lift/2024-09-05-Isaac-Lift-Needle-PSM-Back-Muscle-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 40,
        'episode_len': 600,
        'dual_arm': False,
        'camera_names': ['image']
    },
    'rgb_liver_needle_lift_50': {
        'dataset_dir': '/ACT_Training/liver_needle_lift/2024-09-05-Isaac-Lift-Needle-PSM-Back-Muscle-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 50,
        'episode_len': 600,
        'dual_arm': False,
        'camera_names': ['image']
    },
    'rgb_liver_needle_handover_10': {
        'dataset_dir': '/ACT_Training/liver_needle_handover/2024-09-05-Isaac-Handover-Needle-Dual-PSM-Back-Muscle-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 10,
        'episode_len': 600,
        'dual_arm': True,
        'camera_names': ['image']
    },
    'rgb_liver_needle_handover_20': {
        'dataset_dir': '/ACT_Training/liver_needle_handover/2024-09-05-Isaac-Handover-Needle-Dual-PSM-Back-Muscle-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 20,
        'episode_len': 600,
        'dual_arm': True,
        'camera_names': ['image']
    },
    'rgb_liver_needle_handover_30': {
        'dataset_dir': '/ACT_Training/liver_needle_handover/2024-09-05-Isaac-Handover-Needle-Dual-PSM-Back-Muscle-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 30,
        'episode_len': 600,
        'dual_arm': True,
        'camera_names': ['image']
    },
    'rgb_liver_needle_handover_40': {
        'dataset_dir': '/ACT_Training/liver_needle_handover/2024-09-05-Isaac-Handover-Needle-Dual-PSM-Back-Muscle-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 40,
        'episode_len': 600,
        'dual_arm': True,
        'camera_names': ['image']
    },
    'rgb_liver_needle_handover_50': {
        'dataset_dir': '/ACT_Training/liver_needle_handover/2024-09-05-Isaac-Handover-Needle-Dual-PSM-Back-Muscle-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 50,
        'episode_len': 600,
        'dual_arm': True,
        'camera_names': ['image']
    },
    'rgb_peg_transfer_10': {
        'dataset_dir': '/ACT_Training/peg_transfer/2024-09-06-Isaac-Transfer-Block-PSM-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 10,
        'episode_len': 600,
        'dual_arm': False,
        'camera_names': ['image']
    },
    'rgb_peg_transfer_20': {
        'dataset_dir': '/ACT_Training/peg_transfer/2024-09-06-Isaac-Transfer-Block-PSM-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 20,
        'episode_len': 600,
        'dual_arm': False,
        'camera_names': ['image']
    },
    'rgb_peg_transfer_30': {
        'dataset_dir': '/ACT_Training/peg_transfer/2024-09-06-Isaac-Transfer-Block-PSM-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 30,
        'episode_len': 600,
        'dual_arm': False,
        'camera_names': ['image']
    },
    'rgb_peg_transfer_40': {
        'dataset_dir': '/ACT_Training/peg_transfer/2024-09-06-Isaac-Transfer-Block-PSM-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 40,
        'episode_len': 600,
        'dual_arm': False,
        'camera_names': ['image']
    },
    'rgb_peg_transfer_50': {
        'dataset_dir': '/ACT_Training/peg_transfer/2024-09-06-Isaac-Transfer-Block-PSM-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 50,
        'episode_len': 600,
        'dual_arm': False,
        'camera_names': ['image']
    },
    'rgb_suture_pad_10': {
        'dataset_dir': '/ACT_Training/suture_pad/2024-09-06-Isaac-Handover-Needle-Suture_Pad-Dual-PSM-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 10,
        'episode_len': 600,
        'dual_arm': True,
        'camera_names': ['image']
    },
    'rgb_suture_pad_20': {
        'dataset_dir': '/ACT_Training/suture_pad/2024-09-06-Isaac-Handover-Needle-Suture_Pad-Dual-PSM-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 20,
        'episode_len': 600,
        'dual_arm': True,
        'camera_names': ['image']
    },
    'rgb_suture_pad_30': {
        'dataset_dir': '/ACT_Training/suture_pad/2024-09-06-Isaac-Handover-Needle-Suture_Pad-Dual-PSM-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 30,
        'episode_len': 600,
        'dual_arm': True,
        'camera_names': ['image']
    },
    'rgb_suture_pad_40': {
        'dataset_dir': '/ACT_Training/suture_pad/2024-09-06-Isaac-Handover-Needle-Suture_Pad-Dual-PSM-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 40,
        'episode_len': 600,
        'dual_arm': True,
        'camera_names': ['image']
    },
    'rgb_suture_pad_50': {
        'dataset_dir': '/ACT_Training/suture_pad/2024-09-06-Isaac-Handover-Needle-Suture_Pad-Dual-PSM-IK-Rel-v0-cleaned',
        'num_episodes': 52,
        'total_episodes': 50,
        'episode_len': 600,
        'dual_arm': True,
        'camera_names': ['image']
    },
    'needle_handover': {
        'dataset_dir': '/data',
        'num_episodes': 50,
        'episode_len': 469,
        'camera_names': ['image', 'wrist_image_right', 'wrist_image_left']
    },
    'needle_lift2':{
        'dataset_dir': '/data',
        'num_episodes': 250,
        'episode_len': 522,
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
