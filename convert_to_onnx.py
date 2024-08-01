import torch
import numpy as np
import os
import pickle
import argparse
from flask import Flask, request, jsonify
from einops import rearrange
from torchvision import transforms

from constants import SIM_TASK_CONFIGS
from utils import set_seed
from policy import ACTPolicy
import time

app = Flask(__name__)

def export_to_onnx(ckpt_dir, ckpt_name, policy_config, camera_names):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = ACTPolicy(policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(f'Loaded: {ckpt_path}')
    print(f'Loading status: {loading_status}')
    policy.cuda()
    policy.eval()

    batch_size = 1  # As you mentioned you want batch size of 1
    qpos_dim = 4  # Adjust this to match your robot's degrees of freedom
    image_shape = (3, 512, 512)  # Adjust if your image dimensions are different
    num_cameras = 2

    # Create dummy inputs
    dummy_qpos = torch.randn(batch_size, qpos_dim).cuda()
    dummy_image = torch.randn(batch_size, num_cameras, *image_shape).cuda()

    # Define a new wrapper class
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])

        def forward(self, qpos, image):
            image = self.normalize(image)
            a_hat, is_pad_hat, (mu, logvar) = self.model.model(qpos, image, env_state=None)
            return a_hat, is_pad_hat, mu, logvar

    # Create an instance of the wrapper
    onnx_wrapper = ONNXWrapper(policy).cuda()

    # Debugging step
    with torch.no_grad():
        outputs = onnx_wrapper(dummy_qpos, dummy_image)
        print("Number of outputs:", len(outputs))
        for i, output in enumerate(outputs):
            if output is None:
                continue
            print(f"Output {i} shape:", output.shape)

    # Export the model
    # Export the model
    torch.onnx.export(onnx_wrapper,  # The wrapper module
                    (dummy_qpos, dummy_image),  # Model inputs
                    "act_policy.onnx",  # Output file name
                    export_params=True,
                    opset_version=11,  # Use a recent opset version
                    do_constant_folding=True,
                    input_names=['qpos', 'image'],
                    output_names=['a_hat', 'is_pad_hat'],
                    dynamic_axes=None)


    print("ONNX export completed successfully.")


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

    export_to_onnx(args.ckpt_dir, 'policy_best.ckpt', policy_config, camera_names)