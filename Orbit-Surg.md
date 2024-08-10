## Orbit-Surgical Workflow:


1. collect data using the `lift_needle_sm_rel.py` script within `orbit-surgical`
> Note: only 1 env at a time is currently supported
```bash
${IsaacLab_PATH}/isaaclab.sh -p source/standalone/workflows/act/lift_needle_sm_rel.py --num_envs 1 --num_demos 150
```
2. Train an ACT model within the `act` repo:
```bash
python3 imitate_episodes.py \
--task_name needle_lift2 \
--ckpt_dir ./model_ckpts_chunk15_b64_e8k_kl100_lr4 \
--policy_class ACT --kl_weight 100 --chunk_size 15 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
--num_epochs 8000  --lr 4e-5 \
--seed 0
```

3. convert to an onnx runtime file using `convert_to_onnx.py`
> The arguments that need to be accurate are the `ckpt_dir` and `chunk_size`, all of the other arguments are ignored when simply converting to ONNX. However, setting them to a value is required as existing class definitions are used from the ACT repo that expects these args.

> The script defaults to uses the `policy_best.ckpt`, but you can try different checkpoints by changing the last line of code

> The output model is saved to `<your_act_repo>/act_policy.onnx`
```bash
python3 convert_to_onnx.py \
--task_name needle_lift2 \
--ckpt_dir ./model_checkpoints_chunk15_b64_e8k_kl100_lr4_isaac \
--policy_class ACT --kl_weight 100 --chunk_size 15 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 \
--num_epochs 8000  --lr 4e-5 \
--seed 0
```

4. Run the ACT ONNX policy in `orbit-surgical
> Install pip dependencies in your orbit-surgical conda env:
```bash
pip install onnxruntime einops
```
> You must include the `model_path`,`stats_dir`, and `chunk_size` args:
```bash
${IsaacLab_PATH}/isaaclab.sh -p source/standalone/workflows/act/act_needle_lift_onnx.py \
--model_path=/media/m2/holoscan-dev/holoscan-ml/robots/act/act_policy.onnx \
--stats_dir=/media/m2/holoscan-dev/holoscan-ml/robots/act/model_checkpoints_chunk15_b64_e8k_kl100_lr4_isaac \
--chunk_size=15 \
--max_timesteps=165
```


 Note: Currently, the repo is configured to train the ACT policy using 2 images  + the robot 7-dim state array [abs_x_pos, abs_y_pos, abs_z_pos, abs_x_rot, abs_y_rot, abs_z_rot, gripper_state] as inputs. Then output a 7-dim action array [rel_x_pos, rel_y_pos, rel_z_pos, rel_x_rot, rel_y_rot, rel_z_rot, gripper_state].

#### TODO: complete explanation of how to edit the ACT policy
 To change the dimensions of the input and output actions:
 - update the data collection script
 - update the detr model INPUT_OUT_DIM [here](./detr/models/detr_vae.py)

 To change the images used:
 - update the data collection script to change the images captured but also the names of the images written to the DataCollector
 - update the dataset confit in [constants.py](./constants.py)
 - update the inference code