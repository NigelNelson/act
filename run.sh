#!/bin/bash

# Run the first instance on GPU 0 in the background
CUDA_VISIBLE_DEVICES=0 python wandb_sweep_same.py &

# Sleep for 30 seconds
sleep 30

# Run the second instance on GPU 0 in the background
CUDA_VISIBLE_DEVICES=0 python wandb_sweep_same.py &

# Run instances on GPUs 1 to 7 without sleeps
for i in {1..7}; do
  CUDA_VISIBLE_DEVICES=$i python wandb_sweep_same.py &
  CUDA_VISIBLE_DEVICES=$i python wandb_sweep_same.py &
done

# Wait for all background processes to complete
wait