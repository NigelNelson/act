#!/bin/bash
export HF_HOME=/lustre/fsw/portfolios/healthcareeng/users/nigeln/hf_home
export TRANSFORMERS_CACHE=/lustre/fsw/portfolios/healthcareeng/users/nigeln/hf_home

# Run instances on GPUs 0 to 7
# for i in {0..7}; do
#   CUDA_VISIBLE_DEVICES=$i python batch_imitate_episodes.py "configs/config_${i}.json" &
#   CUDA_VISIBLE_DEVICES=$i python batch_imitate_episodes.py "configs/config_grok_${i}.json" &
# done

CUDA_VISIBLE_DEVICES=0 python batch_imitate_episodes.py --config_file configs/config_3.json --task real_lift &
CUDA_VISIBLE_DEVICES=2 python batch_imitate_episodes.py --config_file configs/config_4.json --task real_lift &
CUDA_VISIBLE_DEVICES=4 python batch_imitate_episodes.py --config_file configs/config_3.json --task real_lift2 &
CUDA_VISIBLE_DEVICES=6 python batch_imitate_episodes.py --config_file configs/config_4.json --task real_lift2 &

# Wait for all background processes to complete
wait

echo "All processes completed."