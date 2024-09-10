#!/bin/bash
export HF_HOME=/lustre/fsw/portfolios/healthcareeng/users/nigeln/hf_home
export TRANSFORMERS_CACHE=/lustre/fsw/portfolios/healthcareeng/users/nigeln/hf_home

# Run instances on GPUs 0 to 7
# for i in {0..7}; do
#   CUDA_VISIBLE_DEVICES=$i python batch_imitate_episodes.py "configs/config_${i}.json" &
#   CUDA_VISIBLE_DEVICES=$i python batch_imitate_episodes.py "configs/config_grok_${i}.json" &
# done
CUDA_VISIBLE_DEVICES=$i python batch_imitate_episodes.py --config_file configs/config_3.json --task 10_pc_peg_transfer &
CUDA_VISIBLE_DEVICES=$i python batch_imitate_episodes.py --config_file configs/config_3.json --task 20_pc_peg_transfer &
CUDA_VISIBLE_DEVICES=$i python batch_imitate_episodes.py --config_file configs/config_3.json --task 30_pc_peg_transfer &
CUDA_VISIBLE_DEVICES=$i python batch_imitate_episodes.py --config_file configs/config_3.json --task 40_pc_peg_transfer &
CUDA_VISIBLE_DEVICES=$i python batch_imitate_episodes.py --config_file configs/config_3.json --task 50_pc_peg_transfer &
# Wait for all background processes to complete
wait

echo "All processes completed."