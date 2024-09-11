#!/bin/bash
export HF_HOME=/lustre/fsw/portfolios/healthcareeng/users/nigeln/hf_home
export TRANSFORMERS_CACHE=/lustre/fsw/portfolios/healthcareeng/users/nigeln/hf_home

# Run instances on GPUs 0 to 7
# for i in {0..7}; do
#   CUDA_VISIBLE_DEVICES=$i python batch_imitate_episodes.py "configs/config_${i}.json" &
#   CUDA_VISIBLE_DEVICES=$i python batch_imitate_episodes.py "configs/config_grok_${i}.json" &
# done
# Peg transfer
CUDA_VISIBLE_DEVICES=0 python batch_imitate_episodes.py --config_file configs/config_3.json --task 10_pc_peg_transfer &
CUDA_VISIBLE_DEVICES=0 python batch_imitate_episodes.py --config_file configs/config_3.json --task 20_pc_peg_transfer &
CUDA_VISIBLE_DEVICES=1 python batch_imitate_episodes.py --config_file configs/config_3.json --task 30_pc_peg_transfer &
CUDA_VISIBLE_DEVICES=0 python batch_imitate_episodes.py --config_file configs/config_3.json --task 40_pc_peg_transfer &
CUDA_VISIBLE_DEVICES=1 python batch_imitate_episodes.py --config_file configs/config_3.json --task 50_pc_peg_transfer &
# liver needle lift
CUDA_VISIBLE_DEVICES=2 python batch_imitate_episodes.py --config_file configs/config_3.json --task 10_pc_liver_needle_lift &
CUDA_VISIBLE_DEVICES=2 python batch_imitate_episodes.py --config_file configs/config_3.json --task 20_pc_liver_needle_lift &
CUDA_VISIBLE_DEVICES=3 python batch_imitate_episodes.py --config_file configs/config_3.json --task 30_pc_liver_needle_lift &
CUDA_VISIBLE_DEVICES=2 python batch_imitate_episodes.py --config_file configs/config_3.json --task 40_pc_liver_needle_lift &
CUDA_VISIBLE_DEVICES=3 python batch_imitate_episodes.py --config_file configs/config_3.json --task 50_pc_liver_needle_lift &
# liver needle handover
CUDA_VISIBLE_DEVICES=4 python batch_imitate_episodes.py --config_file configs/config_3.json --task 10_pc_liver_needle_handover &
CUDA_VISIBLE_DEVICES=4 python batch_imitate_episodes.py --config_file configs/config_3.json --task 20_pc_liver_needle_handover &
CUDA_VISIBLE_DEVICES=5 python batch_imitate_episodes.py --config_file configs/config_3.json --task 30_pc_liver_needle_handover &
CUDA_VISIBLE_DEVICES=4 python batch_imitate_episodes.py --config_file configs/config_3.json --task 40_pc_liver_needle_handover &
CUDA_VISIBLE_DEVICES=5 python batch_imitate_episodes.py --config_file configs/config_3.json --task 50_pc_liver_needle_handover &
# suture pad
CUDA_VISIBLE_DEVICES=6 python batch_imitate_episodes.py --config_file configs/config_3.json --task 10_pc_suture_pad &
CUDA_VISIBLE_DEVICES=6 python batch_imitate_episodes.py --config_file configs/config_3.json --task 20_pc_suture_pad &
CUDA_VISIBLE_DEVICES=7 python batch_imitate_episodes.py --config_file configs/config_3.json --task 30_pc_suture_pad &
CUDA_VISIBLE_DEVICES=6 python batch_imitate_episodes.py --config_file configs/config_3.json --task 40_pc_suture_pad &
CUDA_VISIBLE_DEVICES=7 python batch_imitate_episodes.py --config_file configs/config_3.json --task 50_pc_suture_pad &
# Suture pad
# Wait for all background processes to complete
wait

echo "All processes completed."