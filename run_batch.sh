#!/bin/bash
export HF_HOME=/lustre/fsw/portfolios/healthcareeng/users/nigeln/hf_home
export TRANSFORMERS_CACHE=/lustre/fsw/portfolios/healthcareeng/users/nigeln/hf_home

# Run instances on GPUs 0 to 7
# for i in {0..7}; do
#   CUDA_VISIBLE_DEVICES=$i python batch_imitate_episodes.py "configs/config_${i}.json" &
#   CUDA_VISIBLE_DEVICES=$i python batch_imitate_episodes.py "configs/config_grok_${i}.json" &
# done

CUDA_VISIBLE_DEVICES=0 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_tissue_lift_10 &
CUDA_VISIBLE_DEVICES=0 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_tissue_lift_20 &
CUDA_VISIBLE_DEVICES=0 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_tissue_lift_30 &
CUDA_VISIBLE_DEVICES=1 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_tissue_lift_40 &
CUDA_VISIBLE_DEVICES=2 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_tissue_lift_50 &

CUDA_VISIBLE_DEVICES=1 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_liver_needle_lift_10 &
CUDA_VISIBLE_DEVICES=1 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_liver_needle_lift_20 &
CUDA_VISIBLE_DEVICES=2 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_liver_needle_lift_30 &
CUDA_VISIBLE_DEVICES=7 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_liver_needle_lift_40 &
# CUDA_VISIBLE_DEVICES=1 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_liver_needle_lift_50 &

CUDA_VISIBLE_DEVICES=2 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_liver_needle_handover_10 &
CUDA_VISIBLE_DEVICES=3 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_liver_needle_handover_20 &
CUDA_VISIBLE_DEVICES=3 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_liver_needle_handover_30 &
CUDA_VISIBLE_DEVICES=7 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_liver_needle_handover_40 &
# CUDA_VISIBLE_DEVICES=2 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_liver_needle_handover_50 &

CUDA_VISIBLE_DEVICES=4 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_peg_transfer_10 &
CUDA_VISIBLE_DEVICES=4 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_peg_transfer_20 &
CUDA_VISIBLE_DEVICES=4 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_peg_transfer_30 &
CUDA_VISIBLE_DEVICES=5 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_peg_transfer_40 &
# CUDA_VISIBLE_DEVICES=3 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_peg_transfer_50 &

CUDA_VISIBLE_DEVICES=5 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_suture_pad_10 &
CUDA_VISIBLE_DEVICES=5 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_suture_pad_20 &
CUDA_VISIBLE_DEVICES=6 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_suture_pad_30 &
CUDA_VISIBLE_DEVICES=6 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_suture_pad_40 &
# CUDA_VISIBLE_DEVICES=4 python batch_imitate_episodes.py --config_file configs/config_3.json --task rgb_suture_pad_50 &

# Wait for all background processes to complete
wait

echo "All processes completed."