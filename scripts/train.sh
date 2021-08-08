#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--exp_id "default-setting" \
--mode train \
--start_iter 0 --end_iter 100000 \
--preload_dataset false --cache_dataset false \
--use_tensorboard true --save_loss true \
--dataset CelebA \
--batch_size 8 --img_size 256 \
--train_path ../celeba_hq_smiling/train \
--test_path ../celeba_hq_smiling/train \
--eval_path ../celeba_hq_smiling/train