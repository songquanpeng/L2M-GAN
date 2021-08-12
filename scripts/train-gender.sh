#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py \
--exp_id "default-setting-gender" \
--mode train \
--start_iter 0 --end_iter 100000 \
--preload_dataset false --cache_dataset false \
--use_tensorboard true --save_loss true \
--dataset CelebA \
--batch_size 8 --img_size 256 \
--train_path ./archive/celeba_hq_gender/train \
--eval_path ./archive/celeba_hq_gender/train \
--test_path ./archive/celeba_hq_gender/test
