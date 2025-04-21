#!/bin/bash
#Training script config
python -W ignore main_represent.py --exp_name "resnet50nodown_fc_[False,True,False]_last_v_fixed" \
    --seed 42 \
    --num_workers 16 \
    --save_freq 50 \
    --print_freq 1\
    --method SelfCon \
    --data_folder "" \
    --dataset imaginet \
    --model resnet50nodown \
    --selfcon_pos "[False,True,False]" \
    --selfcon_arch "resnet" \
    --selfcon_size "fc" \
    --batch_size 200 \
    --optimizer_name "sgd" \
    --learning_rate 0.005 \
    --temp 0.07 \
    --epochs 400 \
    --warm \
    --cosine \
    --pretrained_imagenet \
    --grad_cache  \ # Optional if you have GPU with low memory
    --grad_cache_chunk_size 90 \ # The chunk size should be tested (it depends on the memory of the GPU)
