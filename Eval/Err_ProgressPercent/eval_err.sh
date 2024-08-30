#!/bin/bash

file_name=trial_err_LSTM_buffer_01

export MUJOCO_GL="osmesa"
export CUDA_VISIBLE_DEVICES=0
python Err_eval.py\
    --k 0.2 \
    --v_model_ckpt_path "/home/dodo/wgm/CL/BearRobot/BearRobot/Agent/deployment/experiments/libero/libero_goal/$file_name/latest.pth"\
    --policy_ckpt_path "/home/dodo/ljx/BearRobot/experiments/libero/libero_goal/diffusion/err_usage_T5/4999_3.333669900894165.pth"\
    --statistic_path "/home/dodo/wgm/CL/BearRobot/BearRobot/Agent/deployment/experiments/libero/libero_goal/$file_name/statistics.json"\
    --num_episodes 1\
    --eval_horizon 500


    # --policy_ckpt_path "/home/dodo/ljx/BearRobot/experiments/libero/libero_goal/diffusion/resnet34_wstate_0613_T5_42/49999_0.6096223592758179.pth"\

    