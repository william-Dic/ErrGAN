# !/bin/bash

v_file_name=trial_err_GAN_2
q_file_name=trial_err_Q_GAN

export MUJOCO_GL="osmesa"
export CUDA_VISIBLE_DEVICES=1
python Err_eval_GAN.py\
    --v_model_ckpt_path "/home/dodo/wgm/CL/BearRobot/BearRobot/Agent/deployment/experiments/libero/libero_goal/$v_file_name/latest.pth"\
    --q_model_ckpt_path "/home/dodo/wgm/CL/BearRobot/BearRobot/Agent/deployment/experiments/libero/libero_goal/$q_file_name/latest.pth"\
    --policy_ckpt_path "/home/dodo/ljx/BearRobot/experiments/libero/libero_goal/diffusion/resnet34_wstate_0613_T5_42/49999_0.6096223592758179.pth"\
    --statistic_path "/home/dodo/wgm/CL/BearRobot/BearRobot/Agent/deployment/experiments/libero/libero_goal/$v_file_name/statistics.json"\
    --num_episodes 1 \
    --eval_horizon 500 



