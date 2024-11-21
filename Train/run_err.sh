bs=64
ws=2
text_encoder=T5
save_file=trial_err_Q_GAN
#TODO 换个名字 不要覆盖了
for seed in 42; do
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=
torchrun --standalone --nnodes=1 --nproc_per_node=1 Err_train.py \
    --dataset 'libero_goal' \
    --algo_name 'trial and err' \
    --ddp False \
    --img_size 128 \
    --visual_encoder resnet34 \
    --visual_pretrain True \
    --text_encoder $text_encoder \
    --seed $seed \
    --ac_num 1 \
    --norm minmax \
    --norm_type bn \
    --add_spatial_coordinates False \
    --discretize_actions False \
    --s_dim 9 \
    --batch_size $bs \
    --world_size $ws \
    --lr 0.0001 \
    --resume None \
    --wandb True \
    --steps 200000 \
    --save True \
    --save_freq 20000 \
    --T 25 \
    --save_path ./experiments/libero/libero_goal/$save_file \
    --log_path ./experiments/libero/libero_goal/$save_file \
    --port 2060
done
