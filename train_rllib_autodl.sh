#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

# 设置基本参数
ENV="cleanup"
MODEL="baseline"
ALGORITHM="PPO"
POLICY_MODE="decentralized"
CONTROLLER_TYPE="real-llm"  # choices=["rule-based", "real-llm"]
NUM_AGENTS=5

# 实验名称
EXP_NAME="cleanup_controller-type_step-num_v1_date"

# 训练参数
NUM_ENV_RUNNERS=2
NUM_ENVS_PER_ENV_RUNNER=4
ROLLOUT_FRAGMENT_LENGTH=200
TRAIN_BATCH_SIZE=16000
MINIBATCH_SIZE=2000
NUM_EPOCHS=8
STOP_TIMESTEPS=200000000
STORAGE_PATH="${STORAGE_PATH:-$HOME/ray_results}"

# 算力分配
# 主进程：
# 每个env_runner：

NUM_CPUS_PER_ENV_RUNNER=4
NUM_GPUS_PER_ENV_RUNNER=0
NUM_CPUS_FOR_MAIN_PROCESS=16
NUM_GPUS=1

# 学习率调度
LR_SCHEDULE_STEPS=(0 20000000)
LR_SCHEDULE_WEIGHTS=(0.00126 0.000012)

# 运行训练
python train_rllib.py \
    --exp_name $EXP_NAME \
    --env $ENV \
    --algorithm $ALGORITHM \
    --model $MODEL \
    --policy_mode $POLICY_MODE \
    --num_agents $NUM_AGENTS \
    --num_samples 1 \
    --seed 42 \
    --max_cycles 1000 \
    --num_env_runners $NUM_ENV_RUNNERS \
    --num_envs_per_env_runner $NUM_ENVS_PER_ENV_RUNNER \
    --num_cpus_per_env_runner $NUM_CPUS_PER_ENV_RUNNER \
    --num_gpus_per_env_runner $NUM_GPUS_PER_ENV_RUNNER \
    --num_cpus_for_main_process $NUM_CPUS_FOR_MAIN_PROCESS \
    --num_gpus $NUM_GPUS \
    --rollout_fragment_length $ROLLOUT_FRAGMENT_LENGTH \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --minibatch_size $MINIBATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr_schedule_steps "${LR_SCHEDULE_STEPS[@]}" \
    --lr_schedule_weights "${LR_SCHEDULE_WEIGHTS[@]}" \
    --entropy_coeff 0.00176 \
    --vf_loss_coeff 0.5 \
    --clip_param 0.2 \
    --grad_clip 40.0 \
    --lstm_hidden_size 128 \
    --use_controller \
    --controller_f_step 80 \
    --controller_type $CONTROLLER_TYPE \
    --checkpoint_freq 50 \
    --stop_timesteps $STOP_TIMESTEPS \
    --enable_tensorboard \
    --tensorboard_port 6010 \
    --storage_path "$STORAGE_PATH"
