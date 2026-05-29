#!/bin/bash

# Set experiment name
EXP_NAME="cleanup_reward_rulebased_80step_v2_7-16"

# Set environment and model configurations
ENV="cleanup"
MODEL="baseline"
ALGORITHM="PPO"
NUM_AGENTS=5
STOP_TIMESTEPS=30000000
POLICY_MODE="decentralized"  # choices=["centralized", "decentralized", "two_policies"]
CONTROLLER_TYPE="rule-based" # choices=["rule-based", "real-llm", "random"]
# Set hyperparameters

NUM_ENV_RUNNERS=0  # Optimized for multi-GPU parallelism, formerly NUM_WORKERS
NUM_ENVS_PER_ENV_RUNNER=4  # Increased parallelism for environment instances, formerly NUM_ENVS_PER_WORKER
ROLL_OUT_FRAGMENT_LENGTH=300
TRAIN_BATCH_SIZE=16000
MINIBATCH_SIZE=2000  
NUM_EPOCHS=8          
CHECKPOINT_FREQ=50      # save per N iter

ENTROPY_COEFF=0.00176

# 学习率调度 (Learning Rate Schedule)
LR_SCHEDULE_STEPS=(0 20000000)
LR_SCHEDULE_WEIGHTS=(0.00126 0.000012)

GRAD_CLIP=40.0

#PPO Epochs 或 Optimization Epochs

# Set GPU configuration
NUM_CPUS_PER_ENV_RUNNER=8  # Formerly CPUS_PER_ENV_RUNNER
NUM_GPUS_PER_ENV_RUNNER=0  # Formerly GPUS_PER_ENV_RUNNER
NUM_CPUS_FOR_MAIN_PROCESS=32 # Formerly CPUS_FOR_MAIN_PROCESS
NUM_GPUS=1  # Formerly GPUS_FOR_DRIVER

#HORIZON=50 # short episode length, and use soft-horizon

# Set up Ray configuration
export RAY_MEMORY=90000000000  # 160000000000 大约是给了 Ray 149 GB 的共享内存 # Example memory allocation for Ray workers

# Run training with optimizations for multi-GPU setup
python train_rllib.py \
  --exp_name $EXP_NAME \
  --env $ENV \
  --model $MODEL \
  --algorithm $ALGORITHM \
  --policy_mode $POLICY_MODE \
  --num_agents $NUM_AGENTS \
  --num_env_runners $NUM_ENV_RUNNERS \
  --num_envs_per_env_runner $NUM_ENVS_PER_ENV_RUNNER \
  --rollout_fragment_length $ROLL_OUT_FRAGMENT_LENGTH \
  --train_batch_size $TRAIN_BATCH_SIZE \
  --minibatch_size $MINIBATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --checkpoint_freq $CHECKPOINT_FREQ \
  --stop_timesteps $STOP_TIMESTEPS \
  --num_cpus_per_env_runner $NUM_CPUS_PER_ENV_RUNNER \
  --num_gpus_per_env_runner $NUM_GPUS_PER_ENV_RUNNER \
  --num_cpus_for_main_process $NUM_CPUS_FOR_MAIN_PROCESS \
  --num_gpus $NUM_GPUS \
  --entropy_coeff $ENTROPY_COEFF \
  --lr_schedule_steps "${LR_SCHEDULE_STEPS[@]}" \
  --lr_schedule_weights "${LR_SCHEDULE_WEIGHTS[@]}" \
  --grad_clip $GRAD_CLIP \
  --clip_param 0.2 \
  --vf_loss_coeff 0.5 \
  --lstm_hidden_size 128 \
  --controller_f_step 50 \
  --controller_type $CONTROLLER_TYPE \
  --max_cycles 1000 \
  --use_controller