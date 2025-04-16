#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS


MODEL_PATH="$1"

echo $MODEL_PATH

BSZ="$2"
LEN="$3"
CLIP="$4"
LR="$5"
N="$6"
EN_COEFF="$7"
KL_COEFF="$8"
TMP="$9"
PPO_BSZ="${10}"

mkdir ${MODEL_PATH}/grpo_cg_${BSZ}_${LEN}_${CLIP}_${LR}_${EN_COEFF}_${KL_COEFF}_${TMP}_${PPO_BSZ}

# Train over a single node, 8 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/train.parquet \
    data.val_files=data/aime.parquet \
    data.train_batch_size=${BSZ} \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=${LEN} \
    actor_rollout_ref.model.mamba=True \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_BSZ} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEFF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${EN_COEFF} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.temperature=${TMP} \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=${N} \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.grad_clip=${CLIP} \
    algorithm.kl_ctrl.kl_coef=${KL_COEFF} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='M1' \
    trainer.val_generations_to_log_to_wandb=1 \
    trainer.experiment_name=grpo_cg_packfix_${BSZ}_${LEN}_${CLIP}_${LR}_${EN_COEFF}_${KL_COEFF}_${TMP}_${PPO_BSZ}_new_veRL\
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=2 \
    trainer.test_freq=2 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${MODEL_PATH}/grpo_cg_packfix_${BSZ}_${LEN}_${CLIP}_${LR}_${EN_COEFF}_${KL_COEFF}_${TMP}_${PPO_BSZ}_new_veRL \
    trainer.total_epochs=3
