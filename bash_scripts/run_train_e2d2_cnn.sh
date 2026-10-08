#!/bin/bash

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh

# Model arch
BLOCK_SIZE=4
EVAL_BLOCK_SIZE=4
HIDDEN_SIZE=2048
INTERMEDIATE_SIZE=6144
N_ENCODER_LAYERS=28
N_DECODER_LAYERS=4


# Hyperparameters
LR=1e-5
ALPHA_F=0.5
WARMUP_DURATION="100ba"
BATCH_SIZE=32
MICRO_BATCH_SIZE=1
MAX_DURATION="30000ba"

PRETRAINED_MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B-Base

TAG="e2d2"
ENC_LAYERS="enc${N_ENCODER_LAYERS}"
DEC_LAYERS="dec${N_DECODER_LAYERS}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME=cnn_block${BLOCK_SIZE}_lr${LR}_bsz${BATCH_SIZE}_warm${WARMUP_DURATION}_${ENC_LAYERS}_${DEC_LAYERS}_hidden${HIDDEN_SIZE}_inter${INTERMEDIATE_SIZE}_${TAG}_${TIMESTAMP}

GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -E 's/.*(A[0-9]+|H100|A6000).*/\1/' | head -n 1)
NUM_WORKERS=0

NUM_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

composer -n ${NUM_VISIBLE_DEVICES} scripts/composer_scripts/train_discrete_denoiser.py \
  run_name=${RUN_NAME} \
  pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
  dataset@train_dataset=cnn_dailymail_train \
  dataset@eval_dataset=cnn_dailymail_eval \
  composer.optimizer.lr=${LR} \
  composer.trainer.eval_interval="1000ba" \
  composer.trainer.max_duration=${MAX_DURATION} \
  composer.trainer.save_num_checkpoints_to_keep=1 \
  composer/lr_scheduler=cosine_annealing_with_warmup \
  composer.lr_scheduler.t_warmup=${WARMUP_DURATION} \
  composer.lr_scheduler.alpha_f=${ALPHA_F} \
  model=e2d2 \
  model.config.attn_backend="sdpa" \
  training.compile_backbone=false \
  model.config.length=1024 \
  model/backbone@model.config.backbone_config=llm_as_encoder_decoder_share_kv \
  model.config.backbone_config.use_encoder_causal_mask=false \
  model.config.backbone_config.num_encoder_layers=${N_ENCODER_LAYERS} \
  model.config.backbone_config.num_decoder_layers=${N_DECODER_LAYERS} \
  model.config.backbone_config.tie_encoder_decoder_weights=true \
  model.config.backbone_config.reinit_decoder=false \
  model.config.backbone_config.reinit_encoder=false \
  model.config.backbone_config.keep_top_decoder_layers=true \
  model.config.backbone_config.keep_top_encoder_layers=false \
  +model.config.backbone_config.hidden_size=${HIDDEN_SIZE} \
  +model.config.backbone_config.intermediate_size=${INTERMEDIATE_SIZE} \
  training.global_batch_size=${BATCH_SIZE} \
  training.grad_accum=$(( BATCH_SIZE / NUM_VISIBLE_DEVICES / MICRO_BATCH_SIZE )) \
  block_size=${BLOCK_SIZE} \
  eval_block_size=${EVAL_BLOCK_SIZE} \
  training.antithetic_sampling=false \
  hydra.run.dir=${PWD}/outputs/${RUN_NAME} \
  composer.trainer.save_interval="1000ba" \
  composer.loggers.name=${RUN_NAME} \
  train_dataloader.num_workers=${NUM_WORKERS} \
  eval_dataloader.batch_size=1 \
  composer.callbacks.hf_compatible_checkpointing.disable_hf=true
