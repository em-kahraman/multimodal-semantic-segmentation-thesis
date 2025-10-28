#!/usr/bin/env bash
# Usage: bash train.sh <config> [num_gpus]
#   bash train.sh local_configs.NYUDepthv2.DFormerv2_S_text 1   # thesis setup: 1 GPU
#   bash train.sh local_configs.SUNRGBD.DFormerv2_S_text 4      # thesis setup: 4 GPUs
# --pad_SUNRGBD pads the SUN RGB-D validation images to a common size; it is ignored for NYUDepthv2.
CONFIG=${1:-local_configs.NYUDepthv2.DFormerv2_S_text}
GPUS=${2:-1}
PORT=${PORT:-29158}

# Avoids the Hugging Face tokenizers warning when dataloader workers fork (only the main process tokenises)
export TOKENIZERS_PARALLELISM=true

PYTHONPATH="$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=1 \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/train.py \
    --config=$CONFIG \
    --gpus=$GPUS \
    --no-sliding \
    --no-compile \
    --syncbn \
    --mst \
    --no-amp \
    --val_amp \
    --pad_SUNRGBD \
    --use_seed
