#!/usr/bin/env bash
# Multi-scale + flip, sliding-window evaluation (the protocol used for all reported results).
# Usage: bash eval.sh <config> <checkpoint.pth> [num_gpus]
#   bash eval.sh local_configs.NYUDepthv2.DFormerv2_S_text checkpoints/trained/NYUDepthv2_DFormerv2_S_text.pth
# --pad_SUNRGBD pads the SUN RGB-D images to a common size; it is ignored for NYUDepthv2.
CONFIG=${1:?config required}
CHECKPOINT=${2:?checkpoint required}
GPUS=${3:-1}
PORT=${PORT:-29198}

# Avoids the Hugging Face tokenizers warning when dataloader workers fork (only the main process tokenises)
export TOKENIZERS_PARALLELISM=true

PYTHONPATH="$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=1 \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/eval.py \
    --config=$CONFIG \
    --gpus=$GPUS \
    --sliding \
    --no-compile \
    --syncbn \
    --mst \
    --amp \
    --pad_SUNRGBD \
    --continue_fpath=$CHECKPOINT
