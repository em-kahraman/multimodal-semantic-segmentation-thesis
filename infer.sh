#!/usr/bin/env bash
# Save colourised predictions of a trained model to output/.
# Usage: bash infer.sh <config> <checkpoint.pth> [num_gpus]
CONFIG=${1:?config required}
CHECKPOINT=${2:?checkpoint required}
GPUS=${3:-1}
PORT=${PORT:-29958}

# Avoids the Hugging Face tokenizers warning when dataloader workers fork (only the main process tokenises)
export TOKENIZERS_PARALLELISM=true

PYTHONPATH="$(dirname $0)":$PYTHONPATH \
    torchrun \
    --nnodes=1 \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    utils/infer.py \
    --config=$CONFIG \
    --continue_fpath=$CHECKPOINT \
    --save_path "output/" \
    --gpus=$GPUS
