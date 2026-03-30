#!/bin/bash

COMPRESSED_WEIGHT_DIR_PREFIX="./compressed_weights"
SAVE_DIR_PREFIX="./sbvr_models"

mkdir -p "$COMPRESSED_WEIGHT_DIR_PREFIX"
mkdir -p "$SAVE_DIR_PREFIX"

python -m sbvr_utils.comp_to_model \
    --compressed_weight_path "$COMPRESSED_WEIGHT_DIR_PREFIX" \
    --save_dir_path "$SAVE_DIR_PREFIX" \
    