#!/bin/bash

MODEL_LIST=(
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.2-1B"
)

NUM_SUM_LIST=(
    4 2
)

COMPRESSED_WEIGHT_DIR_PREFIX="./compressed_weights"
SAVE_DIR_PREFIX="./sbvr_models"

mkdir -p "$COMPRESSED_WEIGHT_DIR_PREFIX"
mkdir -p "$SAVE_DIR_PREFIX"

for MODEL in "${MODEL_LIST[@]}"; do
    for NUM_SUM in "${NUM_SUM_LIST[@]}"; do
        echo "Running SBVR for model: $MODEL with num_sum: $NUM_SUM"
        
        COMPRESSED_WEIGHT_PATH="${COMPRESSED_WEIGHT_DIR_PREFIX}/${MODEL//\//_}_num_sum_${NUM_SUM}"
        SAVE_PATH="${SAVE_DIR_PREFIX}/${MODEL//\//_}_num_sum_${NUM_SUM}"

        python encode_llama.py \
            --model_path "$MODEL" \
            --num_sums "$NUM_SUM" \
            --compressed_weight_path "$COMPRESSED_WEIGHT_PATH" \
            --save_path "$SAVE_PATH"
    done
done