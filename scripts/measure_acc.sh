#! /bin/bash

# Change this part and the script arguments accordingly !!
MODEL_LIST=(
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.2-1B"
)

SBVR_PATH_LIST=(
    "sbvr_models/meta-llama_Llama-3.2-3B_num_sum_4"
    "sbvr_models/meta-llama_Llama-3.1-8B_num_sum_2"
    "sbvr_models/meta-llama_Llama-3.2-1B_num_sum_4"
)

LOG_FILE_NAME_LIST=(
    "llama-3.2-3B_fp16"
    "llama-3.1-8B_fp16"
    "llama-3.2-1B_fp16"
)


length=${#MODEL_LIST[@]}
for (( i=0; i<${length}; i++ )); do
    MODEL=${MODEL_LIST[$i]}
    SBVR_PATH=${SBVR_PATH_LIST[$i]}
    LOG_FILE_NAME=${LOG_FILE_NAME_LIST[$i]}

    echo "Running SBVR for model: $MODEL with output_name: $LOG_FILE_NAME"

    python -m eval.measure_zero_shot_acc \
        --model_path "$MODEL" \
        --log_file_name "$LOG_FILE_NAME"
        # --sbvr_path "$SBVR_PATH" \
        # --use_llm_int8 \
done