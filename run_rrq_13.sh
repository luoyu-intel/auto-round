#!/usr/bin/env bash

set -euo pipefail

# generate base w2a16g64 quantization results for a model, which will be used for the subsequent auto-rounding process

MODEL_PATH="${1:-}"
OUTPUT_ROOT="${2:-}"

echo "MODEL_PATH: $MODEL_PATH"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"  

if [[ -z "$MODEL_PATH" || -z "$OUTPUT_ROOT" ]]; then
	echo "Usage: $0 <model_path_or_model_card> <output_root_dir>"
	exit 1
fi

MODEL_NAME="$(basename "$MODEL_PATH")"
OUTPUT_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-W1G64"
OUTPUT_R0_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-W1A16-R0"
OUTPUT_R0_OUTPUT_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-W1A16-R0-W3A16"
OUTPUT_R0_OUTPUT_MERGE_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-W1A16-R0-W3A16-MERGE"

export AUTO_ROUND_IMATRIX_FILE="${OUTPUT_ROOT%/}/imatrix.pt"
# save the intermediate imatrix for auto-rounding
echo "Running: auto-round --model \"$MODEL_PATH\" --iters 0 --output_dir \"$OUTPUT_DIR\" --data_type int --bits 1 --group_size 64 --asym --format \"fake\""
python -m auto_round --model "$MODEL_PATH" --iters 0 --output_dir "$OUTPUT_DIR" --data_type int --bits 1 --group_size 64 --asym --format "fake"
unset AUTO_ROUND_IMATRIX_FILE

BASE_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_NAME}-w1g64"

python get_residual.py -i "$MODEL_PATH" -q "$BASE_OUTPUT_DIR" -o "${OUTPUT_R0_DIR}" --operation sub --device cpu

export AUTO_ROUND_LOAD_IMATRIX_FILE="${OUTPUT_ROOT%/}/imatrix.pt"
python -m auto_round --model "$OUTPUT_R0_DIR" --format "fake" --iters 0 --output_dir "$OUTPUT_R0_OUTPUT_DIR" --data_type int --bits 3 --group_size 64 --asym
unset AUTO_ROUND_LOAD_IMATRIX_FILE
R0_OUTPUT_DIR="${OUTPUT_R0_OUTPUT_DIR}/${MODEL_NAME}-W1A16-R0-w3g64"

python get_residual.py -i "$BASE_OUTPUT_DIR" -q "$R0_OUTPUT_DIR" -o "${OUTPUT_R0_OUTPUT_MERGE_DIR}" --operation add --device cpu

echo "Base quantization results are saved in ${OUTPUT_DIR}"
echo "R0 quantization results are saved in ${OUTPUT_R0_OUTPUT_DIR}"
echo "R0 merged quantization results are saved in ${OUTPUT_R0_OUTPUT_MERGE_DIR}"

# lm_eval --model hf --model_args pretrained=${OUTPUT_R0_OUTPUT_MERGE_DIR} --tasks piqa,mmlu,lambada,winogrande,hellaswag --device cuda --batch_size 16