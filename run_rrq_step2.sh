#!/usr/bin/env bash

set -euo pipefail

# run the residual auto-rounding flow from an existing base quantization result and imatrix file

MODEL_PATH="${1:-}"
OUTPUT_ROOT="${2:-}"
SCHEME="${3:-W2A16G64}"
BASE_OUTPUT_DIR="${4:-}"
IMATRIX_FILE="${5:-}"
DEFAULT_SUFFIX="${SCHEME,,}"
DEFAULT_SUFFIX="${DEFAULT_SUFFIX//a16/}"
SUFFIX="${6:-$DEFAULT_SUFFIX}"
echo "MODEL_PATH: $MODEL_PATH"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"  
echo "SCHEME: $SCHEME"
echo "BASE_OUTPUT_DIR: $BASE_OUTPUT_DIR"
echo "IMATRIX_FILE: $IMATRIX_FILE"
echo "SUFFIX: $SUFFIX"

if [[ -z "$MODEL_PATH" || -z "$OUTPUT_ROOT" || -z "$BASE_OUTPUT_DIR" || -z "$IMATRIX_FILE" ]]; then
	echo "Usage: $0 <model_path_or_model_card> <output_root_dir> [scheme] <base_output_dir> <imatrix_file> [suffix]"
	exit 1
fi

MODEL_NAME="$(basename "$MODEL_PATH")"
OUTPUT_R0_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-${SCHEME}-R0"
OUTPUT_R0_OUTPUT_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-${SCHEME}-R0-${SCHEME}"
OUTPUT_R0_OUTPUT_MERGE_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-${SCHEME}-R0-${SCHEME}-MERGE"

python get_residual.py -i "$MODEL_PATH" -q "$BASE_OUTPUT_DIR" -o "${OUTPUT_R0_DIR}" --operation sub --device cpu

export AUTO_ROUND_LOAD_IMATRIX_FILE="$IMATRIX_FILE"
python -m auto_round --model "$OUTPUT_R0_DIR" --scheme "$SCHEME" --format "fake" --iters 0 --output_dir "$OUTPUT_R0_OUTPUT_DIR" --asym --low_gpu_mem_usage
unset AUTO_ROUND_LOAD_IMATRIX_FILE
R0_OUTPUT_DIR="${OUTPUT_R0_OUTPUT_DIR}/${MODEL_NAME}-${SCHEME}-R0-${SUFFIX}"

python get_residual.py -i "$BASE_OUTPUT_DIR" -q "$R0_OUTPUT_DIR" -o "${OUTPUT_R0_OUTPUT_MERGE_DIR}" --operation add --device cpu

echo "Base quantization results are loaded from ${BASE_OUTPUT_DIR}"
echo "Imatrix file is loaded from ${IMATRIX_FILE}"
echo "R0 quantization results are saved in ${OUTPUT_R0_OUTPUT_DIR}"
echo "R0 merged quantization results are saved in ${OUTPUT_R0_OUTPUT_MERGE_DIR}"

lm_eval --model hf --model_args pretrained=${OUTPUT_R0_OUTPUT_MERGE_DIR} --tasks piqa,mmlu,lambada,winogrande,hellaswag,arc_easy,arc_challenge --device cuda --batch_size 16