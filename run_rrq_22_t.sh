#!/usr/bin/env bash

set -euo pipefail

# generate base w2a16g64 quantization results for a model, which will be used for the subsequent auto-rounding process

MODEL_PATH="${1:-}"
OUTPUT_ROOT="${2:-}"
SCHEME="${3:-W2A16G64}"
DEFAULT_SUFFIX="${SCHEME,,}"
DEFAULT_SUFFIX="${DEFAULT_SUFFIX//a16/}"
SUFFIX="${4:-$DEFAULT_SUFFIX}"
echo "MODEL_PATH: $MODEL_PATH"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"  
echo "SCHEME: $SCHEME"
echo "SUFFIX: $SUFFIX"

if [[ -z "$MODEL_PATH" || -z "$OUTPUT_ROOT" ]]; then
	echo "Usage: $0 <model_path_or_model_card> <output_root_dir>"
	exit 1
fi

MODEL_NAME="$(basename "$MODEL_PATH")"
OUTPUT_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-${SCHEME}"
OUTPUT_R0_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-${SCHEME}-R0"
OUTPUT_R0_OUTPUT_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-${SCHEME}-R0-${SCHEME}"
OUTPUT_R0_OUTPUT_MERGE_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-${SCHEME}-R0-${SCHEME}-MERGE"

export AUTO_ROUND_IMATRIX_FILE="${OUTPUT_ROOT%/}/imatrix.pt"
# save the intermediate imatrix for auto-rounding
auto-round --model "$MODEL_PATH" --scheme "$SCHEME"  --iters 0 --output_dir "$OUTPUT_DIR"
unset AUTO_ROUND_IMATRIX_FILE

BASE_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_NAME}-${SUFFIX}"
rm -rf "$BASE_OUTPUT_DIR"


auto-round-best --enable_alg_ext --lr 2e-3 --model "$MODEL_PATH" --scheme "$SCHEME" --format "fake" --output_dir "$OUTPUT_DIR"


python get_residual.py -i "$MODEL_PATH" -q "$BASE_OUTPUT_DIR" -o "${OUTPUT_R0_DIR}" --operation sub --device cuda

export AUTO_ROUND_LOAD_IMATRIX_FILE="${OUTPUT_ROOT%/}/imatrix.pt"
auto-round --model "$OUTPUT_R0_DIR" --scheme "$SCHEME" --format "fake" --iters 0 --output_dir "$OUTPUT_R0_OUTPUT_DIR" --asym
unset AUTO_ROUND_LOAD_IMATRIX_FILE
R0_OUTPUT_DIR="${OUTPUT_R0_OUTPUT_DIR}/${MODEL_NAME}-${SCHEME}-R0-${SUFFIX}"

python get_residual.py -i "$BASE_OUTPUT_DIR" -q "$R0_OUTPUT_DIR" -o "${OUTPUT_R0_OUTPUT_MERGE_DIR}" --operation add --device cuda

echo "Base quantization results are saved in ${OUTPUT_DIR}"
echo "R0 quantization results are saved in ${OUTPUT_R0_OUTPUT_DIR}"
echo "R0 merged quantization results are saved in ${OUTPUT_R0_OUTPUT_MERGE_DIR}"

lm_eval --model hf --model_args pretrained=${OUTPUT_R0_OUTPUT_MERGE_DIR} --tasks piqa,mmlu,lambada,winogrande,hellaswag --device cuda --batch_size 16