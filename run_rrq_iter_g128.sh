#!/usr/bin/env bash

set -euo pipefail

START_TIME="$(date +%s)"

format_duration() {
	local total_seconds="$1"
	local hours=$((total_seconds / 3600))
	local minutes=$(((total_seconds % 3600) / 60))
	local seconds=$((total_seconds % 60))
	printf '%02d:%02d:%02d' "$hours" "$minutes" "$seconds"
}

# generate base w2a16g64 quantization results for a model, which will be used for the subsequent auto-rounding process

MODEL_PATH="${1:-}"
OUTPUT_ROOT="${2:-}"
SCHEME="${3:-W2A16}"
DEFAULT_SUFFIX="${SCHEME,,}"
DEFAULT_SUFFIX="${DEFAULT_SUFFIX//a16/}"
SUFFIX="w2g128"
ITERATION_COUNT="${4:-3}"
USE_LOW_GPU_MEM_USAGE="${5:-false}"
echo "MODEL_PATH: $MODEL_PATH"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"  
echo "SCHEME: $SCHEME"
echo "SUFFIX: $SUFFIX"
echo "ITERATION_COUNT: $ITERATION_COUNT"
echo "USE_LOW_GPU_MEM_USAGE: $USE_LOW_GPU_MEM_USAGE"

if [[ -z "$MODEL_PATH" || -z "$OUTPUT_ROOT" ]]; then
	echo "Usage: $0 <model_path_or_model_card> <output_root_dir> [scheme] [iteration_count] [use_low_gpu_mem_usage]"
	exit 1
fi

if ! [[ "$ITERATION_COUNT" =~ ^[1-9][0-9]*$ ]]; then
	echo "iteration_count must be a positive integer"
	exit 1
fi

LOW_GPU_MEM_USAGE_ARGS=()
case "$USE_LOW_GPU_MEM_USAGE" in
	true|TRUE|1|yes|YES)
		LOW_GPU_MEM_USAGE_ARGS+=(--low_gpu_mem_usage)
		;;
	false|FALSE|0|no|NO)
		;;
	*)
		echo "use_low_gpu_mem_usage must be one of: true, false, 1, 0, yes, no"
		exit 1
		;;
esac

MODEL_NAME="$(basename "$MODEL_PATH")"
OUTPUT_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-${SCHEME}"

export AUTO_ROUND_IMATRIX_FILE="${OUTPUT_ROOT%/}/imatrix.pt"
# save the intermediate imatrix for auto-rounding
python -m auto_round --model "$MODEL_PATH" --scheme "$SCHEME"  --iters 0 --output_dir "$OUTPUT_DIR" --group 128
unset AUTO_ROUND_IMATRIX_FILE

BASE_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_NAME}-${SUFFIX}"
rm -rf "$BASE_OUTPUT_DIR"

python -m auto_round --model "$MODEL_PATH" --scheme "$SCHEME"  --format "fake" --iters 200 --output_dir "$OUTPUT_DIR" "${LOW_GPU_MEM_USAGE_ARGS[@]}" --group 128

PREV_MERGE_DIR="$BASE_OUTPUT_DIR"
FINAL_OUTPUT_DIR="$BASE_OUTPUT_DIR"

for ((iteration = 0; iteration < ITERATION_COUNT; iteration++)); do
	OUTPUT_RESIDUAL_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-${SCHEME}-R${iteration}"
	OUTPUT_RESIDUAL_QUANT_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-${SCHEME}-R${iteration}-${SCHEME}"
	OUTPUT_RESIDUAL_MERGE_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-${SCHEME}-R${iteration}-${SCHEME}-MERGE"
	RESIDUAL_OUTPUT_DIR="${OUTPUT_RESIDUAL_QUANT_DIR}/${MODEL_NAME}-${SCHEME}-R${iteration}-${SUFFIX}"

	python get_residual.py -i "$MODEL_PATH" -q "$PREV_MERGE_DIR" -o "$OUTPUT_RESIDUAL_DIR" --operation sub --device cpu

	export AUTO_ROUND_LOAD_IMATRIX_FILE="${OUTPUT_ROOT%/}/imatrix.pt"
	python -m auto_round --model "$OUTPUT_RESIDUAL_DIR" --scheme "$SCHEME" --format "fake" --iters 0 --output_dir "$OUTPUT_RESIDUAL_QUANT_DIR" "${LOW_GPU_MEM_USAGE_ARGS[@]}" --group 128
	unset AUTO_ROUND_LOAD_IMATRIX_FILE

	python get_residual.py -i "$PREV_MERGE_DIR" -q "$RESIDUAL_OUTPUT_DIR" -o "$OUTPUT_RESIDUAL_MERGE_DIR" --operation add --device cpu

	echo "R${iteration} residual quantization results are saved in ${OUTPUT_RESIDUAL_QUANT_DIR}"
	echo "R${iteration} merged quantization results are saved in ${OUTPUT_RESIDUAL_MERGE_DIR}"

	PREV_MERGE_DIR="$OUTPUT_RESIDUAL_MERGE_DIR"
	FINAL_OUTPUT_DIR="$OUTPUT_RESIDUAL_MERGE_DIR"
done

END_TIME="$(date +%s)"
TOTAL_DURATION="$((END_TIME - START_TIME))"

echo "Base quantization results are saved in ${OUTPUT_DIR}"
echo "Final merged quantization results are saved in ${FINAL_OUTPUT_DIR}"
echo "Total elapsed time: $(format_duration "$TOTAL_DURATION") (${TOTAL_DURATION}s)"

