#!/usr/bin/env bash

set -euo pipefail

SCRIPT_START_TS="$(date +%s)"

log_time() {
	local message="$1"
	local now
	now="$(date '+%Y-%m-%d %H:%M:%S')"
	echo "[$now] $message"
}

print_elapsed() {
	local start_ts="$1"
	local end_ts
	local elapsed
	end_ts="$(date +%s)"
	elapsed=$((end_ts - start_ts))
	log_time "elapsed: ${elapsed}s"
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
log_time "script started"

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

print_cache_and_skip() {
	local step_name="$1"
	local output_path="$2"
	echo "[cache] Skip ${step_name}, output already exists: ${output_path}"
}

MODEL_NAME="$(basename "$MODEL_PATH")"
OUTPUT_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-${SCHEME}"
IMATRIX_FILE="${OUTPUT_ROOT%/}/imatrix.pt"
BASE_OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_NAME}-${SUFFIX}"

if [[ -e "$IMATRIX_FILE" ]]; then
	print_cache_and_skip "imatrix generation" "$IMATRIX_FILE"
else
	STEP_START_TS="$(date +%s)"
	log_time "start imatrix generation"
	export AUTO_ROUND_IMATRIX_FILE="$IMATRIX_FILE"
	# save the intermediate imatrix for auto-rounding
	python -m auto_round --model "$MODEL_PATH" --scheme "$SCHEME"  --iters 0 --output_dir "$OUTPUT_DIR"
	unset AUTO_ROUND_IMATRIX_FILE
	rm -rf ${BASE_OUTPUT_DIR}
	log_time "finish imatrix generation"
	print_elapsed "$STEP_START_TS"
fi


if [[ -e "$BASE_OUTPUT_DIR" ]]; then
	print_cache_and_skip "base auto-round-best" "$BASE_OUTPUT_DIR"
else
	STEP_START_TS="$(date +%s)"
	log_time "start base auto-round-best"
	auto-round-best --enable_alg_ext --lr 2e-3 --model "$MODEL_PATH" --scheme "$SCHEME" --format "fake" --output_dir "$OUTPUT_DIR" "${LOW_GPU_MEM_USAGE_ARGS[@]}"
	log_time "finish base auto-round-best"
	print_elapsed "$STEP_START_TS"
fi

PREV_MERGE_DIR="$BASE_OUTPUT_DIR"
FINAL_OUTPUT_DIR="$BASE_OUTPUT_DIR"

for ((iteration = 0; iteration < ITERATION_COUNT; iteration++)); do
	ITER_START_TS="$(date +%s)"
	log_time "start iteration R${iteration}"
	OUTPUT_RESIDUAL_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-${SCHEME}-R${iteration}"
	OUTPUT_RESIDUAL_QUANT_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-${SCHEME}-R${iteration}-${SCHEME}"
	OUTPUT_RESIDUAL_MERGE_DIR="${OUTPUT_ROOT%/}/${MODEL_NAME}-${SCHEME}-R${iteration}-${SCHEME}-MERGE"
	RESIDUAL_OUTPUT_DIR="${OUTPUT_RESIDUAL_QUANT_DIR}/${MODEL_NAME}-${SCHEME}-R${iteration}-${SUFFIX}"

	STEP_START_TS="$(date +%s)"
	log_time "R${iteration}: start residual subtraction"
	python get_residual.py -i "$MODEL_PATH" -q "$PREV_MERGE_DIR" -o "$OUTPUT_RESIDUAL_DIR" --operation sub --device cpu
	log_time "R${iteration}: finish residual subtraction"
	print_elapsed "$STEP_START_TS"

	if [[ -e "$RESIDUAL_OUTPUT_DIR" ]]; then
		print_cache_and_skip "R${iteration} residual auto_round" "$RESIDUAL_OUTPUT_DIR"
	else
		STEP_START_TS="$(date +%s)"
		log_time "R${iteration}: start residual auto_round"
		export AUTO_ROUND_LOAD_IMATRIX_FILE="$IMATRIX_FILE"
		python -m auto_round --model "$OUTPUT_RESIDUAL_DIR" --scheme "$SCHEME" --format "fake" --iters 0 --output_dir "$OUTPUT_RESIDUAL_QUANT_DIR" "${LOW_GPU_MEM_USAGE_ARGS[@]}"
		unset AUTO_ROUND_LOAD_IMATRIX_FILE
		log_time "R${iteration}: finish residual auto_round"
		print_elapsed "$STEP_START_TS"
	fi

	STEP_START_TS="$(date +%s)"
	log_time "R${iteration}: start residual merge"
	python get_residual.py -i "$PREV_MERGE_DIR" -q "$RESIDUAL_OUTPUT_DIR" -o "$OUTPUT_RESIDUAL_MERGE_DIR" --operation add --device cpu
	log_time "R${iteration}: finish residual merge"
	print_elapsed "$STEP_START_TS"

	echo "R${iteration} residual quantization results are saved in ${OUTPUT_RESIDUAL_QUANT_DIR}"
	echo "R${iteration} merged quantization results are saved in ${OUTPUT_RESIDUAL_MERGE_DIR}"
	log_time "finish iteration R${iteration}"
	print_elapsed "$ITER_START_TS"

	PREV_MERGE_DIR="$OUTPUT_RESIDUAL_MERGE_DIR"
	FINAL_OUTPUT_DIR="$OUTPUT_RESIDUAL_MERGE_DIR"
done

echo "Base quantization results are saved in ${OUTPUT_DIR}"
echo "Final merged quantization results are saved in ${FINAL_OUTPUT_DIR}"
log_time "script finished"
print_elapsed "$SCRIPT_START_TS"

