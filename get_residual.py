import argparse
import json
import os
import shutil
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from auto_round.utils import normalize_tied_weight_keys_for_save


TOKENIZER_ARTIFACT_NAMES = {
	"added_tokens.json",
	"chat_template.jinja",
	"merges.txt",
	"sentencepiece.bpe.model",
	"special_tokens_map.json",
	"spiece.model",
	"tokenizer.json",
	"tokenizer.jsonl",
	"tokenizer.model",
	"tokenizer_config.json",
	"vocab.json",
	"vocab.txt",
	"tekken.json",
}


def get_option_parser():
	parser = argparse.ArgumentParser(prog="get_residual")
	parser.add_argument(
		"-i",
		"--input",
		required=True,
		help="original model repo id or local path",
	)
	parser.add_argument(
		"-q",
		"--quantized",
		required=True,
		help="quantized model repo id or local path",
	)
	parser.add_argument(
		"-o",
		"--output",
		required=True,
		help="output directory for the residual model",
	)
	parser.add_argument(
		"--device",
		default="cpu",
		help="device for residual computation, e.g. cpu, cuda, cuda:0",
	)
	parser.add_argument(
		"--operation",
		choices=["sub", "add"],
		default="sub",
		help="tensor combine operation: sub=original-quantized, add=original+quantized",
	)
	return parser


def normalize_local_path(path: str) -> str:
	return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def resolve_model_source(source: str) -> str:
	local_path = normalize_local_path(source)
	if os.path.exists(local_path):
		return local_path
	return source


def load_model(model_source: str):
	return AutoModelForCausalLM.from_pretrained(
		model_source,
		trust_remote_code=True,
		device_map="cpu",
		low_cpu_mem_usage=True,
		torch_dtype="auto",
	)


def should_copy_tokenizer_artifact(file_name: str) -> bool:
	base_name = os.path.basename(file_name)
	return "/" not in file_name and (
		base_name in TOKENIZER_ARTIFACT_NAMES
		or base_name.startswith("tokenizer.")
		or base_name.startswith("tokenizer_")
	)


def list_local_tokenizer_artifacts(model_source: str) -> list[str]:
	return sorted(
		file_name
		for file_name in os.listdir(model_source)
		if os.path.isfile(os.path.join(model_source, file_name)) and should_copy_tokenizer_artifact(file_name)
	)


def list_remote_tokenizer_artifacts(model_source: str) -> list[str]:
	from huggingface_hub import list_repo_files

	return sorted(file_name for file_name in list_repo_files(model_source) if should_copy_tokenizer_artifact(file_name))


def copy_local_tokenizer_artifacts(model_source: str, output_dir: str, file_names: list[str]) -> None:
	for file_name in file_names:
		shutil.copy2(os.path.join(model_source, file_name), os.path.join(output_dir, file_name))


def copy_remote_tokenizer_artifacts(model_source: str, output_dir: str, file_names: list[str]) -> None:
	from huggingface_hub import hf_hub_download

	for file_name in file_names:
		source_path = hf_hub_download(repo_id=model_source, filename=file_name)
		shutil.copy2(source_path, os.path.join(output_dir, file_name))


def try_copy_model_config(model_source: str, output_dir: str) -> None:
	try:
		os.makedirs(output_dir, exist_ok=True)
		if os.path.isdir(model_source):
			source_path = os.path.join(model_source, "config.json")
			if not os.path.exists(source_path):
				print(f"Skip config copy: no config.json found in {model_source}")
				return
		else:
			from huggingface_hub import hf_hub_download

			source_path = hf_hub_download(repo_id=model_source, filename="config.json")
		shutil.copy2(source_path, os.path.join(output_dir, "config.json"))
	except Exception as exc:  # pylint: disable=broad-except
		print(f"Skip config copy: {exc}")


def try_save_tokenizer(model_source: str, output_dir: str):
	try:
		os.makedirs(output_dir, exist_ok=True)
		if os.path.isdir(model_source):
			file_names = list_local_tokenizer_artifacts(model_source)
			copy_local_tokenizer_artifacts(model_source, output_dir, file_names)
		else:
			file_names = list_remote_tokenizer_artifacts(model_source)
			copy_remote_tokenizer_artifacts(model_source, output_dir, file_names)
		if not file_names:
			print(f"Skip tokenizer copy: no tokenizer artifacts found in {model_source}")
	except Exception as exc:  # pylint: disable=broad-except
		print(f"Skip tokenizer copy: {exc}")


def compute_residual_state(
	original_state: dict[str, torch.Tensor],
	quantized_state: dict[str, torch.Tensor],
	compute_device: str,
	operation: str,
):
	missing_in_quantized = []
	shape_mismatches = []
	skipped_non_float = []
	updated = []

	for key in tqdm(original_state.keys(), desc="Computing residual", unit="tensor"):
		if key not in quantized_state:
			missing_in_quantized.append(key)
			continue

		original_tensor = original_state[key]
		quantized_tensor = quantized_state[key]

		if original_tensor.shape != quantized_tensor.shape:
			shape_mismatches.append(
				{
					"name": key,
					"original_shape": list(original_tensor.shape),
					"quantized_shape": list(quantized_tensor.shape),
				}
			)
			continue

		if torch.is_floating_point(original_tensor) or torch.is_complex(original_tensor):
			original_fp32 = original_tensor.to(compute_device, dtype=torch.float32)
			quantized_fp32 = quantized_tensor.to(compute_device, dtype=torch.float32)
			if operation == "add":
				residual_tensor = original_fp32 + quantized_fp32
			else:
				residual_tensor = original_fp32 - quantized_fp32
			original_state[key] = residual_tensor.to(original_tensor.dtype).contiguous()
			updated.append(key)
		else:
			skipped_non_float.append(key)

	extra_in_quantized = sorted(set(quantized_state.keys()) - set(original_state.keys()))

	report = {
		"updated_count": len(updated),
		"missing_in_quantized_count": len(missing_in_quantized),
		"extra_in_quantized_count": len(extra_in_quantized),
		"shape_mismatch_count": len(shape_mismatches),
		"skipped_non_float_count": len(skipped_non_float),
		"missing_in_quantized": missing_in_quantized,
		"extra_in_quantized": extra_in_quantized,
		"shape_mismatches": shape_mismatches,
		"skipped_non_float": skipped_non_float,
	}
	return original_state, report


def validate_compute_device(device: str) -> str:
	if device.startswith("cuda") and not torch.cuda.is_available():
		raise RuntimeError("CUDA device requested, but CUDA is not available")
	torch.empty(0, device=device)
	return device


def save_report(output_dir: str, report: dict[str, Any], original_source: str, quantized_source: str, operation: str):
	payload = {
		"original_model": original_source,
		"quantized_model": quantized_source,
		"operation": operation,
		**report,
	}
	report_path = os.path.join(output_dir, "residual_report.json")
	with open(report_path, "w", encoding="utf-8") as file_obj:
		json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def run(args):
	original_source = resolve_model_source(args.input)
	quantized_source = resolve_model_source(args.quantized)
	output_dir = normalize_local_path(args.output)
	compute_device = validate_compute_device(args.device)
	operation = args.operation
	os.makedirs(output_dir, exist_ok=True)

	print(f"Loading original model from: {original_source}")
	original_model = load_model(original_source)
	print(f"Loading quantized model from: {quantized_source}")
	quantized_model = load_model(quantized_source)

	original_state = original_model.state_dict()
	quantized_state = quantized_model.state_dict()
	residual_state, report = compute_residual_state(original_state, quantized_state, compute_device, operation)

	load_result = original_model.load_state_dict(residual_state, strict=False)
	if load_result.missing_keys or load_result.unexpected_keys:
		print(f"load_state_dict missing_keys={load_result.missing_keys}")
		print(f"load_state_dict unexpected_keys={load_result.unexpected_keys}")

	normalize_tied_weight_keys_for_save(original_model)
	original_model.save_pretrained(output_dir, safe_serialization=True)
	try_copy_model_config(original_source, output_dir)
	try_save_tokenizer(original_source, output_dir)
	save_report(output_dir, report, original_source, quantized_source, operation)

	print(f"Residual model saved to: {output_dir}")
	print(f"Computation device: {compute_device}")
	print(f"Operation: {operation}")
	print(
		"Updated tensors: "
		f"{report['updated_count']}, "
		f"missing: {report['missing_in_quantized_count']}, "
		f"extra: {report['extra_in_quantized_count']}, "
		f"shape mismatches: {report['shape_mismatch_count']}, "
		f"skipped non-float: {report['skipped_non_float_count']}"
	)


if __name__ == "__main__":
	parser = get_option_parser()
	run(parser.parse_args())
