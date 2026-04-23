import argparse
import json
import math
import os
import re
from collections.abc import Iterable
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors_file


def get_option_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="outlier_profile")
	parser.add_argument(
		"-i",
		"--input",
		required=True,
		help="Hugging Face model repo id or local path",
	)
	parser.add_argument(
		"--model-name",
		default=None,
		help="Optional display name used in summaries and figure titles. Defaults to the current path-based name",
	)
	parser.add_argument(
		"-g",
		"--group-size",
		type=int,
		default=64,
		help="Group size used on the last tensor axis",
	)
	parser.add_argument(
		"-o",
		"--output",
		default=None,
		help="Output JSON path. Defaults to <model_dir>/outlier_profile_g{group_size}.json",
	)
	parser.add_argument(
		"--per-tensor-output",
		default=None,
		help="Optional JSONL output path with one record per tensor",
	)
	parser.add_argument(
		"--figure-output",
		default=None,
		help="Output image path for the paper-style K/r figure. Defaults to <output_stem>_paper_kr.png",
	)
	parser.add_argument(
		"--topk",
		type=int,
		default=20,
		help="Number of highest-ratio tensors and groups kept in the summary",
	)
	parser.add_argument(
        "-r",
		"--ratio-threshold",
		type=float,
		default=None,
		help="Optional threshold used to count tensors whose max_ratio is above, below, or equal to it",
	)
	parser.add_argument(
		"--kr-percentile",
		type=float,
		default=0.99,
		help="Percentile used to estimate the inlier radius r for the paper-style K/r metric",
	)
	parser.add_argument(
		"--device",
		default="cpu",
		help="Device used for statistics, e.g. cpu, cuda, cuda:0",
	)
	parser.add_argument(
		"--include",
		nargs="*",
		default=None,
		help="Optional substrings used to include tensor names",
	)
	parser.add_argument(
		"--exclude",
		nargs="*",
		default=None,
		help="Optional substrings used to exclude tensor names",
	)
	return parser


def normalize_local_path(path: str) -> str:
	return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def resolve_model_source(source: str) -> str:
	local_path = normalize_local_path(source)
	if os.path.exists(local_path):
		return local_path
	raise FileNotFoundError(f"Only local model paths are supported, got: {source}")


def resolve_model_display_name(model_dir: str, model_name: str | None) -> str:
	if model_name:
		return model_name
	return os.path.basename(os.path.normpath(model_dir)) or model_dir


def validate_args(args: argparse.Namespace) -> None:
	if args.group_size <= 1:
		raise ValueError("group-size must be greater than 1 so the second-largest value exists")
	if args.topk <= 0:
		raise ValueError("topk must be positive")
	if not 0 < args.kr_percentile <= 1:
		raise ValueError("kr-percentile must be in the interval (0, 1]")
	if args.device.startswith("cuda") and not torch.cuda.is_available():
		raise RuntimeError("CUDA device requested, but CUDA is not available")
	torch.empty(0, device=args.device)


def should_include_tensor(name: str, include_filters: list[str] | None, exclude_filters: list[str] | None) -> bool:
	if include_filters and not any(token in name for token in include_filters):
		return False
	if exclude_filters and any(token in name for token in exclude_filters):
		return False
	return True


def load_json(path: str) -> dict[str, Any]:
	with open(path, "r", encoding="utf-8") as file_obj:
		return json.load(file_obj)


def get_weight_files(model_dir: str) -> list[str]:
	safetensors_index = os.path.join(model_dir, "model.safetensors.index.json")
	pytorch_index = os.path.join(model_dir, "pytorch_model.bin.index.json")
	safetensors_single = os.path.join(model_dir, "model.safetensors")
	pytorch_single = os.path.join(model_dir, "pytorch_model.bin")

	if os.path.exists(safetensors_index):
		weight_map = load_json(safetensors_index)["weight_map"]
		return sorted({os.path.join(model_dir, filename) for filename in weight_map.values()})
	if os.path.exists(pytorch_index):
		weight_map = load_json(pytorch_index)["weight_map"]
		return sorted({os.path.join(model_dir, filename) for filename in weight_map.values()})
	if os.path.exists(safetensors_single):
		return [safetensors_single]
	if os.path.exists(pytorch_single):
		return [pytorch_single]
	raise FileNotFoundError(f"No supported weight files found under: {model_dir}")


def load_state_dict_file(path: str) -> dict[str, torch.Tensor]:
	if path.endswith(".safetensors"):
		return load_safetensors_file(path, device="cpu")

	payload = torch.load(path, map_location="cpu")
	if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
		return payload["state_dict"]
	if isinstance(payload, dict):
		return payload
	raise TypeError(f"Unsupported checkpoint payload type in {path}: {type(payload)!r}")


def iter_named_tensors(model_dir: str) -> Iterable[tuple[str, torch.Tensor, str]]:
	for weight_file in get_weight_files(model_dir):
		state_dict = load_state_dict_file(weight_file)
		for name, tensor in state_dict.items():
			if isinstance(tensor, torch.Tensor):
				yield name, tensor, os.path.basename(weight_file)


def compute_linear_quantile(values: torch.Tensor, quantile: float) -> float:
	flat_values = values.reshape(-1)
	count = flat_values.numel()
	if count == 0:
		raise ValueError("quantile input tensor is empty")
	if count == 1:
		return float(flat_values[0].item())

	position = (count - 1) * quantile
	lower_index = int(math.floor(position))
	upper_index = int(math.ceil(position))
	lower_value = flat_values.kthvalue(lower_index + 1).values
	if lower_index == upper_index:
		return float(lower_value.item())
	upper_value = flat_values.kthvalue(upper_index + 1).values
	weight = position - lower_index
	return float(torch.lerp(lower_value, upper_value, weight).item())


def compute_group_ratio_stats(
	tensor: torch.Tensor,
	group_size: int,
	device: str,
	topk: int,
	kr_percentile: float,
) -> dict[str, Any]:
	if tensor.ndim == 0:
		raise ValueError("scalar tensor is not supported")
	if tensor.shape[-1] < group_size:
		raise ValueError("last dimension is smaller than group size")
	if tensor.shape[-1] % group_size != 0:
		raise ValueError("last dimension is not divisible by group size")

	work_tensor = tensor.detach()
	if work_tensor.is_complex():
		work_tensor = work_tensor.abs()
	elif not torch.is_floating_point(work_tensor):
		work_tensor = work_tensor.to(torch.float32)

	work_tensor = work_tensor.to(device=device, dtype=torch.float32)
	reshaped = work_tensor.reshape(-1, work_tensor.shape[-1] // group_size, group_size)
	abs_values = reshaped.abs()
	top2_values, top2_indices = torch.topk(abs_values, k=2, dim=-1)

	largest = top2_values[..., 0]
	second_largest = top2_values[..., 1]
	ratios = torch.full_like(largest, fill_value=torch.inf)
	nonzero_mask = second_largest > 0
	ratios[nonzero_mask] = largest[nonzero_mask] / second_largest[nonzero_mask]
	same_zero_mask = (~nonzero_mask) & (largest == 0)
	ratios[same_zero_mask] = 1.0

	flat_ratios = ratios.reshape(-1)
	finite_mask = torch.isfinite(flat_ratios)
	finite_ratios = flat_ratios[finite_mask]
	flat_abs_values = abs_values.reshape(-1)
	max_abs = float(flat_abs_values.max().item())
	inlier_radius = compute_linear_quantile(flat_abs_values, kr_percentile)
	if inlier_radius > 0:
		paper_kr_ratio = max_abs / inlier_radius
	elif max_abs == 0:
		paper_kr_ratio = 1.0
	else:
		paper_kr_ratio = math.inf

	max_ratio_value, max_ratio_position = flat_ratios.max(dim=0)
	group_index = int(max_ratio_position.item())
	groups_per_row = reshaped.shape[1]
	row_index = group_index // groups_per_row
	inner_group_index = group_index % groups_per_row
	group_slice = reshaped[row_index, inner_group_index]
	top_values = top2_values.reshape(-1, 2)[group_index]
	top_indices = top2_indices.reshape(-1, 2)[group_index]

	top_groups_count = min(topk, flat_ratios.numel())
	top_group_ratios, top_group_positions = torch.topk(flat_ratios, k=top_groups_count)

	finite_count = int(finite_mask.sum().item())
	inf_count = int((~finite_mask).sum().item())
	if finite_count > 0:
		finite_mean = float(finite_ratios.mean().item())
		finite_median = float(finite_ratios.median().item())
		finite_std = float(finite_ratios.std(unbiased=False).item())
		finite_p95 = compute_linear_quantile(finite_ratios, 0.95)
		finite_p99 = compute_linear_quantile(finite_ratios, 0.99)
	else:
		finite_mean = math.nan
		finite_median = math.nan
		finite_std = math.nan
		finite_p95 = math.nan
		finite_p99 = math.nan

	return {
		"shape": list(tensor.shape),
		"dtype": str(tensor.dtype),
		"num_groups": int(flat_ratios.numel()),
		"max_ratio": float(max_ratio_value.item()),
		"paper_kr_ratio": float(paper_kr_ratio),
		"paper_kr_max_abs": max_abs,
		"paper_kr_inlier_radius": inlier_radius,
		"paper_kr_percentile": kr_percentile,
		"mean_ratio": finite_mean,
		"median_ratio": finite_median,
		"std_ratio": finite_std,
		"p95_ratio": finite_p95,
		"p99_ratio": finite_p99,
		"finite_group_count": finite_count,
		"infinite_group_count": inf_count,
		"max_ratio_group": {
			"flat_index": group_index,
			"row_index": row_index,
			"group_index": inner_group_index,
			"largest_abs": float(top_values[0].item()),
			"second_abs": float(top_values[1].item()),
			"largest_abs_index": int(top_indices[0].item()),
			"second_abs_index": int(top_indices[1].item()),
			"values": [float(value.item()) for value in group_slice.detach().cpu()],
		},
		"top_groups": [
			{
				"flat_index": int(position.item()),
				"row_index": int(position.item() // groups_per_row),
				"group_index": int(position.item() % groups_per_row),
				"ratio": float(ratio.item()),
			}
			for ratio, position in zip(top_group_ratios.detach().cpu(), top_group_positions.detach().cpu(), strict=False)
		],
	}


def build_summary(
	tensor_records: list[dict[str, Any]],
	model_dir: str,
	model_name: str,
	group_size: int,
	device: str,
	topk: int,
	kr_percentile: float,
	ratio_threshold: float | None,
	skipped: list[dict[str, Any]],
) -> dict[str, Any]:
	sorted_records = sorted(tensor_records, key=lambda item: item["max_ratio"], reverse=True)
	ratios = [record["max_ratio"] for record in sorted_records if math.isfinite(record["max_ratio"])]
	paper_kr_ratios = [record["paper_kr_ratio"] for record in sorted_records if math.isfinite(record["paper_kr_ratio"])]
	threshold_summary = None
	if ratio_threshold is not None:
		greater_count = sum(1 for record in sorted_records if record["paper_kr_ratio"] > ratio_threshold)
		less_count = sum(1 for record in sorted_records if record["paper_kr_ratio"] < ratio_threshold)
		equal_count = sum(1 for record in sorted_records if record["paper_kr_ratio"] == ratio_threshold)
		threshold_summary = {
			"threshold": ratio_threshold,
			"greater_than_count": greater_count,
			"less_than_count": less_count,
			"equal_count": equal_count,
		}
	return {
		"model_dir": model_dir,
		"model_name": model_name,
		"group_size": group_size,
		"device": device,
		"paper_kr_percentile": kr_percentile,
		"tensor_count": len(tensor_records),
		"skipped_count": len(skipped),
		"global_max_ratio": sorted_records[0]["max_ratio"] if sorted_records else math.nan,
		"global_mean_max_ratio": sum(ratios) / len(ratios) if ratios else math.nan,
		"global_max_paper_kr_ratio": max(paper_kr_ratios) if paper_kr_ratios else math.nan,
		"global_mean_paper_kr_ratio": sum(paper_kr_ratios) / len(paper_kr_ratios) if paper_kr_ratios else math.nan,
		"ratio_threshold_summary": threshold_summary,
		"top_tensors": sorted_records[:topk],
		"skipped": skipped,
	}


def save_json(path: str, payload: dict[str, Any]) -> None:
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	with open(path, "w", encoding="utf-8") as file_obj:
		json.dump(round_nested_floats(payload), file_obj, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
	os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
	with open(path, "w", encoding="utf-8") as file_obj:
		for row in rows:
			file_obj.write(json.dumps(round_nested_floats(row), ensure_ascii=False))
			file_obj.write("\n")


def format_metric(value: float) -> str:
	if math.isnan(value):
		return "nan"
	if math.isinf(value):
		return "inf"
	return f"{value:.3f}"


def round_nested_floats(value: Any, digits: int = 3) -> Any:
	if isinstance(value, float):
		if math.isfinite(value):
			return round(value, digits)
		return value
	if isinstance(value, list):
		return [round_nested_floats(item, digits) for item in value]
	if isinstance(value, tuple):
		return [round_nested_floats(item, digits) for item in value]
	if isinstance(value, dict):
		return {key: round_nested_floats(item, digits) for key, item in value.items()}
	return value


def add_threshold_line(axis, threshold: float | None) -> None:
	import matplotlib.transforms as transforms

	if threshold is None or not math.isfinite(threshold):
		return
	axis.axhline(
		threshold,
		color="#dc2626",
		linestyle="--",
		linewidth=1.4,
		alpha=0.9,
	)
	blended_transform = transforms.blended_transform_factory(axis.transAxes, axis.transData)
	axis.text(
		1.01,
		threshold,
		f"K/r = {format_metric(threshold)}",
		transform=blended_transform,
		color="#dc2626",
		fontsize=9,
		va="bottom",
		ha="left",
		bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#fca5a5", "alpha": 0.92},
		clip_on=False,
	)


def add_vertical_threshold_line(axis, threshold: float | None) -> None:
	import matplotlib.transforms as transforms

	if threshold is None or not math.isfinite(threshold):
		return
	axis.axvline(
		threshold,
		color="#dc2626",
		linestyle="--",
		linewidth=1.4,
		alpha=0.9,
	)
	blended_transform = transforms.blended_transform_factory(axis.transData, axis.transAxes)
	axis.text(
		threshold,
		1.01,
		f"K/r = {format_metric(threshold)}",
		transform=blended_transform,
		color="#dc2626",
		fontsize=9,
		va="bottom",
		ha="left",
		rotation=90,
		bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#fca5a5", "alpha": 0.92},
		clip_on=False,
	)


def shorten_tensor_name(name: str, max_length: int = 42) -> str:
	if len(name) <= max_length:
		return name
	keep = max_length - 3
	head = keep // 2
	tail = keep - head
	return f"{name[:head]}...{name[-tail:]}"


MODULE_FAMILY_ORDER = {
	"embed": 0,
	"attn": 1,
	"mlp": 2,
	"norm": 3,
	"head": 4,
	"other": 5,
}


MODULE_FAMILY_COLORS = {
	"embed": "#14b8a6",
	"attn": "#f97316",
	"mlp": "#2563eb",
	"norm": "#8b5cf6",
	"head": "#ef4444",
	"other": "#64748b",
}


def get_module_family(tensor_name: str) -> str:
	if tensor_name.startswith(("model.embed_tokens", "embed_tokens")):
		return "embed"
	if tensor_name.startswith("lm_head"):
		return "head"
	if ".self_attn." in tensor_name or ".attn." in tensor_name:
		return "attn"
	if ".mlp." in tensor_name or ".feed_forward." in tensor_name or ".experts." in tensor_name:
		return "mlp"
	if tensor_name.startswith(("model.norm", "norm")) or ".norm." in tensor_name:
		return "norm"
	return "other"


def extract_layer_name(tensor_name: str) -> str:
	match = re.search(r"(?:^|\.)(layers\.(\d+))(?:\.|$)", tensor_name)
	if match:
		return f"L{int(match.group(2)):02d}"
	for prefix in ("model.embed_tokens", "embed_tokens", "lm_head", "model.norm", "norm"):
		if tensor_name.startswith(prefix):
			return prefix
	parts = tensor_name.split(".")
	if len(parts) >= 2:
		return ".".join(parts[:2])
	return tensor_name


def get_layer_sort_key(layer_name: str, module_family: str) -> tuple[int, int, str]:
	if layer_name.startswith("L") and layer_name[1:].isdigit():
		return (1, int(layer_name[1:]), module_family)
	special_order = {
		"model.embed_tokens": (0, 0),
		"embed_tokens": (0, 0),
		"model.norm": (2, 0),
		"norm": (2, 1),
		"lm_head": (3, 0),
	}
	group_order, index = special_order.get(layer_name, (4, 0))
	return (group_order, index, layer_name)


def build_layer_aggregates(tensor_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
	layer_buckets: dict[tuple[str, str], list[float]] = {}
	for record in tensor_records:
		paper_kr_ratio = record["paper_kr_ratio"]
		if not math.isfinite(paper_kr_ratio):
			continue
		layer_name = extract_layer_name(record["name"])
		module_family = get_module_family(record["name"])
		layer_buckets.setdefault((layer_name, module_family), []).append(paper_kr_ratio)

	layer_records = []
	for (layer_name, module_family), values in layer_buckets.items():
		layer_records.append(
			{
				"layer_name": layer_name,
				"module_family": module_family,
				"tensor_count": len(values),
				"mean_paper_kr_ratio": sum(values) / len(values),
				"max_paper_kr_ratio": max(values),
				"sort_key": get_layer_sort_key(layer_name, module_family),
			}
		)

	return sorted(
		layer_records,
		key=lambda item: (item["sort_key"][0], item["sort_key"][1], MODULE_FAMILY_ORDER.get(item["module_family"], 99), item["layer_name"]),
	)


def plot_paper_kr_figure(
	model_dir: str,
	model_name: str,
	tensor_records: list[dict[str, Any]],
	output_path: str,
	kr_percentile: float,
	topk: int,
	ratio_threshold: float | None,
) -> None:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	finite_records = [record for record in tensor_records if math.isfinite(record["paper_kr_ratio"])]
	if not finite_records:
		raise ValueError("No finite paper_kr_ratio values available for plotting")

	sorted_records = sorted(finite_records, key=lambda item: item["paper_kr_ratio"], reverse=True)
	rank_values = [record["paper_kr_ratio"] for record in sorted_records]
	ranks = list(range(1, len(rank_values) + 1))
	top_count = min(topk, len(sorted_records), 15)
	top_records = list(reversed(sorted_records[:top_count]))

	figure, (curve_axis, bar_axis) = plt.subplots(
		1,
		2,
		figsize=(13.5, 5.8),
		gridspec_kw={"width_ratios": [1.8, 1.2]},
		constrained_layout=True,
	)
	figure.patch.set_facecolor("white")

	curve_axis.plot(ranks, rank_values, color="#0f766e", linewidth=2.2)
	curve_axis.fill_between(ranks, rank_values, color="#99f6e4", alpha=0.35)
	curve_axis.scatter(ranks[:top_count], rank_values[:top_count], color="#134e4a", s=18, zorder=3)
	curve_axis.set_title("Sorted paper-style K/r across tensors", fontsize=13, pad=10)
	curve_axis.set_xlabel("Tensor rank", fontsize=11)
	curve_axis.set_ylabel("K/r ratio", fontsize=11)
	curve_axis.grid(alpha=0.25, linestyle="--", linewidth=0.8)
	curve_axis.spines[["top", "right"]].set_visible(False)
	add_threshold_line(curve_axis, ratio_threshold)
	if rank_values[-1] > 0 and rank_values[0] / rank_values[-1] >= 20:
		curve_axis.set_yscale("log")
	stats_text = (
		f"model tensors = {len(sorted_records)}\n"
		f"max K/r = {format_metric(rank_values[0])}\n"
		f"mean K/r = {format_metric(sum(rank_values) / len(rank_values))}\n"
		f"r percentile = p{kr_percentile:.2f}"
	)
	curve_axis.text(
		0.02,
		0.98,
		stats_text,
		transform=curve_axis.transAxes,
		va="top",
		ha="left",
		fontsize=10,
		bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f0fdfa", "edgecolor": "#99f6e4"},
	)

	bar_labels = [shorten_tensor_name(record["name"]) for record in top_records]
	bar_values = [record["paper_kr_ratio"] for record in top_records]
	bar_colors = ["#f97316" if index == len(top_records) - 1 else "#fb923c" for index in range(len(top_records))]
	bar_axis.barh(bar_labels, bar_values, color=bar_colors)
	bar_axis.set_title(f"Top {top_count} tensors by K/r", fontsize=13, pad=10)
	bar_axis.set_xlabel("K/r ratio", fontsize=11)
	add_vertical_threshold_line(bar_axis, ratio_threshold)
	bar_axis.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.8)
	bar_axis.spines[["top", "right"]].set_visible(False)
	for label_index, value in enumerate(bar_values):
		bar_axis.text(value, label_index, f" {format_metric(value)}", va="center", ha="left", fontsize=9)

	figure.suptitle(
		f"Tensor outlier profile for {model_name}",
		fontsize=15,
		fontweight="bold",
	)
	os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
	figure.savefig(output_path, dpi=240, bbox_inches="tight")
	plt.close(figure)


def plot_layer_paper_kr_figure(
	model_dir: str,
	model_name: str,
	layer_records: list[dict[str, Any]],
	output_path: str,
	kr_percentile: float,
	ratio_threshold: float | None,
) -> None:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	if not layer_records:
		raise ValueError("No finite paper_kr_ratio values available for layer plotting")

	ordered_layers = []
	for record in layer_records:
		if record["layer_name"] not in ordered_layers:
			ordered_layers.append(record["layer_name"])
	present_families = [
		family
		for family in MODULE_FAMILY_ORDER
		if any(record["module_family"] == family for record in layer_records)
	]
	record_lookup = {(record["layer_name"], record["module_family"]): record for record in layer_records}
	x_positions = list(range(len(ordered_layers)))
	bar_width = 0.78 / max(len(present_families), 1)

	figure, axis = plt.subplots(figsize=(max(10, len(layer_records) * 0.42), 5.8), constrained_layout=True)
	figure.patch.set_facecolor("white")
	for family_index, family in enumerate(present_families):
		offsets = [position + (family_index - (len(present_families) - 1) / 2) * bar_width for position in x_positions]
		mean_values = []
		max_values = []
		for layer_name in ordered_layers:
			record = record_lookup.get((layer_name, family))
			mean_values.append(record["mean_paper_kr_ratio"] if record else math.nan)
			max_values.append(record["max_paper_kr_ratio"] if record else math.nan)
		axis.bar(
			offsets,
			mean_values,
			width=bar_width * 0.92,
			color=MODULE_FAMILY_COLORS[family],
			alpha=0.82,
			label=f"{family} mean",
		)
		finite_offsets = [offset for offset, value in zip(offsets, max_values, strict=False) if not math.isnan(value)]
		finite_max_values = [value for value in max_values if not math.isnan(value)]
		axis.scatter(
			finite_offsets,
			finite_max_values,
			color=MODULE_FAMILY_COLORS[family],
			edgecolors="white",
			linewidths=0.7,
			marker="D",
			s=28,
			zorder=3,
			label=f"{family} max",
		)
	axis.set_title("Layer-ordered paper-style K/r by module family", fontsize=14, pad=10)
	axis.set_xlabel("Layer / module", fontsize=11)
	axis.set_ylabel("K/r ratio", fontsize=11)
	axis.set_xticks(x_positions)
	axis.set_xticklabels(ordered_layers, rotation=60, ha="right", fontsize=9)
	axis.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
	axis.spines[["top", "right"]].set_visible(False)
	add_threshold_line(axis, ratio_threshold)
	axis.legend(frameon=False, loc="upper right", ncols=2, fontsize=9)
	all_positive_means = [record["mean_paper_kr_ratio"] for record in layer_records if record["mean_paper_kr_ratio"] > 0]
	all_max_values = [record["max_paper_kr_ratio"] for record in layer_records]
	if all_positive_means and max(all_max_values) / min(all_positive_means) >= 20:
		axis.set_yscale("log")

	stats_text = (
		f"aggregated layers = {len(layer_records)}\n"
		f"max layer K/r = {format_metric(max(all_max_values))}\n"
		f"mean layer K/r = {format_metric(sum(record['mean_paper_kr_ratio'] for record in layer_records) / len(layer_records))}\n"
		f"r percentile = p{kr_percentile:.2f}"
	)
	axis.text(
		0.015,
		0.98,
		stats_text,
		transform=axis.transAxes,
		va="top",
		ha="left",
		fontsize=10,
		bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eff6ff", "edgecolor": "#93c5fd"},
	)

	figure.suptitle(
		f"Layer outlier profile for {model_name}",
		fontsize=15,
		fontweight="bold",
	)
	os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
	figure.savefig(output_path, dpi=240, bbox_inches="tight")
	plt.close(figure)


def profile_model(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
	tensor_records: list[dict[str, Any]] = []
	skipped: list[dict[str, Any]] = []

	for name, tensor, source_file in iter_named_tensors(args.input):
		if not should_include_tensor(name, args.include, args.exclude):
			continue

		try:
			stats = compute_group_ratio_stats(tensor, args.group_size, args.device, args.topk, args.kr_percentile)
		except ValueError as exc:
			skipped.append(
				{
					"name": name,
					"shape": list(tensor.shape),
					"dtype": str(tensor.dtype),
					"source_file": source_file,
					"reason": str(exc),
				}
			)
			print(f"[SKIP] {name} ({source_file}): {exc}")
			continue

		stats["name"] = name
		stats["source_file"] = source_file
		tensor_records.append(stats)
		print(
			f"[TENSOR] {name} ({source_file}) "
			f"max_ratio={format_metric(stats['max_ratio'])} "
			f"paper_kr_ratio={format_metric(stats['paper_kr_ratio'])}"
		)

	return (
		build_summary(
			tensor_records,
			args.input,
			resolve_model_display_name(args.input, args.model_name),
			args.group_size,
			args.device,
			args.topk,
			args.kr_percentile,
			args.ratio_threshold,
			skipped,
		),
		tensor_records,
	)


def run(args: argparse.Namespace) -> None:
	args.input = resolve_model_source(args.input)
	validate_args(args)

	if args.output is None:
		args.output = os.path.join(args.input, f"outlier_profile_g{args.group_size}.json")
	else:
		args.output = normalize_local_path(args.output)

	if args.per_tensor_output is not None:
		args.per_tensor_output = normalize_local_path(args.per_tensor_output)
	if args.figure_output is None:
		output_stem, _ = os.path.splitext(args.output)
		args.figure_output = f"{output_stem}_paper_kr.png"
	else:
		args.figure_output = normalize_local_path(args.figure_output)
	layer_figure_output = f"{os.path.splitext(args.figure_output)[0]}_by_layer.png"
	model_name = resolve_model_display_name(args.input, args.model_name)

	summary, tensor_records = profile_model(args)
	layer_records = build_layer_aggregates(tensor_records)
	save_json(args.output, summary)
	if args.per_tensor_output:
		save_jsonl(args.per_tensor_output, tensor_records)
	plot_paper_kr_figure(model_dir=args.input, model_name=model_name, tensor_records=tensor_records, output_path=args.figure_output, kr_percentile=args.kr_percentile, topk=args.topk, ratio_threshold=args.ratio_threshold)
	plot_layer_paper_kr_figure(model_dir=args.input, model_name=model_name, layer_records=layer_records, output_path=layer_figure_output, kr_percentile=args.kr_percentile, ratio_threshold=args.ratio_threshold)
	summary["model_name"] = model_name
	summary["figure_output"] = args.figure_output
	summary["layer_figure_output"] = layer_figure_output
	summary["layer_topk"] = sorted(layer_records, key=lambda item: item["max_paper_kr_ratio"], reverse=True)[: args.topk]
	save_json(args.output, summary)

	print(f"Profile saved to: {args.output}")
	print(f"Figure saved to: {args.figure_output}")
	print(f"Layer figure saved to: {layer_figure_output}")
	print(f"Profiled tensors: {summary['tensor_count']}")
	print(f"Skipped tensors: {summary['skipped_count']}")
	print(f"Global max ratio: {summary['global_max_ratio']}")
	print(
		"Global paper-style K/r ratio: "
		f"{summary['global_max_paper_kr_ratio']} "
		f"(r estimated by p={summary['paper_kr_percentile']})"
	)
	if summary["ratio_threshold_summary"] is not None:
		threshold_summary = summary["ratio_threshold_summary"]
		print(
			"Threshold summary: "
			f"> {threshold_summary['threshold']}: {threshold_summary['greater_than_count']}, "
			f"< {threshold_summary['threshold']}: {threshold_summary['less_than_count']}, "
			f"= {threshold_summary['threshold']}: {threshold_summary['equal_count']}"
		)


if __name__ == "__main__":
	parser = get_option_parser()
	run(parser.parse_args())
