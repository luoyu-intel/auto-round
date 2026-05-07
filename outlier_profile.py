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
        help="Optional threshold used for max_ratio summaries and as a reference line on the paper-style K/r plots",
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
        raise ValueError(
            "group-size must be greater than 1 so the second-largest value exists"
        )
    if args.topk <= 0:
        raise ValueError("topk must be positive")
    if not 0 < args.kr_percentile <= 1:
        raise ValueError("kr-percentile must be in the interval (0, 1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but CUDA is not available")
    torch.empty(0, device=args.device)


def should_include_tensor(
    name: str, include_filters: list[str] | None, exclude_filters: list[str] | None
) -> bool:
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
        return sorted(
            {os.path.join(model_dir, filename) for filename in weight_map.values()}
        )
    if os.path.exists(pytorch_index):
        weight_map = load_json(pytorch_index)["weight_map"]
        return sorted(
            {os.path.join(model_dir, filename) for filename in weight_map.values()}
        )
    if os.path.exists(safetensors_single):
        return [safetensors_single]
    if os.path.exists(pytorch_single):
        return [pytorch_single]
    raise FileNotFoundError(f"No supported weight files found under: {model_dir}")


def load_state_dict_file(path: str) -> dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        return load_safetensors_file(path, device="cpu")

    payload = torch.load(path, map_location="cpu")
    if (
        isinstance(payload, dict)
        and "state_dict" in payload
        and isinstance(payload["state_dict"], dict)
    ):
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


def compute_linear_quantile_along_last_dim(
    values: torch.Tensor, quantile: float
) -> torch.Tensor:
    count = values.shape[-1]
    if count == 0:
        raise ValueError("quantile input tensor is empty")
    if count == 1:
        return values[..., 0]

    sorted_values, _ = torch.sort(values, dim=-1)
    position = (count - 1) * quantile
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower_value = sorted_values[..., lower_index]
    if lower_index == upper_index:
        return lower_value
    upper_value = sorted_values[..., upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def compute_lower_quantile_along_last_dim(
    values: torch.Tensor, quantile: float
) -> torch.Tensor:
    count = values.shape[-1]
    if count == 0:
        raise ValueError("quantile input tensor is empty")
    if count == 1:
        return values[..., 0]

    sorted_values, _ = torch.sort(values, dim=-1)
    index = int(math.floor((count - 1) * quantile))
    return sorted_values[..., index]


def compute_signed_group_anchor_along_last_dim(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[-1] == 0:
        raise ValueError("anchor input tensor is empty")

    max_values = values.max(dim=-1).values
    min_values = values.min(dim=-1).values
    positive_mask = values > 0
    negative_mask = values < 0
    positive_count = positive_mask.sum(dim=-1)
    negative_count = negative_mask.sum(dim=-1)

    positive_values = torch.where(
        positive_mask, values, torch.full_like(values, -torch.inf)
    )
    top_positive_values = torch.topk(
        positive_values, k=min(2, values.shape[-1]), dim=-1
    ).values
    if values.shape[-1] >= 2:
        second_largest_positive = top_positive_values[..., 1]
    else:
        second_largest_positive = torch.full_like(max_values, -torch.inf)

    negative_abs_values = torch.where(
        negative_mask, -values, torch.full_like(values, -torch.inf)
    )
    top_negative_abs_values = torch.topk(
        negative_abs_values, k=min(2, values.shape[-1]), dim=-1
    ).values
    if values.shape[-1] >= 2:
        second_smallest_negative = -top_negative_abs_values[..., 1]
    else:
        second_smallest_negative = torch.full_like(min_values, torch.inf)

    # K is the signed endpoint with the largest magnitude in the group.
    k_is_positive = max_values.abs() >= min_values.abs()
    k_values = torch.where(k_is_positive, max_values, min_values)
    r_values = torch.zeros_like(k_values)

    positive_k_with_negative = k_is_positive & (negative_count > 0)
    positive_k_with_second_positive = (
        k_is_positive & (negative_count == 0) & (positive_count > 1)
    )
    negative_k_with_positive = (~k_is_positive) & (positive_count > 0)
    negative_k_with_second_negative = (
        (~k_is_positive) & (positive_count == 0) & (negative_count > 1)
    )

    r_values = torch.where(positive_k_with_negative, min_values, r_values)
    r_values = torch.where(
        positive_k_with_second_positive, second_largest_positive / 2, r_values
    )
    r_values = torch.where(negative_k_with_positive, max_values, r_values)
    r_values = torch.where(
        negative_k_with_second_negative, second_smallest_negative / 2, r_values
    )
    return k_values, r_values


def compute_group_ratio_stats(
    tensor: torch.Tensor,
    group_size: int,
    device: str,
    topk: int,
    kr_percentile: float,
    ratio_threshold: float | None,
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
    topk_value_count = min(3, group_size)
    topk_values, topk_indices = torch.topk(abs_values, k=topk_value_count, dim=-1)
    top2_values = topk_values[..., :2]
    top2_indices = topk_indices[..., :2]

    largest = top2_values[..., 0]
    second_largest = top2_values[..., 1]
    if topk_value_count >= 3:
        third_largest = topk_values[..., 2]
    else:
        third_largest = second_largest

    # UPPER STRATEGY (上策): Use group Mean Absolute Value as the bulk radius proxy r
    # instead of second_largest, to accurately reflect how far the apex escapes the "bulk".
    group_bulk_radius = abs_values.mean(dim=-1)

    ratios = torch.full_like(largest, fill_value=torch.inf)
    nonzero_mask = group_bulk_radius > 0
    ratios[nonzero_mask] = largest[nonzero_mask] / group_bulk_radius[nonzero_mask]
    same_zero_mask = (~nonzero_mask) & (largest == 0)
    ratios[same_zero_mask] = 1.0

    flat_ratios = ratios.reshape(-1)
    finite_mask = torch.isfinite(flat_ratios)
    finite_ratios = flat_ratios[finite_mask]
    ratio_finite_count = int(finite_mask.sum().item())
    ratio_infinite_count = int((~finite_mask).sum().item())
    ratio_finite_sum = (
        float(finite_ratios.sum().item()) if ratio_finite_count > 0 else 0.0
    )
    # Mean statistics describe the average severity across all groups in a tensor.
    mean_group_max_ratio = float(flat_ratios.mean().item())
    # Any-group statistics reproduce the older "one bad group is enough" behavior.
    any_group_max_ratio = float(flat_ratios.max().item())
    flat_abs_values = abs_values.reshape(-1)
    whole_tensor_max_abs = float(flat_abs_values.max().item())
    whole_tensor_inlier_radius = compute_linear_quantile(flat_abs_values, kr_percentile)
    if whole_tensor_inlier_radius > 0:
        whole_tensor_paper_kr_ratio = whole_tensor_max_abs / whole_tensor_inlier_radius
    elif whole_tensor_max_abs == 0:
        whole_tensor_paper_kr_ratio = 1.0
    else:
        whole_tensor_paper_kr_ratio = math.inf

    group_k_values, group_inlier_radius = compute_signed_group_anchor_along_last_dim(
        reshaped
    )
    group_max_abs = group_k_values.abs()
    group_inlier_radius_abs = group_inlier_radius.abs()
    group_paper_kr_ratios = torch.full_like(group_max_abs, fill_value=torch.inf)
    nonzero_group_radius_mask = group_inlier_radius_abs > 0
    group_paper_kr_ratios[nonzero_group_radius_mask] = (
        group_max_abs[nonzero_group_radius_mask]
        / group_inlier_radius_abs[nonzero_group_radius_mask]
    )
    zero_group_mask = (~nonzero_group_radius_mask) & (group_max_abs == 0)
    group_paper_kr_ratios[zero_group_mask] = 1.0
    flat_group_paper_kr_ratios = group_paper_kr_ratios.reshape(-1)
    group_paper_kr_finite_mask = torch.isfinite(flat_group_paper_kr_ratios)
    finite_group_paper_kr_ratios = flat_group_paper_kr_ratios[
        group_paper_kr_finite_mask
    ]
    group_paper_kr_finite_count = int(group_paper_kr_finite_mask.sum().item())
    group_paper_kr_infinite_count = int((~group_paper_kr_finite_mask).sum().item())
    group_paper_kr_finite_sum = (
        float(finite_group_paper_kr_ratios.sum().item())
        if group_paper_kr_finite_count > 0
        else 0.0
    )
    mean_group_paper_kr_ratio = float(flat_group_paper_kr_ratios.mean().item())
    any_group_paper_kr_ratio = float(flat_group_paper_kr_ratios.max().item())

    max_ratio_value, max_ratio_position = flat_ratios.max(dim=0)
    group_index = int(max_ratio_position.item())
    groups_per_row = reshaped.shape[1]
    row_index = group_index // groups_per_row
    inner_group_index = group_index % groups_per_row
    group_slice = reshaped[row_index, inner_group_index]
    top_values = top2_values.reshape(-1, 2)[group_index]
    top_indices = top2_indices.reshape(-1, 2)[group_index]
    group_topk_values = topk_values.reshape(-1, topk_value_count)[group_index]
    group_topk_indices = topk_indices.reshape(-1, topk_value_count)[group_index]

    top_groups_count = min(topk, flat_ratios.numel())
    top_group_ratios, top_group_positions = torch.topk(flat_ratios, k=top_groups_count)

    if ratio_finite_count > 0:
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

    group_threshold_summary = None
    group_paper_kr_threshold_summary = None
    if ratio_threshold is not None:
        greater_count = int((flat_ratios > ratio_threshold).sum().item())
        less_count = int((flat_ratios < ratio_threshold).sum().item())
        equal_count = int((flat_ratios == ratio_threshold).sum().item())
        total_count = int(flat_ratios.numel())
        group_threshold_summary = {
            "threshold": ratio_threshold,
            "total_group_count": total_count,
            "greater_than_count": greater_count,
            "greater_than_percentage": (
                greater_count / total_count if total_count else math.nan
            ),
            "less_than_count": less_count,
            "equal_count": equal_count,
        }
        group_paper_kr_greater_count = int(
            (flat_group_paper_kr_ratios > ratio_threshold).sum().item()
        )
        group_paper_kr_less_count = int(
            (flat_group_paper_kr_ratios < ratio_threshold).sum().item()
        )
        group_paper_kr_equal_count = int(
            (flat_group_paper_kr_ratios == ratio_threshold).sum().item()
        )
        group_paper_kr_threshold_summary = {
            "threshold": ratio_threshold,
            "total_group_count": total_count,
            "greater_than_count": group_paper_kr_greater_count,
            "greater_than_percentage": (
                group_paper_kr_greater_count / total_count if total_count else math.nan
            ),
            "less_than_count": group_paper_kr_less_count,
            "equal_count": group_paper_kr_equal_count,
        }

    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "num_groups": int(flat_ratios.numel()),
        "mean_group_max_ratio": mean_group_max_ratio,
        "any_group_max_ratio": any_group_max_ratio,
        "max_ratio_group_finite_sum": ratio_finite_sum,
        "max_ratio_group_finite_count": ratio_finite_count,
        "max_ratio_group_infinite_count": ratio_infinite_count,
        "mean_group_paper_kr_ratio": mean_group_paper_kr_ratio,
        "any_group_paper_kr_ratio": any_group_paper_kr_ratio,
        "whole_tensor_paper_kr_ratio": float(whole_tensor_paper_kr_ratio),
        "whole_tensor_paper_kr_max_abs": whole_tensor_max_abs,
        "whole_tensor_paper_kr_inlier_radius": whole_tensor_inlier_radius,
        # Backward-compatible aliases for earlier JSON consumers.
        "max_ratio": mean_group_max_ratio,
        "max_group_ratio": any_group_max_ratio,
        "paper_kr_ratio": mean_group_paper_kr_ratio,
        "paper_kr_max_group_ratio": any_group_paper_kr_ratio,
        "legacy_tensor_paper_kr_ratio": float(whole_tensor_paper_kr_ratio),
        "paper_kr_group_finite_sum": group_paper_kr_finite_sum,
        "paper_kr_group_finite_count": group_paper_kr_finite_count,
        "paper_kr_group_infinite_count": group_paper_kr_infinite_count,
        "paper_kr_percentile": kr_percentile,
        "group_paper_kr_rule": "signed_endpoint",
        "mean_ratio": finite_mean,
        "median_ratio": finite_median,
        "std_ratio": finite_std,
        "p95_ratio": finite_p95,
        "p99_ratio": finite_p99,
        "finite_group_count": ratio_finite_count,
        "infinite_group_count": ratio_infinite_count,
        "group_threshold_summary": group_threshold_summary,
        "group_paper_kr_threshold_summary": group_paper_kr_threshold_summary,
        "max_ratio_group": {
            "flat_index": group_index,
            "row_index": row_index,
            "group_index": inner_group_index,
            "largest_abs": float(top_values[0].item()),
            "second_abs": float(top_values[1].item()),
            "third_abs": (
                float(group_topk_values[2].item())
                if topk_value_count >= 3
                else float(top_values[1].item())
            ),
            "largest_abs_index": int(top_indices[0].item()),
            "second_abs_index": int(top_indices[1].item()),
            "third_abs_index": (
                int(group_topk_indices[2].item())
                if topk_value_count >= 3
                else int(top_indices[1].item())
            ),
            "values": [float(value.item()) for value in group_slice.detach().cpu()],
        },
        "top_groups": [
            {
                "flat_index": int(position.item()),
                "row_index": int(position.item() // groups_per_row),
                "group_index": int(position.item() % groups_per_row),
                "ratio": float(ratio.item()),
            }
            for ratio, position in zip(
                top_group_ratios.detach().cpu(),
                top_group_positions.detach().cpu(),
                strict=False,
            )
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
    sorted_records = sorted(
        tensor_records, key=lambda item: item["mean_group_max_ratio"], reverse=True
    )
    any_group_sorted_records = sorted(
        tensor_records, key=lambda item: item["any_group_max_ratio"], reverse=True
    )
    tensor_max_ratios = [
        record["mean_group_max_ratio"]
        for record in sorted_records
        if math.isfinite(record["mean_group_max_ratio"])
    ]
    tensor_paper_kr_ratios = [
        record["mean_group_paper_kr_ratio"]
        for record in sorted_records
        if math.isfinite(record["mean_group_paper_kr_ratio"])
    ]
    tensor_any_group_max_ratios = [
        record["any_group_max_ratio"]
        for record in sorted_records
        if math.isfinite(record["any_group_max_ratio"])
    ]
    tensor_any_group_paper_kr_ratios = [
        record["any_group_paper_kr_ratio"]
        for record in sorted_records
        if math.isfinite(record["any_group_paper_kr_ratio"])
    ]
    tensor_whole_tensor_paper_kr_ratios = [
        record["whole_tensor_paper_kr_ratio"]
        for record in sorted_records
        if math.isfinite(record["whole_tensor_paper_kr_ratio"])
    ]
    global_group_ratio_finite_count = sum(
        record["max_ratio_group_finite_count"] for record in sorted_records
    )
    global_group_ratio_infinite_count = sum(
        record["max_ratio_group_infinite_count"] for record in sorted_records
    )
    global_group_ratio_finite_sum = sum(
        record["max_ratio_group_finite_sum"] for record in sorted_records
    )
    global_max_group_ratio = max(
        (record["max_group_ratio"] for record in sorted_records), default=math.nan
    )
    if global_group_ratio_infinite_count > 0:
        global_group_mean_ratio = math.inf
    elif global_group_ratio_finite_count > 0:
        global_group_mean_ratio = (
            global_group_ratio_finite_sum / global_group_ratio_finite_count
        )
    else:
        global_group_mean_ratio = math.nan
    global_group_paper_kr_finite_count = sum(
        record["paper_kr_group_finite_count"] for record in sorted_records
    )
    global_group_paper_kr_infinite_count = sum(
        record["paper_kr_group_infinite_count"] for record in sorted_records
    )
    global_group_paper_kr_finite_sum = sum(
        record["paper_kr_group_finite_sum"] for record in sorted_records
    )
    global_group_paper_kr_max_ratio = max(
        (record["paper_kr_max_group_ratio"] for record in sorted_records),
        default=math.nan,
    )
    if global_group_paper_kr_infinite_count > 0:
        global_group_paper_kr_mean_ratio = math.inf
    elif global_group_paper_kr_finite_count > 0:
        global_group_paper_kr_mean_ratio = (
            global_group_paper_kr_finite_sum / global_group_paper_kr_finite_count
        )
    else:
        global_group_paper_kr_mean_ratio = math.nan
    threshold_summary = None
    paper_kr_threshold_summary = None
    whole_tensor_paper_kr_threshold_summary = None
    if ratio_threshold is not None:
        tensor_count = len(sorted_records)
        total_group_count = sum(record["num_groups"] for record in sorted_records)
        group_greater_count = sum(
            record["group_threshold_summary"]["greater_than_count"]
            for record in sorted_records
            if record["group_threshold_summary"] is not None
        )
        group_less_count = sum(
            record["group_threshold_summary"]["less_than_count"]
            for record in sorted_records
            if record["group_threshold_summary"] is not None
        )
        group_equal_count = sum(
            record["group_threshold_summary"]["equal_count"]
            for record in sorted_records
            if record["group_threshold_summary"] is not None
        )
        group_paper_kr_greater_count = sum(
            record["group_paper_kr_threshold_summary"]["greater_than_count"]
            for record in sorted_records
            if record["group_paper_kr_threshold_summary"] is not None
        )
        group_paper_kr_less_count = sum(
            record["group_paper_kr_threshold_summary"]["less_than_count"]
            for record in sorted_records
            if record["group_paper_kr_threshold_summary"] is not None
        )
        group_paper_kr_equal_count = sum(
            record["group_paper_kr_threshold_summary"]["equal_count"]
            for record in sorted_records
            if record["group_paper_kr_threshold_summary"] is not None
        )
        max_ratio_tensor_greater_count = sum(
            1
            for record in sorted_records
            if record["group_threshold_summary"] is not None
            and record["group_threshold_summary"]["greater_than_count"] > 0
        )
        max_ratio_tensor_equal_count = sum(
            1
            for record in sorted_records
            if record["group_threshold_summary"] is not None
            and record["group_threshold_summary"]["greater_than_count"] == 0
            and record["group_threshold_summary"]["equal_count"] > 0
        )
        max_ratio_tensor_less_count = (
            tensor_count - max_ratio_tensor_greater_count - max_ratio_tensor_equal_count
        )
        paper_kr_tensor_greater_count = sum(
            1
            for record in sorted_records
            if record["group_paper_kr_threshold_summary"] is not None
            and record["group_paper_kr_threshold_summary"]["greater_than_count"] > 0
        )
        paper_kr_tensor_equal_count = sum(
            1
            for record in sorted_records
            if record["group_paper_kr_threshold_summary"] is not None
            and record["group_paper_kr_threshold_summary"]["greater_than_count"] == 0
            and record["group_paper_kr_threshold_summary"]["equal_count"] > 0
        )
        paper_kr_tensor_less_count = (
            tensor_count - paper_kr_tensor_greater_count - paper_kr_tensor_equal_count
        )
        whole_tensor_paper_kr_greater_count = sum(
            1
            for record in sorted_records
            if record["whole_tensor_paper_kr_ratio"] > ratio_threshold
        )
        whole_tensor_paper_kr_less_count = sum(
            1
            for record in sorted_records
            if record["whole_tensor_paper_kr_ratio"] < ratio_threshold
        )
        whole_tensor_paper_kr_equal_count = sum(
            1
            for record in sorted_records
            if record["whole_tensor_paper_kr_ratio"] == ratio_threshold
        )
        threshold_summary = {
            "threshold": ratio_threshold,
            "tensor_count": {
                "total": tensor_count,
                "greater_than_count": max_ratio_tensor_greater_count,
                "greater_than_percentage": (
                    max_ratio_tensor_greater_count / tensor_count
                    if tensor_count
                    else math.nan
                ),
                "less_than_count": max_ratio_tensor_less_count,
                "equal_count": max_ratio_tensor_equal_count,
            },
            "group_count": {
                "total": total_group_count,
                "greater_than_count": group_greater_count,
                "greater_than_percentage": (
                    group_greater_count / total_group_count
                    if total_group_count
                    else math.nan
                ),
                "less_than_count": group_less_count,
                "equal_count": group_equal_count,
            },
        }
        paper_kr_threshold_summary = {
            "threshold": ratio_threshold,
            "tensor_count": {
                "total": tensor_count,
                "greater_than_count": paper_kr_tensor_greater_count,
                "greater_than_percentage": (
                    paper_kr_tensor_greater_count / tensor_count
                    if tensor_count
                    else math.nan
                ),
                "less_than_count": paper_kr_tensor_less_count,
                "equal_count": paper_kr_tensor_equal_count,
            },
            "group_count": {
                "total": total_group_count,
                "greater_than_count": group_paper_kr_greater_count,
                "greater_than_percentage": (
                    group_paper_kr_greater_count / total_group_count
                    if total_group_count
                    else math.nan
                ),
                "less_than_count": group_paper_kr_less_count,
                "equal_count": group_paper_kr_equal_count,
            },
        }
        whole_tensor_paper_kr_threshold_summary = {
            "threshold": ratio_threshold,
            "tensor_count": {
                "total": tensor_count,
                "greater_than_count": whole_tensor_paper_kr_greater_count,
                "greater_than_percentage": (
                    whole_tensor_paper_kr_greater_count / tensor_count
                    if tensor_count
                    else math.nan
                ),
                "less_than_count": whole_tensor_paper_kr_less_count,
                "equal_count": whole_tensor_paper_kr_equal_count,
            },
        }
    return {
        "model_dir": model_dir,
        "model_name": model_name,
        "group_size": group_size,
        "device": device,
        "paper_kr_percentile": kr_percentile,
        "group_paper_kr_rule": "signed_endpoint",
        "field_naming": {
            "primary_tensor_fields": [
                "mean_group_max_ratio",
                "any_group_max_ratio",
                "mean_group_paper_kr_ratio",
                "any_group_paper_kr_ratio",
                "whole_tensor_paper_kr_ratio",
            ],
            "primary_summary_fields": [
                "global_max_group_max_ratio",
                "global_mean_group_max_ratio",
                "global_max_mean_group_max_ratio",
                "global_mean_mean_group_max_ratio",
                "global_max_any_group_max_ratio",
                "global_mean_any_group_max_ratio",
                "global_max_group_paper_kr_ratio",
                "global_mean_group_paper_kr_ratio",
                "global_max_mean_group_paper_kr_ratio",
                "global_mean_mean_group_paper_kr_ratio",
                "global_max_any_group_paper_kr_ratio",
                "global_mean_any_group_paper_kr_ratio",
                "global_max_whole_tensor_paper_kr_ratio",
                "global_mean_whole_tensor_paper_kr_ratio",
            ],
            "compatibility_aliases": [
                "max_ratio",
                "max_group_ratio",
                "paper_kr_ratio",
                "paper_kr_max_group_ratio",
                "global_max_ratio",
                "global_mean_max_ratio",
                "global_max_paper_kr_ratio",
                "global_mean_paper_kr_ratio",
            ],
        },
        "tensor_count": len(tensor_records),
        "skipped_count": len(skipped),
        "global_max_group_max_ratio": global_max_group_ratio,
        "global_mean_group_max_ratio": global_group_mean_ratio,
        "global_max_mean_group_max_ratio": (
            max(tensor_max_ratios) if tensor_max_ratios else math.nan
        ),
        "global_mean_mean_group_max_ratio": (
            sum(tensor_max_ratios) / len(tensor_max_ratios)
            if tensor_max_ratios
            else math.nan
        ),
        "global_max_group_paper_kr_ratio": global_group_paper_kr_max_ratio,
        "global_mean_group_paper_kr_ratio": global_group_paper_kr_mean_ratio,
        "global_max_mean_group_paper_kr_ratio": (
            max(tensor_paper_kr_ratios) if tensor_paper_kr_ratios else math.nan
        ),
        "global_mean_mean_group_paper_kr_ratio": (
            sum(tensor_paper_kr_ratios) / len(tensor_paper_kr_ratios)
            if tensor_paper_kr_ratios
            else math.nan
        ),
        "global_max_any_group_max_ratio": (
            max(tensor_any_group_max_ratios)
            if tensor_any_group_max_ratios
            else math.nan
        ),
        "global_mean_any_group_max_ratio": (
            sum(tensor_any_group_max_ratios) / len(tensor_any_group_max_ratios)
            if tensor_any_group_max_ratios
            else math.nan
        ),
        "global_max_any_group_paper_kr_ratio": (
            max(tensor_any_group_paper_kr_ratios)
            if tensor_any_group_paper_kr_ratios
            else math.nan
        ),
        "global_mean_any_group_paper_kr_ratio": (
            sum(tensor_any_group_paper_kr_ratios)
            / len(tensor_any_group_paper_kr_ratios)
            if tensor_any_group_paper_kr_ratios
            else math.nan
        ),
        "global_max_whole_tensor_paper_kr_ratio": (
            max(tensor_whole_tensor_paper_kr_ratios)
            if tensor_whole_tensor_paper_kr_ratios
            else math.nan
        ),
        "global_mean_whole_tensor_paper_kr_ratio": (
            sum(tensor_whole_tensor_paper_kr_ratios)
            / len(tensor_whole_tensor_paper_kr_ratios)
            if tensor_whole_tensor_paper_kr_ratios
            else math.nan
        ),
        # Backward-compatible aliases for earlier JSON consumers.
        "global_max_ratio": global_max_group_ratio,
        "global_mean_max_ratio": global_group_mean_ratio,
        "global_max_tensor_max_ratio": (
            max(tensor_max_ratios) if tensor_max_ratios else math.nan
        ),
        "global_mean_tensor_max_ratio": (
            sum(tensor_max_ratios) / len(tensor_max_ratios)
            if tensor_max_ratios
            else math.nan
        ),
        "global_max_tensor_any_group_max_ratio": (
            max(tensor_any_group_max_ratios)
            if tensor_any_group_max_ratios
            else math.nan
        ),
        "global_mean_tensor_any_group_max_ratio": (
            sum(tensor_any_group_max_ratios) / len(tensor_any_group_max_ratios)
            if tensor_any_group_max_ratios
            else math.nan
        ),
        "global_max_ratio_group_count": global_group_ratio_finite_count
        + global_group_ratio_infinite_count,
        "global_max_paper_kr_ratio": global_group_paper_kr_max_ratio,
        "global_mean_paper_kr_ratio": global_group_paper_kr_mean_ratio,
        "global_max_tensor_paper_kr_ratio": (
            max(tensor_paper_kr_ratios) if tensor_paper_kr_ratios else math.nan
        ),
        "global_mean_tensor_paper_kr_ratio": (
            sum(tensor_paper_kr_ratios) / len(tensor_paper_kr_ratios)
            if tensor_paper_kr_ratios
            else math.nan
        ),
        "global_max_tensor_any_group_paper_kr_ratio": (
            max(tensor_any_group_paper_kr_ratios)
            if tensor_any_group_paper_kr_ratios
            else math.nan
        ),
        "global_mean_tensor_any_group_paper_kr_ratio": (
            sum(tensor_any_group_paper_kr_ratios)
            / len(tensor_any_group_paper_kr_ratios)
            if tensor_any_group_paper_kr_ratios
            else math.nan
        ),
        "global_max_legacy_tensor_paper_kr_ratio": (
            max(tensor_whole_tensor_paper_kr_ratios)
            if tensor_whole_tensor_paper_kr_ratios
            else math.nan
        ),
        "global_mean_legacy_tensor_paper_kr_ratio": (
            sum(tensor_whole_tensor_paper_kr_ratios)
            / len(tensor_whole_tensor_paper_kr_ratios)
            if tensor_whole_tensor_paper_kr_ratios
            else math.nan
        ),
        "global_paper_kr_group_count": global_group_paper_kr_finite_count
        + global_group_paper_kr_infinite_count,
        "ratio_threshold_summary": threshold_summary,
        "paper_kr_threshold_summary": paper_kr_threshold_summary,
        "whole_tensor_paper_kr_threshold_summary": whole_tensor_paper_kr_threshold_summary,
        "top_mean_group_tensors": sorted_records[:topk],
        "top_any_group_tensors": any_group_sorted_records[:topk],
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
    blended_transform = transforms.blended_transform_factory(
        axis.transAxes, axis.transData
    )
    axis.text(
        1.01,
        threshold,
        f"K/r = {format_metric(threshold)}",
        transform=blended_transform,
        color="#dc2626",
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#fca5a5",
            "alpha": 0.92,
        },
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
    blended_transform = transforms.blended_transform_factory(
        axis.transData, axis.transAxes
    )
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
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#fca5a5",
            "alpha": 0.92,
        },
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
    if (
        ".mlp." in tensor_name
        or ".feed_forward." in tensor_name
        or ".experts." in tensor_name
    ):
        return "mlp"
    if tensor_name.startswith(("model.norm", "norm")) or ".norm." in tensor_name:
        return "norm"
    return "other"


def extract_layer_name(tensor_name: str) -> str:
    match = re.search(r"(?:^|\.)(layers\.(\d+))(?:\.|$)", tensor_name)
    if match:
        return f"L{int(match.group(2)):02d}"
    for prefix in (
        "model.embed_tokens",
        "embed_tokens",
        "lm_head",
        "model.norm",
        "norm",
    ):
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


def build_layer_aggregates(
    tensor_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    layer_buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
    for record in tensor_records:
        paper_kr_ratio = record["mean_group_paper_kr_ratio"]
        any_group_paper_kr_ratio = record["any_group_paper_kr_ratio"]
        if not math.isfinite(paper_kr_ratio) and not math.isfinite(
            any_group_paper_kr_ratio
        ):
            continue
        layer_name = extract_layer_name(record["name"])
        module_family = get_module_family(record["name"])
        bucket = layer_buckets.setdefault(
            (layer_name, module_family), {"mean_group": [], "any_group": []}
        )
        if math.isfinite(paper_kr_ratio):
            bucket["mean_group"].append(paper_kr_ratio)
        if math.isfinite(any_group_paper_kr_ratio):
            bucket["any_group"].append(any_group_paper_kr_ratio)

    layer_records = []
    for (layer_name, module_family), values in layer_buckets.items():
        mean_group_values = values["mean_group"]
        any_group_values = values["any_group"]
        layer_records.append(
            {
                "layer_name": layer_name,
                "module_family": module_family,
                "tensor_count": max(len(mean_group_values), len(any_group_values)),
                "mean_paper_kr_ratio": (
                    sum(mean_group_values) / len(mean_group_values)
                    if mean_group_values
                    else math.nan
                ),
                "max_paper_kr_ratio": (
                    max(mean_group_values) if mean_group_values else math.nan
                ),
                "mean_any_group_paper_kr_ratio": (
                    sum(any_group_values) / len(any_group_values)
                    if any_group_values
                    else math.nan
                ),
                "max_any_group_paper_kr_ratio": (
                    max(any_group_values) if any_group_values else math.nan
                ),
                "sort_key": get_layer_sort_key(layer_name, module_family),
            }
        )

    return sorted(
        layer_records,
        key=lambda item: (
            item["sort_key"][0],
            item["sort_key"][1],
            MODULE_FAMILY_ORDER.get(item["module_family"], 99),
            item["layer_name"],
        ),
    )


def plot_paper_kr_figure(
    model_dir: str,
    model_name: str,
    tensor_records: list[dict[str, Any]],
    output_path: str,
    kr_percentile: float,
    topk: int,
    ratio_threshold: float | None,
    metric_key: str,
    metric_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite_records = [
        record for record in tensor_records if math.isfinite(record[metric_key])
    ]
    if not finite_records:
        raise ValueError(f"No finite {metric_key} values available for plotting")

    sorted_records = sorted(
        finite_records, key=lambda item: item[metric_key], reverse=True
    )
    rank_values = [record[metric_key] for record in sorted_records]
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
    curve_axis.scatter(
        ranks[:top_count], rank_values[:top_count], color="#134e4a", s=18, zorder=3
    )
    curve_axis.set_title(f"Sorted {metric_label} across tensors", fontsize=13, pad=10)
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
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#f0fdfa",
            "edgecolor": "#99f6e4",
        },
    )

    bar_labels = [shorten_tensor_name(record["name"]) for record in top_records]
    bar_values = [record[metric_key] for record in top_records]
    bar_colors = [
        "#f97316" if index == len(top_records) - 1 else "#fb923c"
        for index in range(len(top_records))
    ]
    bar_axis.barh(bar_labels, bar_values, color=bar_colors)
    bar_axis.set_title(
        f"Top {top_count} tensors by {metric_label}", fontsize=13, pad=10
    )
    bar_axis.set_xlabel("K/r ratio", fontsize=11)
    add_vertical_threshold_line(bar_axis, ratio_threshold)
    bar_axis.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.8)
    bar_axis.spines[["top", "right"]].set_visible(False)
    for label_index, value in enumerate(bar_values):
        bar_axis.text(
            value,
            label_index,
            f" {format_metric(value)}",
            va="center",
            ha="left",
            fontsize=9,
        )

    figure.suptitle(
        f"Tensor outlier profile for {model_name} ({metric_label})",
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
    mean_metric_key: str,
    max_metric_key: str,
    metric_label: str,
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
    record_lookup = {
        (record["layer_name"], record["module_family"]): record
        for record in layer_records
    }
    x_positions = list(range(len(ordered_layers)))
    bar_width = 0.78 / max(len(present_families), 1)

    figure, axis = plt.subplots(
        figsize=(max(10, len(layer_records) * 0.42), 5.8), constrained_layout=True
    )
    figure.patch.set_facecolor("white")
    for family_index, family in enumerate(present_families):
        offsets = [
            position + (family_index - (len(present_families) - 1) / 2) * bar_width
            for position in x_positions
        ]
        mean_values = []
        max_values = []
        for layer_name in ordered_layers:
            record = record_lookup.get((layer_name, family))
            mean_values.append(record[mean_metric_key] if record else math.nan)
            max_values.append(record[max_metric_key] if record else math.nan)
        axis.bar(
            offsets,
            mean_values,
            width=bar_width * 0.92,
            color=MODULE_FAMILY_COLORS[family],
            alpha=0.82,
            label=f"{family} mean",
        )
        finite_offsets = [
            offset
            for offset, value in zip(offsets, max_values, strict=False)
            if not math.isnan(value)
        ]
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
    axis.set_title(
        f"Layer-ordered {metric_label} by module family", fontsize=14, pad=10
    )
    axis.set_xlabel("Layer / module", fontsize=11)
    axis.set_ylabel("K/r ratio", fontsize=11)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(ordered_layers, rotation=60, ha="right", fontsize=9)
    axis.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    add_threshold_line(axis, ratio_threshold)
    axis.legend(frameon=False, loc="upper right", ncols=2, fontsize=9)
    all_positive_means = [
        record[mean_metric_key]
        for record in layer_records
        if record[mean_metric_key] > 0
    ]
    all_max_values = [record[max_metric_key] for record in layer_records]
    if all_positive_means and max(all_max_values) / min(all_positive_means) >= 20:
        axis.set_yscale("log")

    stats_text = (
        f"aggregated layers = {len(layer_records)}\n"
        f"max layer K/r = {format_metric(max(all_max_values))}\n"
        f"mean layer K/r = {format_metric(sum(record[mean_metric_key] for record in layer_records) / len(layer_records))}\n"
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
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#eff6ff",
            "edgecolor": "#93c5fd",
        },
    )

    figure.suptitle(
        f"Layer outlier profile for {model_name} ({metric_label})",
        fontsize=15,
        fontweight="bold",
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def profile_model(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tensor_records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for name, tensor, source_file in iter_named_tensors(args.input):
        if not should_include_tensor(name, args.include, args.exclude):
            continue

        try:
            stats = compute_group_ratio_stats(
                tensor,
                args.group_size,
                args.device,
                args.topk,
                args.kr_percentile,
                args.ratio_threshold,
            )
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
            f"mean_group_max_ratio={format_metric(stats['mean_group_max_ratio'])} "
            f"any_group_max_ratio={format_metric(stats['any_group_max_ratio'])} "
            f"mean_group_paper_kr_ratio={format_metric(stats['mean_group_paper_kr_ratio'])} "
            f"any_group_paper_kr_ratio={format_metric(stats['any_group_paper_kr_ratio'])}"
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
        args.output = os.path.join(
            args.input, f"outlier_profile_g{args.group_size}.json"
        )
    else:
        args.output = normalize_local_path(args.output)

    if args.per_tensor_output is not None:
        args.per_tensor_output = normalize_local_path(args.per_tensor_output)
    if args.figure_output is None:
        output_stem, _ = os.path.splitext(args.output)
        args.figure_output = f"{output_stem}_mean_group_paper_kr.png"
    else:
        args.figure_output = normalize_local_path(args.figure_output)
    mean_group_figure_output = args.figure_output
    any_group_figure_output = f"{os.path.splitext(args.figure_output)[0].removesuffix('_mean_group_paper_kr')}_any_group_paper_kr.png"
    mean_group_layer_figure_output = (
        f"{os.path.splitext(mean_group_figure_output)[0]}_by_layer.png"
    )
    any_group_layer_figure_output = (
        f"{os.path.splitext(any_group_figure_output)[0]}_by_layer.png"
    )
    model_name = resolve_model_display_name(args.input, args.model_name)

    summary, tensor_records = profile_model(args)
    layer_records = build_layer_aggregates(tensor_records)
    save_json(args.output, summary)
    if args.per_tensor_output:
        save_jsonl(args.per_tensor_output, tensor_records)
    plot_paper_kr_figure(
        model_dir=args.input,
        model_name=model_name,
        tensor_records=tensor_records,
        output_path=mean_group_figure_output,
        kr_percentile=args.kr_percentile,
        topk=args.topk,
        ratio_threshold=args.ratio_threshold,
        metric_key="mean_group_paper_kr_ratio",
        metric_label="mean-group paper K/r",
    )
    plot_paper_kr_figure(
        model_dir=args.input,
        model_name=model_name,
        tensor_records=tensor_records,
        output_path=any_group_figure_output,
        kr_percentile=args.kr_percentile,
        topk=args.topk,
        ratio_threshold=args.ratio_threshold,
        metric_key="any_group_paper_kr_ratio",
        metric_label="any-group paper K/r",
    )
    plot_layer_paper_kr_figure(
        model_dir=args.input,
        model_name=model_name,
        layer_records=layer_records,
        output_path=mean_group_layer_figure_output,
        kr_percentile=args.kr_percentile,
        ratio_threshold=args.ratio_threshold,
        mean_metric_key="mean_paper_kr_ratio",
        max_metric_key="max_paper_kr_ratio",
        metric_label="mean-group paper K/r",
    )
    plot_layer_paper_kr_figure(
        model_dir=args.input,
        model_name=model_name,
        layer_records=layer_records,
        output_path=any_group_layer_figure_output,
        kr_percentile=args.kr_percentile,
        ratio_threshold=args.ratio_threshold,
        mean_metric_key="mean_any_group_paper_kr_ratio",
        max_metric_key="max_any_group_paper_kr_ratio",
        metric_label="any-group paper K/r",
    )
    summary["model_name"] = model_name
    summary["figure_output"] = mean_group_figure_output
    summary["any_group_figure_output"] = any_group_figure_output
    summary["layer_figure_output"] = mean_group_layer_figure_output
    summary["any_group_layer_figure_output"] = any_group_layer_figure_output
    summary["top_mean_group_layers"] = sorted(
        layer_records, key=lambda item: item["max_paper_kr_ratio"], reverse=True
    )[: args.topk]
    summary["top_any_group_layers"] = sorted(
        layer_records,
        key=lambda item: item["max_any_group_paper_kr_ratio"],
        reverse=True,
    )[: args.topk]
    save_json(args.output, summary)

    print(f"Profile saved to: {args.output}")
    print(f"Mean-group figure saved to: {mean_group_figure_output}")
    print(f"Any-group figure saved to: {any_group_figure_output}")
    print(f"Mean-group layer figure saved to: {mean_group_layer_figure_output}")
    print(f"Any-group layer figure saved to: {any_group_layer_figure_output}")
    print(f"Profiled tensors: {summary['tensor_count']}")
    print(f"Skipped tensors: {summary['skipped_count']}")
    print(
        "Global tensor mean-group max ratio: "
        f"max={summary['global_max_mean_group_max_ratio']} "
        f"mean={summary['global_mean_mean_group_max_ratio']}"
    )
    print(
        "Global tensor any-group max ratio: "
        f"max={summary['global_max_any_group_max_ratio']} "
        f"mean={summary['global_mean_any_group_max_ratio']}"
    )
    print(
        "Global group max ratio: "
        f"max={summary['global_max_group_max_ratio']} "
        f"mean={summary['global_mean_group_max_ratio']} "
        f"across {summary['global_max_ratio_group_count']} groups"
    )
    print(
        "Global tensor mean-group paper-style K/r: "
        f"max={summary['global_max_mean_group_paper_kr_ratio']} "
        f"mean={summary['global_mean_mean_group_paper_kr_ratio']}"
    )
    print(
        "Global tensor any-group paper-style K/r: "
        f"max={summary['global_max_any_group_paper_kr_ratio']} "
        f"mean={summary['global_mean_any_group_paper_kr_ratio']}"
    )
    print(
        "Global tensor whole-tensor paper-style K/r (legacy 690b2826... semantics): "
        f"max={summary['global_max_whole_tensor_paper_kr_ratio']} "
        f"mean={summary['global_mean_whole_tensor_paper_kr_ratio']} "
        f"(r estimated by p={summary['paper_kr_percentile']})"
    )
    print(
        "Global group paper-style K/r: "
        f"max={summary['global_max_group_paper_kr_ratio']} "
        f"mean={summary['global_mean_group_paper_kr_ratio']} "
        f"across {summary['global_paper_kr_group_count']} groups"
    )
    if summary["ratio_threshold_summary"] is not None:
        threshold_summary = summary["ratio_threshold_summary"]
        print(
            "Threshold summary (tensor max_ratio, any group): "
            f"> {threshold_summary['threshold']}: {threshold_summary['tensor_count']['greater_than_count']} "
            f"({format_metric(threshold_summary['tensor_count']['greater_than_percentage'] * 100)}%), "
            f"< {threshold_summary['threshold']}: {threshold_summary['tensor_count']['less_than_count']}, "
            f"= {threshold_summary['threshold']}: {threshold_summary['tensor_count']['equal_count']}"
        )
        print(
            "Threshold summary (group max_ratio): "
            f"> {threshold_summary['threshold']}: {threshold_summary['group_count']['greater_than_count']} "
            f"({format_metric(threshold_summary['group_count']['greater_than_percentage'] * 100)}%), "
            f"< {threshold_summary['threshold']}: {threshold_summary['group_count']['less_than_count']}, "
            f"= {threshold_summary['threshold']}: {threshold_summary['group_count']['equal_count']}"
        )
        paper_kr_threshold_summary = summary["paper_kr_threshold_summary"]
        print(
            "Threshold summary (tensor paper K/r, any group): "
            f"> {paper_kr_threshold_summary['threshold']}: {paper_kr_threshold_summary['tensor_count']['greater_than_count']} "
            f"({format_metric(paper_kr_threshold_summary['tensor_count']['greater_than_percentage'] * 100)}%), "
            f"< {paper_kr_threshold_summary['threshold']}: {paper_kr_threshold_summary['tensor_count']['less_than_count']}, "
            f"= {paper_kr_threshold_summary['threshold']}: {paper_kr_threshold_summary['tensor_count']['equal_count']}"
        )
        whole_tensor_paper_kr_threshold_summary = summary[
            "whole_tensor_paper_kr_threshold_summary"
        ]
        print(
            "Threshold summary (tensor paper K/r, whole tensor legacy 690b2826... semantics): "
            f"> {whole_tensor_paper_kr_threshold_summary['threshold']}: {whole_tensor_paper_kr_threshold_summary['tensor_count']['greater_than_count']} "
            f"({format_metric(whole_tensor_paper_kr_threshold_summary['tensor_count']['greater_than_percentage'] * 100)}%), "
            f"< {whole_tensor_paper_kr_threshold_summary['threshold']}: {whole_tensor_paper_kr_threshold_summary['tensor_count']['less_than_count']}, "
            f"= {whole_tensor_paper_kr_threshold_summary['threshold']}: {whole_tensor_paper_kr_threshold_summary['tensor_count']['equal_count']}"
        )
        print(
            "Threshold summary (group paper K/r): "
            f"> {paper_kr_threshold_summary['threshold']}: {paper_kr_threshold_summary['group_count']['greater_than_count']} "
            f"({format_metric(paper_kr_threshold_summary['group_count']['greater_than_percentage'] * 100)}%), "
            f"< {paper_kr_threshold_summary['threshold']}: {paper_kr_threshold_summary['group_count']['less_than_count']}, "
            f"= {paper_kr_threshold_summary['threshold']}: {paper_kr_threshold_summary['group_count']['equal_count']}"
        )


if __name__ == "__main__":
    parser = get_option_parser()
    run(parser.parse_args())
