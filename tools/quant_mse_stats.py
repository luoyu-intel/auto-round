from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from auto_round.data_type.int import dynamic_quantize_tensor
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare legacy one-dimensional asym quantization search with hybrid refine.")
    parser.add_argument("path", type=Path, help="Path to a safetensors shard")
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--chunk-groups", type=int, default=8192)
    parser.add_argument("--device", default="xpu", choices=["cpu", "xpu"], help="Compute device for chunked MSE stats")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU device was requested but is not available in this PyTorch environment")
        return torch.device("xpu")
    return torch.device("cpu")


def categorize_tensor(name: str) -> str:
    if "embed_tokens" in name:
        return "embed_tokens"
    if ".mlp.down_proj.weight" in name:
        return "mlp.down_proj"
    if ".mlp.gate_proj.weight" in name:
        return "mlp.gate_proj"
    if ".mlp.up_proj.weight" in name:
        return "mlp.up_proj"
    if ".self_attn.q_proj.weight" in name:
        return "self_attn.q_proj"
    if ".self_attn.k_proj.weight" in name:
        return "self_attn.k_proj"
    if ".self_attn.v_proj.weight" in name:
        return "self_attn.v_proj"
    if ".self_attn.o_proj.weight" in name:
        return "self_attn.o_proj"
    return "other"


def legacy_one_dim(arr: torch.Tensor, bits: int, iters: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_min = arr.min(-1, keepdim=True)[0].clamp(max=0)
    a_max = arr.max(-1, keepdim=True)[0].clamp(min=0)
    maxq = (1 << bits) - 1
    fullq = 1 << (bits - 1)
    denorm = a_max - a_min

    def asym_quant_iter(search_q: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inverse_scale = search_q / denorm
        inverse_scale = torch.where(denorm.abs() <= 1e-4, torch.ones_like(inverse_scale), inverse_scale)
        zp = torch.round(-a_min * inverse_scale).clamp(0, maxq)
        q = torch.clamp(torch.round(arr * inverse_scale + zp), 0, maxq)
        scale = 1 / inverse_scale
        centered_q = q - fullq
        centered_zp = zp - fullq
        loss = torch.sum((scale * (centered_q - centered_zp) - arr).pow(2), dim=-1, keepdim=True)
        return loss, centered_q, scale, centered_zp

    err, qarr, scale, zp = asym_quant_iter(maxq)
    if iters > 1:
        delta = 4 / (iters - 1)
        start_q = maxq - 0.5
        for _ in range(iters - 1):
            candidate = asym_quant_iter(start_q)
            replace_id = candidate[0] < err
            qarr = torch.where(replace_id, candidate[1], qarr)
            scale = torch.where(replace_id, candidate[2], scale)
            zp = torch.where(replace_id, candidate[3], zp)
            err = torch.where(replace_id, candidate[0], err)
            start_q += delta
    return qarr, scale, zp


def chunk_sse(grouped: torch.Tensor, bits: int, iters: int, chunk_groups: int, device: torch.device) -> tuple[float, float]:
    legacy_sse = 0.0
    hybrid_sse = 0.0
    for start in range(0, grouped.shape[0], chunk_groups):
        chunk = grouped[start:start + chunk_groups].to(device)
        legacy_q, legacy_scale, legacy_zp = legacy_one_dim(chunk, bits=bits, iters=iters)
        hybrid_q, hybrid_scale, hybrid_zp = dynamic_quantize_tensor(chunk, bits=bits, dir=-1, asym=True, iter=iters, qw=None)

        legacy_sse += float(torch.sum((legacy_scale * (legacy_q - legacy_zp) - chunk).pow(2)).cpu().item())
        hybrid_sse += float(torch.sum((hybrid_scale * (hybrid_q - hybrid_zp) - chunk).pow(2)).cpu().item())
    return legacy_sse, hybrid_sse


def summarize_by_category(rows: list[dict]) -> list[dict]:
    bucket: dict[str, dict[str, float]] = {}
    for row in rows:
        category = row["category"]
        stats = bucket.setdefault(
            category,
            {"legacy_sse": 0.0, "hybrid_sse": 0.0, "numel": 0, "tensor_count": 0},
        )
        stats["legacy_sse"] += row["legacy_sse"]
        stats["hybrid_sse"] += row["hybrid_sse"]
        stats["numel"] += row["padded_numel"]
        stats["tensor_count"] += 1

    summary = []
    for category, stats in bucket.items():
        legacy_mse = stats["legacy_sse"] / stats["numel"]
        hybrid_mse = stats["hybrid_sse"] / stats["numel"]
        improvement_pct = 100.0 * (legacy_mse - hybrid_mse) / legacy_mse if legacy_mse > 0 else 0.0
        summary.append(
            {
                "category": category,
                "tensor_count": int(stats["tensor_count"]),
                "legacy_mse": legacy_mse,
                "hybrid_mse": hybrid_mse,
                "improvement_pct": improvement_pct,
                "sse_reduction": stats["legacy_sse"] - stats["hybrid_sse"],
            }
        )
    return sorted(summary, key=lambda item: item["improvement_pct"], reverse=True)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    start_time = time.time()
    rows: list[dict] = []
    total_legacy_sse = 0.0
    total_hybrid_sse = 0.0
    total_elems = 0
    num_tensors = 0
    num_groups = 0

    with safe_open(args.path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name)
            if tensor.ndim != 2 or not tensor.is_floating_point():
                continue

            grouped, _, _ = reshape_pad_tensor_by_group_size(tensor.to(torch.float32), args.group_size)
            legacy_sse, hybrid_sse = chunk_sse(grouped, args.bits, args.iters, args.chunk_groups, device)
            padded_numel = grouped.numel()
            legacy_mse = legacy_sse / padded_numel
            hybrid_mse = hybrid_sse / padded_numel
            improvement_pct = 100.0 * (legacy_mse - hybrid_mse) / legacy_mse if legacy_mse > 0 else 0.0

            row = {
                "name": name,
                "category": categorize_tensor(name),
                "shape": list(tensor.shape),
                "groups": int(grouped.shape[0]),
                "padded_numel": int(padded_numel),
                "legacy_sse": legacy_sse,
                "hybrid_sse": hybrid_sse,
                "legacy_mse": legacy_mse,
                "hybrid_mse": hybrid_mse,
                "improvement_pct": improvement_pct,
                "sse_reduction": legacy_sse - hybrid_sse,
            }
            rows.append(row)
            total_legacy_sse += legacy_sse
            total_hybrid_sse += hybrid_sse
            total_elems += padded_numel
            num_tensors += 1
            num_groups += grouped.shape[0]
            print(f"processed {num_tensors:3d} tensors | groups={grouped.shape[0]:8d} | {name}", flush=True)

    overall_legacy_mse = total_legacy_sse / total_elems
    overall_hybrid_mse = total_hybrid_sse / total_elems
    overall_improvement_pct = 100.0 * (overall_legacy_mse - overall_hybrid_mse) / overall_legacy_mse if overall_legacy_mse > 0 else 0.0

    improved = [row for row in rows if row["hybrid_mse"] + 1e-18 < row["legacy_mse"]]
    degraded = [row for row in rows if row["hybrid_mse"] > row["legacy_mse"] + 1e-18]
    unchanged = len(rows) - len(improved) - len(degraded)
    by_pct = sorted(rows, key=lambda item: item["improvement_pct"], reverse=True)
    by_delta = sorted(rows, key=lambda item: item["sse_reduction"], reverse=True)
    category_summary = summarize_by_category(rows)

    result = {
        "summary": {
            "path": str(args.path),
            "device": str(device),
            "bits": args.bits,
            "group_size": args.group_size,
            "iters": args.iters,
            "chunk_groups": args.chunk_groups,
            "num_2d_tensors": num_tensors,
            "num_groups": num_groups,
            "exact_overall_legacy_mse": overall_legacy_mse,
            "exact_overall_hybrid_mse": overall_hybrid_mse,
            "exact_relative_improvement_pct": overall_improvement_pct,
            "improved_tensor_count": len(improved),
            "degraded_tensor_count": len(degraded),
            "unchanged_tensor_count": unchanged,
            "elapsed_sec": time.time() - start_time,
        },
        "top_by_relative_improvement": by_pct[: args.top_k],
        "top_by_sse_reduction": by_delta[: args.top_k],
        "category_summary": category_summary,
        "degraded_tensors": sorted(degraded, key=lambda item: item["improvement_pct"]),
    }

    print("RESULT_JSON_START")
    print(json.dumps(result, indent=2))
    print("RESULT_JSON_END")


if __name__ == "__main__":
    main()
