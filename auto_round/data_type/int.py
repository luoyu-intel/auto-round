# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Tuple, Union

import torch

from auto_round.data_type.register import register_dtype
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad, round_ste
from auto_round.logger import logger
from auto_round.utils import get_reciprocal


def search_scales(data: torch.Tensor, bits: int, qw: Union[None, torch.Tensor, float] = None) -> torch.Tensor:
    # Maximum absolute value for symmetric quantization
    nmax = 1 << (bits - 1)  # equivalent to pow(2, bits-1)

    # Find per-group max along the last dimension
    imax = torch.abs(data).argmax(dim=-1, keepdim=True)
    group_max = torch.take_along_dim(data, imax, dim=-1)

    # Compute initial inverse scales
    iscales = -nmax * get_reciprocal(group_max)
    scales = get_reciprocal(iscales)  # scale = 1 / iscales

    # Initial quantized values (in-place round and clamp)
    L = torch.empty_like(data)
    torch.round(iscales * data, out=L)
    L.clamp_(-nmax, nmax - 1)

    # Set default weight if None
    if qw is None:
        qw = 1.0

    # Compute initial best loss
    best_loss = ((scales * L - data).to(torch.float32)) ** 2
    if isinstance(qw, torch.Tensor):
        best_loss.mul_(qw)  # inplace multiply by weight
    best_loss = torch.sum(best_loss, dim=-1)

    # Iterative search over small adjustments
    for _is in range(-18 * 5, 18 * 5 + 1):
        if _is == 0:
            continue

        # Update iscales in-place
        iscales_tmp = -(nmax - 0.01 * _is) * get_reciprocal(group_max)

        # Compute temporary quantized values (in-place round + clamp)
        tmp_L = torch.empty_like(data)
        torch.round(iscales_tmp * data, out=tmp_L)
        tmp_L.clamp_(-nmax, nmax - 1)

        # Compute temporary scales
        tmp_scales = get_reciprocal(iscales_tmp)

        # Compute temporary loss
        loss = ((tmp_scales * tmp_L - data).to(torch.float32)) ** 2
        if isinstance(qw, torch.Tensor):
            loss.mul_(qw)
        loss = torch.sum(loss, dim=-1)

        # Replace scales where loss improves (in-place)
        replace_id = loss < best_loss
        if replace_id.any():
            scales[replace_id] = tmp_scales[replace_id]
            best_loss[replace_id] = loss[replace_id]

    return scales


def _compute_asym_search_loss(
    data: torch.Tensor,
    scale: torch.Tensor,
    zp: torch.Tensor,
    maxq: int,
    qw: Union[None, torch.Tensor, float] = None,
) -> torch.Tensor:
    inverse_scale = get_reciprocal(scale)
    q = torch.empty_like(data)
    torch.round(data * inverse_scale + zp, out=q)
    q.clamp_(0, maxq)

    loss = ((scale * (q - zp) - data).to(torch.float32)) ** 2
    if isinstance(qw, torch.Tensor):
        loss.mul_(qw)
    elif qw is not None:
        loss.mul_(float(qw))
    return torch.sum(loss, dim=-1, keepdim=True)


def search_scales_zp(
    data: torch.Tensor,
    bits: int,
    qw: Union[None, torch.Tensor, float] = None,
    q_scale_thresh: float = 1e-5,
    search_steps: int = 8,
    step_size: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Jointly search asymmetric scale and zero-point with reconstruction loss."""
    data = data.to(torch.float32)
    maxq = 2**bits - 1

    group_min = torch.min(data, dim=-1, keepdim=True)[0]
    group_max = torch.max(data, dim=-1, keepdim=True)[0]
    group_range = torch.clamp(group_max - group_min, min=q_scale_thresh * maxq)

    scale = torch.clamp(group_range / maxq, min=q_scale_thresh)
    zp = torch.clamp(torch.round(-group_min * get_reciprocal(scale)), 0, maxq)
    best_loss = _compute_asym_search_loss(data, scale, zp, maxq, qw=qw)

    for min_step in range(-search_steps, search_steps + 1):
        candidate_min = group_min + min_step * step_size * group_range
        for max_step in range(-search_steps, search_steps + 1):
            if min_step == 0 and max_step == 0:
                continue

            candidate_max = group_max + max_step * step_size * group_range
            valid = candidate_max > candidate_min
            candidate_scale = torch.clamp((candidate_max - candidate_min) / maxq, min=q_scale_thresh)
            candidate_zp = torch.clamp(torch.round(-candidate_min * get_reciprocal(candidate_scale)), 0, maxq)
            loss = _compute_asym_search_loss(data, candidate_scale, candidate_zp, maxq, qw=qw)

            replace_id = valid & (loss < best_loss)
            if replace_id.any():
                scale = torch.where(replace_id, candidate_scale, scale)
                zp = torch.where(replace_id, candidate_zp, zp)
                best_loss = torch.where(replace_id, loss, best_loss)

    return scale, zp

def t_DQ(Q):
    if len(Q) == 3:
        return (Q[0] - Q[2]) * Q[1]
    return Q[0] * Q[1]


def t_dequant_weight(Q_main, Q_res, shape, n_iter=100):
    dq = t_DQ(Q_main).reshape(shape)
    n = min(len(Q_res), n_iter)
    for i in range(n):
        dq += t_DQ(Q_res[i]).reshape(shape)
    return dq

def t_dyn_quant(arr, bits=4, dir=-1, asym=True, iter=2, qw=None):
    a_min = arr.min(dir, keepdim=True)[0]
    a_min = torch.clamp(a_min, max=0)
    a_max = arr.max(dir, keepdim=True)[0]
    a_max = torch.clamp(a_max, min=0)
    a_min_abs = torch.abs(a_min)
    a_max_abs = torch.abs(a_max)
    a_absmax = torch.max(a_min_abs, a_max_abs)
    Q = (1 << bits) - 1
    FullQ = 1 << (bits - 1)
    denorm = a_max - a_min

    def asym_quant_iter(_Q):
        scale0 = _Q / denorm
        scale0[denorm.abs() <= 1e-4] = 1
        zero_point0 = torch.round(-a_min * scale0)
        zero_point0 = torch.clamp(zero_point0, 0, Q) - FullQ
        qarr0 = torch.clamp(torch.round(arr * scale0 + zero_point0), 0, Q) - FullQ
        scale0 = 1 / scale0
        dq0 = t_dequant_weight([qarr0, scale0, zero_point0], [], arr.shape)
        err0 = abs(arr - dq0)
        err0 = torch.sum(err0, dim=dir, keepdim=True)
        if qw is not None:
            err0.mul_(qw)
        return err0, qarr0, scale0, zero_point0

    if asym:
        StartQ = Q
        err, qarr, scale, zero_point = asym_quant_iter(StartQ)
        if iter > 1:
            delta = 4 / (iter - 1)
            StartQ = Q - 0.5
            for i in range(iter - 1):
                _ret = asym_quant_iter(StartQ)
                qarr = torch.where(_ret[0] < err, _ret[1], qarr)
                scale = torch.where(_ret[0] < err, _ret[2], scale)
                zero_point = torch.where(_ret[0] < err, _ret[3], zero_point)
                err = torch.where(_ret[0] < err, _ret[0], err)
                StartQ += delta
        qarr_a = qarr
        scale_a = scale
        zero_point_a = zero_point
        err_a = err

    def full_quant_iter(_Q):
        # fullrange
        max_v = (2 * (a_min_abs > a_max_abs).int() - 1) * a_absmax
        scale1 = max_v / _Q
        scale1[scale1 == 0] = 1
        qarr1 = arr / scale1
        qarr1 = torch.round(qarr1)
        qarr1 = torch.clamp(qarr1, -FullQ, FullQ - 1)
        dq1 = t_dequant_weight([qarr1, scale1], [], arr.shape)
        err1 = abs(arr - dq1)
        err1 = torch.sum(err1, dim=dir, keepdim=True)
        if qw is not None:
            err1.mul_(qw)
        return err1, qarr1, scale1

    StartQ = FullQ
    err, qarr, scale = full_quant_iter(StartQ)
    if iter > 1:
        delta = 4 / (iter - 1)
        StartQ = FullQ - 1
        for i in range(iter - 1):
            _ret = full_quant_iter(StartQ)
            qarr = torch.where(_ret[0] < err, _ret[1], qarr)
            scale = torch.where(_ret[0] < err, _ret[2], scale)
            err = torch.where(_ret[0] < err, _ret[0], err)
            StartQ += delta
    if asym:
        zero_point = torch.zeros_like(zero_point_a)
        qarr = torch.where(err_a < err, qarr_a, qarr)
        scale = torch.where(err_a < err, scale_a, scale)
        zero_point = torch.where(err_a < err, zero_point_a, zero_point)
        return qarr, scale, zero_point
    else:
        return qarr, scale


@register_dtype("rtn_int_sym")
def quant_tensor_rtn_sym(tensor, bits=4, group_size=-1, v=0, q_scale_thresh=1e-5, imatrix=None, **kwargs):
    """Quantize and de-quantize tensor asymmetrically. full range, credict goes to llamacpp community

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """
    from auto_round.data_type.gguf import _imatrix_handle_zero

    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = 2 ** (bits - 1)
    if imatrix is None:
        imatrix = 1.0
    else:
        imatrix = imatrix.reshape(1, -1)
        imatrix = reshape_pad_tensor_by_group_size(imatrix, group_size, val=1e-5)[0].view(1, -1)
        imatrix = imatrix.expand(tensor.numel() // imatrix.numel(), -1)
        imatrix = imatrix.reshape(tensor.shape)

        imatrix = _imatrix_handle_zero(imatrix, tensor, bits)

    scale = search_scales(tensor, bits, qw=imatrix)
    scale = torch.where(scale < 0, torch.clamp(scale, max=-q_scale_thresh), torch.clamp(scale, min=q_scale_thresh))
    int_w = tensor.div(scale).round_().clamp_(-maxq, maxq - 1)
    qdq_result = (int_w.mul_(scale)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, maxq


@register_dtype("rtn_int_asym")
def quant_tensor_rtn_asym(tensor, bits=4, group_size=-1, v=0, q_scale_thresh=1e-5, imatrix=None, **kwargs):
    """Quantize and de-quantize tensor asymmetrically. full range, credict goes to llamacpp community

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """
    from auto_round.data_type.gguf import _imatrix_handle_zero

    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = 2 ** (bits - 1)
    if imatrix is None:
        imatrix = 1.0
    else:
        imatrix = imatrix.reshape(1, -1)
        imatrix = reshape_pad_tensor_by_group_size(imatrix, group_size, val=1e-5)[0].view(1, -1)
        imatrix = imatrix.expand(tensor.numel() // imatrix.numel(), -1)
        imatrix = imatrix.reshape(tensor.shape)

        imatrix = _imatrix_handle_zero(imatrix, tensor, bits)
    if True:
        q, scale, zp =t_dyn_quant(tensor, bits=bits, dir=-1, asym=True, iter=100, qw=imatrix)
    else:
        scale, zp = search_scales_zp(tensor, bits, qw=imatrix)
        scale = torch.where(scale < 0, torch.clamp(scale, max=-q_scale_thresh), torch.clamp(scale, min=q_scale_thresh))
        q = torch.round(tensor / scale  + zp)
        q = torch.clamp(q, 0, maxq)
    qdq_result = (scale * (q - zp)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, zp

@register_dtype("int_sym")
def quant_tensor_sym(
    tensor,
    bits=4,
    group_size=-1,
    v=0,
    min_scale=1.0,
    max_scale=1.0,
    scale_dtype=torch.float16,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh=1e-5,
    **kwargs
):
    """Quantize and de-quantize tensor asymmetrically. full range, credict goes to llamacpp community

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for tensor
        max_scale: Maximum scale coefficient for tensor
        tensor_min (Tensor, optional): Minimum tensor value for quantization. Defaults to None.
        tensor_max (Tensor, optional): Maximum tensor value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """

    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = 2 ** (bits - 1)
    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max

    wmin_abs = -(wmin_tmp * min_scale)  # pylint: disable=E1130
    wmax_abs = wmax_tmp * max_scale
    max_v = (2 * (wmax_abs < wmin_abs).int() - 1) * torch.max(wmax_abs, wmin_abs)
    scale = (max_v / maxq).to(scale_dtype)
    scale = torch.where(scale < 0, torch.clamp(scale, max=-q_scale_thresh), torch.clamp(scale, min=q_scale_thresh))
    scale = scale.unsqueeze(dim=-1)
    int_w = round_ste(tensor / scale + v)
    q = torch.clamp(int_w, -maxq, maxq - 1)
    qdq_result = (scale * q).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, maxq


@register_dtype("int_asym")
def quant_tensor_asym(
    tensor,
    bits=4,
    group_size=-1,
    v=0,
    min_scale=1.0,
    max_scale=1.0,
    scale_dtype=torch.float16,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh=1e-5,
    **kwargs
):
    """Quantize and de-quantize tensor asymmetrically.

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for tensor
        max_scale: Maximum scale coefficient for tensor
        tensor_min (Tensor, optional): Minimum tensor value for quantization. Defaults to None.
        tensor_max (Tensor, optional): Maximum tensor value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = 2**bits - 1
    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max
    if isinstance(min_scale, torch.Tensor):
        wmin = wmin_tmp * min_scale
        wmax = wmax_tmp * max_scale
    else:
        wmin = wmin_tmp
        wmax = wmax_tmp
    scale = ((wmax - wmin) / maxq).to(scale_dtype)
    scale = torch.clamp(scale, min=q_scale_thresh)
    zp = round_ste(-wmin / scale)  # pylint: disable=E1130
    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    int_w = round_ste(tensor / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    qdq_result = (scale * (q - zp)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, zp


@register_dtype("int_sym_gptq")
def quant_tensor_sym_gptq(
    tensor,
    bits=4,
    group_size=-1,
    v=0,
    min_scale=1.0,
    max_scale=1.0,
    scale_dtype=torch.float16,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh=1e-5,
    **kwargs
):
    """Quantize and de-quantize tensor asymmetrically.

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for tensor
        max_scale: Maximum scale coefficient for tensor
        tensor_min (Tensor, optional): Minimum tensor value for quantization. Defaults to None.
        tensor_max (Tensor, optional): Maximum tensor value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = 2**bits - 1
    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max
    if isinstance(min_scale, torch.Tensor):
        wmin = wmin_tmp * min_scale
        wmax = wmax_tmp * max_scale
    else:
        wmin = wmin_tmp
        wmax = wmax_tmp

    wmax_new = torch.max(wmin.abs(), wmax)
    tmp = wmin < 0
    wmin_new = wmin.clone()  ##must clone, otherwise inplace backward will occur
    if torch.any(tmp):
        wmin_new[tmp] = -wmax_new[tmp]

    scale = ((wmax_new - wmin_new) / maxq).to(scale_dtype)
    scale = torch.clamp(scale, min=q_scale_thresh)
    scale = scale.unsqueeze(dim=-1)
    zp = torch.full_like(scale, (maxq + 1) / 2)

    int_w = round_ste(tensor / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    qdq_result = (scale * (q - zp)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, zp


def quant_tensor_asym_wo_round(
    tensor,
    bits=4,
    group_size=-1,
    v=0,
    min_scale=1.0,
    max_scale=1.0,
    scale_dtype=torch.float16,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh=1e-5,
    **kwargs
):
    """Quantize and de-quantize tensor asymmetrically without rounding, this is mainly for tuning bias, norm.

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for tensor
        max_scale: Maximum scale coefficient for tensor
        tensor_min (Tensor, optional): Minimum tensor value for quantization. Defaults to None.
        tensor_max (Tensor, optional): Maximum tensor value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantize tensor, scale, zero-point
    """
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = 2**bits - 1
    if tensor_min is None or tensor_max is None:
        wmin_tmp = torch.clamp(tensor.min(-1)[0], max=0)
        wmax_tmp = torch.clamp(tensor.max(-1)[0], min=0)
    else:
        wmin_tmp = tensor_min
        wmax_tmp = tensor_max
    if isinstance(min_scale, torch.Tensor):
        wmin = wmin_tmp * min_scale
        wmax = wmax_tmp * max_scale
    else:
        wmin = wmin_tmp
        wmax = wmax_tmp

    scale = ((wmax - wmin) / maxq).to(scale_dtype)
    scale = torch.clamp(scale, min=q_scale_thresh)
    zp = -wmin / scale  # pylint: disable=E1130
    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    int_w = tensor / scale + v
    q = torch.clamp(int_w + zp, 0, maxq)
    qdq_result = (scale * (q - zp)).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, zp
