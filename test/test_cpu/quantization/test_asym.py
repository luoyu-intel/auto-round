import copy
import shutil
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.data_type.int import dynamic_quantize_tensor, search_scales_zp

from ...helpers import get_model_path, model_infer


class TestAutoRoundAsym:
    @staticmethod
    def _legacy_weighted_one_dim_search(data, weights, bits=4, iters=100):
        a_min = data.min(-1, keepdim=True)[0].clamp(max=0)
        a_max = data.max(-1, keepdim=True)[0].clamp(min=0)
        maxq = (1 << bits) - 1
        fullq = 1 << (bits - 1)
        denorm = a_max - a_min

        def asym_quant_iter(search_q):
            inverse_scale = search_q / denorm
            inverse_scale[denorm.abs() <= 1e-4] = 1
            zp = torch.round(-a_min * inverse_scale).clamp(0, maxq)
            q = torch.clamp(torch.round(data * inverse_scale + zp), 0, maxq)
            scale = 1 / inverse_scale
            centered_q = q - fullq
            centered_zp = zp - fullq
            loss = torch.sum(weights * (scale * (centered_q - centered_zp) - data).pow(2), dim=-1, keepdim=True)
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

    @pytest.fixture(autouse=True)
    def setup_save_folder(self, tmp_path):
        self.save_folder = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_folder, ignore_errors=True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree("runs", ignore_errors=True)

    def test_asym_group_size(self, tiny_opt_model_path):
        for group_size in [32, 64, 128]:
            bits, sym = 4, False
            ar = AutoRound(
                tiny_opt_model_path, bits=bits, group_size=group_size, sym=sym, iters=0, seqlen=2, nsamples=1
            )
            ar.quantize_and_save(format="auto_round", output_dir=self.save_folder)

            model = AutoModelForCausalLM.from_pretrained(
                self.save_folder,
                torch_dtype="auto",
                device_map="auto",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
            model_infer(model, tokenizer)

    def test_asym_bits(self, tiny_opt_model_path):
        for bits in [2, 8]:
            group_size, sym = 128, False
            ar = AutoRound(
                tiny_opt_model_path, bits=bits, group_size=group_size, sym=sym, iters=0, seqlen=2, nsamples=1
            )
            ar.quantize_and_save(format="auto_round", output_dir=self.save_folder)

            model = AutoModelForCausalLM.from_pretrained(
                self.save_folder,
                torch_dtype="auto",
                device_map="auto",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
            model_infer(model, tokenizer)

    # use parameters later
    def test_asym_format(self, tiny_opt_model_path):
        for format in ["auto_round", "auto_round:auto_gptq", "auto_round:gptqmodel"]:
            bits, group_size, sym = 4, 128, False
            ar = AutoRound(
                tiny_opt_model_path,
                bits=bits,
                group_size=group_size,
                sym=sym,
                iters=0,
                seqlen=2,
                nsamples=1,
                disable_opt_rtn=True,
            )
            ar.quantize_and_save(format=format, output_dir=self.save_folder)

            model = AutoModelForCausalLM.from_pretrained(
                self.save_folder,
                torch_dtype="auto",
                device_map="auto",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
            model_infer(model, tokenizer)

    def test_search_scales_zp(self):
        data = torch.tensor(
            [[-1.2, -0.4, 0.3, 1.7], [-0.1, 0.0, 0.8, 2.1]],
            dtype=torch.float32,
        )
        weights = torch.tensor(
            [[1.0, 2.0, 0.5, 1.5], [0.1, 1.0, 3.0, 2.0]],
            dtype=torch.float32,
        )
        bits = 4
        maxq = 2**bits - 1

        scale, zp = search_scales_zp(data, bits, qw=weights)

        baseline_min = torch.min(data, dim=-1, keepdim=True)[0]
        baseline_max = torch.max(data, dim=-1, keepdim=True)[0]
        baseline_scale = torch.clamp((baseline_max - baseline_min) / maxq, min=1e-5)
        baseline_zp = torch.clamp(torch.round(-baseline_min / baseline_scale), 0, maxq)

        q = torch.clamp(torch.round(data / scale + zp), 0, maxq)
        baseline_q = torch.clamp(torch.round(data / baseline_scale + baseline_zp), 0, maxq)
        loss = torch.sum(weights * (scale * (q - zp) - data) ** 2, dim=-1)
        baseline_loss = torch.sum(weights * (baseline_scale * (baseline_q - baseline_zp) - data) ** 2, dim=-1)

        assert scale.shape == (data.shape[0], 1)
        assert zp.shape == (data.shape[0], 1)
        assert torch.all(zp >= 0)
        assert torch.all(zp <= maxq)
        assert torch.allclose(zp, torch.round(zp))
        assert torch.all(loss <= baseline_loss + 1e-6)

    def test_t_dyn_quant_not_worse_than_baseline_mse(self):
        data = torch.tensor(
            [
                [-1.2, -0.4, 0.3, 1.7, -0.6, 0.2, 0.9, 1.1],
                [-0.1, 0.0, 0.8, 2.1, -0.7, 0.4, 1.3, 1.8],
                [10.551, 16.721, -11.184, 0.631, 4.470, 6.765, -20.073, 6.353],
                [-10.918, -28.377, -6.493, 4.112, 5.383, -14.386, -14.023, 8.338],
            ],
            dtype=torch.float32,
        )
        weights = torch.tensor(
            [
                [1.0, 2.0, 0.5, 1.5, 1.0, 1.0, 1.0, 1.0],
                [0.1, 1.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                [1.2, 0.8, 2.0, 1.0, 1.5, 0.5, 2.5, 1.0],
                [0.7, 1.4, 1.2, 0.9, 1.1, 2.0, 1.6, 0.8],
            ],
            dtype=torch.float32,
        )
        bits = 4
        maxq = 2**bits - 1

        q, scale, zp = dynamic_quantize_tensor(data, bits=bits, dir=-1, asym=True, iter=100, qw=weights)
        loss = torch.sum(weights * (scale * (q - zp) - data).pow(2), dim=-1)

        baseline_min = torch.min(data, dim=-1, keepdim=True)[0]
        baseline_max = torch.max(data, dim=-1, keepdim=True)[0]
        baseline_scale = torch.clamp((baseline_max - baseline_min) / maxq, min=1e-5)
        baseline_zp = torch.clamp(torch.round(-baseline_min / baseline_scale), 0, maxq)
        baseline_q = torch.clamp(torch.round(data / baseline_scale + baseline_zp), 0, maxq)
        baseline_loss = torch.sum(weights * (baseline_scale * (baseline_q - baseline_zp) - data).pow(2), dim=-1)

        assert torch.all(loss <= baseline_loss + 1e-6)

    def test_t_dyn_quant_unweighted_refine_improves_one_dim_search(self):
        data = torch.tensor(
            [
                [-1.6407, 0.2948, -1.2780, 0.9452, 1.5191, 0.5372, 2.1224, 1.5235],
                [-0.1069, 2.5773, -0.6690, 0.3575, 1.7686, 7.1066, -4.2067, -4.5601],
                [-1.8193, 2.8185, 3.3661, 0.7335, -0.4910, -0.9753, 1.8299, -0.9855],
                [-1.1387, 2.4321, 4.7629, 10.4315, -4.3936, -3.5024, 5.6592, -1.4546],
            ],
            dtype=torch.float32,
        )

        refined_q, refined_scale, refined_zp = dynamic_quantize_tensor(data, bits=4, dir=-1, asym=True, iter=100, qw=None)
        oned_q, oned_scale, oned_zp = self._legacy_weighted_one_dim_search(
            data,
            torch.ones_like(data),
            bits=4,
            iters=100,
        )

        refined_loss = torch.sum((refined_scale * (refined_q - refined_zp) - data).pow(2), dim=-1)
        oned_loss = torch.sum((oned_scale * (oned_q - oned_zp) - data).pow(2), dim=-1)

        assert torch.all(refined_loss <= oned_loss + 1e-6)
        assert torch.any(refined_loss < oned_loss - 1e-6)

    def test_t_dyn_quant_weighted_refine_improves_one_dim_search(self):
        data = torch.tensor(
            [
                [-1.9604, 1.6869, 3.1967, -3.6652, -0.0010, 1.3554, -1.4233, -2.1810],
                [3.2796, -1.1065, -2.9792, -0.8805, 2.6256, 3.5383, 1.1912, 3.7705],
                [-4.2480, -0.4737, 6.2791, -0.4900, 1.0953, 2.8462, 3.6173, -1.6735],
                [-0.1280, -0.1460, -4.9472, -0.5357, -5.6844, 2.5223, -3.4700, -3.5793],
            ],
            dtype=torch.float32,
        )
        weights = torch.tensor(
            [
                [1.3533, 1.2381, 1.5875, 2.0184, 0.8775, 0.5429, 0.8484, 0.4905],
                [1.5810, 0.6058, 0.5663, 1.9628, 2.0151, 1.2150, 0.9268, 0.9709],
                [1.5737, 0.1662, 0.2828, 1.8988, 2.0872, 1.0406, 0.3098, 1.1273],
                [0.6348, 1.0981, 1.5895, 1.5427, 0.9827, 1.2101, 1.3721, 0.3162],
            ],
            dtype=torch.float32,
        )

        refined_q, refined_scale, refined_zp = dynamic_quantize_tensor(data, bits=4, dir=-1, asym=True, iter=100, qw=weights)
        oned_q, oned_scale, oned_zp = self._legacy_weighted_one_dim_search(data, weights, bits=4, iters=100)

        refined_loss = torch.sum(weights * (refined_scale * (refined_q - refined_zp) - data).pow(2), dim=-1)
        oned_loss = torch.sum(weights * (oned_scale * (oned_q - oned_zp) - data).pow(2), dim=-1)

        assert torch.all(refined_loss <= oned_loss + 1e-6)
        assert torch.any(refined_loss < oned_loss - 1e-6)
