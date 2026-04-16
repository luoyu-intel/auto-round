import copy
import shutil
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.data_type.int import search_scales_zp

from ...helpers import get_model_path, model_infer


class TestAutoRoundAsym:
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
