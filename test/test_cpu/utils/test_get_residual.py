from types import SimpleNamespace

from transformers import GPT2Config, GPT2LMHeadModel

import get_residual


def test_run_normalizes_legacy_tied_weight_keys_before_save(tmp_path, monkeypatch):
    config = GPT2Config(n_layer=1, n_head=1, n_embd=16, n_positions=16, n_ctx=16, vocab_size=32)
    original_model = GPT2LMHeadModel(config)
    quantized_model = GPT2LMHeadModel(config)
    original_model._tied_weights_keys = ["lm_head.weight"]

    models = [original_model, quantized_model]

    def fake_load_model(_model_source):
        return models.pop(0)

    monkeypatch.setattr(get_residual, "load_model", fake_load_model)

    args = SimpleNamespace(
        input="original-model",
        quantized="quantized-model",
        output=str(tmp_path / "residual"),
        device="cpu",
        operation="sub",
    )

    get_residual.run(args)

    assert (tmp_path / "residual" / "config.json").exists()
    assert (tmp_path / "residual" / "residual_report.json").exists()
    assert original_model._tied_weights_keys == {"lm_head.weight": "lm_head.weight"}