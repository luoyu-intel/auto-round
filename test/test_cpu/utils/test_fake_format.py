from types import SimpleNamespace

from transformers import GPT2Config, GPT2LMHeadModel

from auto_round.formats import FakeFormat


def test_fake_format_normalizes_legacy_tied_weight_keys_before_save(tmp_path):
    model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=16, n_positions=16, n_ctx=16, vocab_size=32))
    model._tied_weights_keys = ["lm_head.weight"]

    fake_format = FakeFormat("fake", SimpleNamespace(scheme=None))
    fake_format.save_quantized(output_dir=str(tmp_path), model=model)

    assert (tmp_path / "config.json").exists()
    assert model._tied_weights_keys == {"lm_head.weight": "lm_head.weight"}