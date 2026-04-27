import json
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
    monkeypatch.setattr(get_residual, "try_save_tokenizer", lambda *args, **kwargs: None)

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


def test_try_save_tokenizer_copies_local_artifacts(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "residual"
    source_dir.mkdir()
    output_dir.mkdir()

    expected_files = {
        "tokenizer.json": "tokenizer-json",
        "tokenizer_config.json": json.dumps({"tokenizer_class": "Qwen2Tokenizer"}),
        "special_tokens_map.json": json.dumps({"eos_token": "<|im_end|>"}),
        "chat_template.jinja": "template",
    }
    for file_name, content in expected_files.items():
        (source_dir / file_name).write_text(content, encoding="utf-8")
    (source_dir / "config.json").write_text("model-config", encoding="utf-8")

    get_residual.try_save_tokenizer(str(source_dir), str(output_dir))

    for file_name, content in expected_files.items():
        assert (output_dir / file_name).read_text(encoding="utf-8") == content
    assert not (output_dir / "config.json").exists()


def test_try_save_tokenizer_copies_remote_artifacts(tmp_path, monkeypatch):
    output_dir = tmp_path / "residual"
    output_dir.mkdir()
    downloaded_dir = tmp_path / "downloaded"
    downloaded_dir.mkdir()
    source_file = downloaded_dir / "tokenizer_config.json"
    source_file.write_text('{"tokenizer_class": "Qwen2Tokenizer"}', encoding="utf-8")

    monkeypatch.setattr(
        get_residual,
        "list_remote_tokenizer_artifacts",
        lambda model_source: ["tokenizer_config.json"],
    )
    monkeypatch.setattr(
        get_residual,
        "copy_remote_tokenizer_artifacts",
        lambda model_source, target_dir, file_names: __import__("shutil").copy2(
            source_file, output_dir / "tokenizer_config.json"
        ),
    )

    get_residual.try_save_tokenizer("Qwen/Qwen3-8B", str(output_dir))

    assert (output_dir / "tokenizer_config.json").read_text(encoding="utf-8") == source_file.read_text(encoding="utf-8")