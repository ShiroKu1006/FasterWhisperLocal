from pathlib import Path

import pytest

from faster_whisper_local.model_manager import validate_model_directory


def test_model_validation_lists_missing_required_files(tmp_path: Path) -> None:
    model_directory = tmp_path / "large-v3"
    model_directory.mkdir()
    (model_directory / "config.json").touch()

    with pytest.raises(RuntimeError) as error:
        validate_model_directory(model_directory)

    message = str(error.value)
    assert "model.bin" in message
    assert "tokenizer.json" in message


def test_model_validation_requires_preprocessor_configuration(tmp_path: Path) -> None:
    model_directory = tmp_path / "faster-whisper-large-v3"
    model_directory.mkdir()
    for filename in ("config.json", "model.bin", "tokenizer.json"):
        (model_directory / filename).touch()

    with pytest.raises(RuntimeError) as error:
        validate_model_directory(model_directory)

    assert "preprocessor_config.json" in str(error.value)


def test_model_validation_identifies_transformers_weights_as_not_ct2(
    tmp_path: Path,
) -> None:
    model_directory = tmp_path / "whisper-large-v3"
    model_directory.mkdir()
    (model_directory / "config.json").write_text(
        '{"architectures": ["WhisperForConditionalGeneration"]}', encoding="utf-8"
    )
    (model_directory / "model.safetensors").touch()

    with pytest.raises(RuntimeError) as error:
        validate_model_directory(model_directory)

    message = str(error.value)
    assert "不是 CTranslate2 格式" in message
    assert "Faster-Whisper" in message
