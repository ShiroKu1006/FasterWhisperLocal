from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REQUIRED_MODEL_FILES = (
    "config.json",
    "model.bin",
    "tokenizer.json",
    "preprocessor_config.json",
)


def is_transformers_whisper_directory(model_directory: Path) -> bool:
    config_path = model_directory / "config.json"
    raw_weight_files = ("model.safetensors", "pytorch_model.bin")
    if not config_path.is_file() or not any(
        (model_directory / filename).is_file() for filename in raw_weight_files
    ):
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return "WhisperForConditionalGeneration" in config.get("architectures", [])


def validate_model_directory(model_directory: Path) -> Path:
    missing = [
        filename
        for filename in REQUIRED_MODEL_FILES
        if not (model_directory / filename).is_file()
    ]
    if missing:
        if is_transformers_whisper_directory(model_directory):
            raise RuntimeError(
                f"模型目錄不是 CTranslate2 格式，不能直接供 Faster-Whisper 使用："
                f"{model_directory}。請先轉換，或下載預轉換的 CTranslate2 模型。"
            )
        raise RuntimeError(
            f"large-v3 模型目錄不完整：{model_directory}，缺少 {', '.join(missing)}"
        )
    return model_directory


def download_large_v3(model_directory: Path) -> Path:
    if model_directory.exists() and any(model_directory.iterdir()):
        return validate_model_directory(model_directory)
    model_directory.mkdir(parents=True, exist_ok=True)
    from faster_whisper.utils import download_model

    downloaded_path = Path(
        download_model("large-v3", output_dir=str(model_directory))
    )
    return validate_model_directory(downloaded_path)


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="下載 Faster-Whisper large-v3 模型")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "models" / "faster-whisper-large-v3",
    )
    arguments = parser.parse_args(argv)
    try:
        path = download_large_v3(arguments.output)
    except Exception as error:
        print(f"模型下載失敗：{error}", file=sys.stderr)
        return 1
    print(f"large-v3 模型已就緒：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
