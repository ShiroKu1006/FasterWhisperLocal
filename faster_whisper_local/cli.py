from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .domain import OutputFormat, TranscriptionOptions
from .outputs import plan_output_paths, write_outputs


SUPPORTED_EXTENSIONS = frozenset(
    {
        ".aac",
        ".avi",
        ".flac",
        ".m4a",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".ogg",
        ".opus",
        ".wav",
        ".webm",
        ".wma",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 Faster-Whisper large-v3 在本機轉錄影音檔案。"
    )
    parser.add_argument("input", type=Path, help="單一影音檔或資料夾")
    parser.add_argument("--output", type=Path, help="輸出根目錄")
    parser.add_argument(
        "--format",
        dest="formats",
        nargs="+",
        choices=[output_format.value for output_format in OutputFormat],
        default=[OutputFormat.TXT.value],
        help="輸出格式，可同時指定多種（預設：txt）",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--compute-type",
        choices=("float16", "int8_float16", "int8"),
        default="float16",
    )
    parser.add_argument("--language", help="指定語言代碼；省略時自動辨識")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--no-vad", dest="vad_filter", action="store_false", default=True
    )
    parser.add_argument(
        "--no-traditional", dest="traditional", action="store_false", default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def discover_inputs(input_path: Path) -> tuple[Path, ...]:
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支援的影音格式：{input_path.suffix}")
        return (input_path,)
    if not input_path.is_dir():
        raise FileNotFoundError(f"找不到輸入路徑：{input_path}")

    inputs = tuple(
        sorted(
            (
                path
                for path in input_path.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ),
            key=lambda path: str(path.relative_to(input_path)).casefold(),
        )
    )
    if not inputs:
        raise ValueError(f"資料夾內沒有可轉錄的影音檔案：{input_path}")
    return inputs


def parse_output_formats(values: Iterable[str]) -> tuple[OutputFormat, ...]:
    formats: list[OutputFormat] = []
    for value in values:
        output_format = OutputFormat(value.lower())
        if output_format not in formats:
            formats.append(output_format)
    return tuple(formats)


@dataclass(frozen=True, slots=True)
class BatchSummary:
    succeeded: int
    skipped: int
    failed: int
    messages: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


def run_batch(
    input_path: Path,
    output_root: Path,
    options: TranscriptionOptions,
    service: Any,
) -> BatchSummary:
    inputs = discover_inputs(input_path)
    output_directories = {
        source: output_root
        / (source.relative_to(input_path).parent if input_path.is_dir() else Path())
        for source in inputs
    }
    stem_counts = Counter(
        (str(output_directories[source]).casefold(), source.stem.casefold())
        for source in inputs
    )
    planned_targets = {
        source: plan_output_paths(
            source,
            output_directories[source],
            options.formats,
            include_source_suffix=stem_counts[
                (str(output_directories[source]).casefold(), source.stem.casefold())
            ]
            > 1,
        )
        for source in inputs
    }
    succeeded = 0
    skipped = 0
    failed = 0
    messages: list[str] = []
    for source in inputs:
        output_directory = output_directories[source]
        targets = planned_targets[source]
        if not options.overwrite:
            conflict = next((target for target in targets if target.exists()), None)
            if conflict is not None:
                skipped += 1
                messages.append(f"跳過：輸出檔案已存在：{conflict}")
                continue
        try:
            result = service.transcribe(source, options)
            write_outputs(
                result,
                output_directory=output_directory,
                formats=options.formats,
                overwrite=options.overwrite,
                target_paths=targets,
            )
            succeeded += 1
            messages.append(f"完成：{source}")
        except FileExistsError as error:
            skipped += 1
            messages.append(f"跳過：{error}")
        except Exception as error:
            failed += 1
            messages.append(f"失敗：{source}（{error}）")
    return BatchSummary(
        succeeded=succeeded,
        skipped=skipped,
        failed=failed,
        messages=tuple(messages),
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    project_root: Path | None = None,
) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.device == "cpu" and arguments.compute_type != "int8":
        parser.error("CPU 模式請同時指定 --compute-type int8")
    if arguments.beam_size < 1:
        parser.error("--beam-size 必須大於 0")

    root = project_root or Path(__file__).resolve().parent.parent
    model_path = root / "models" / "faster-whisper-large-v3"
    legacy_transformers_path = root / "models" / "whisper-large-v3"
    if not model_path.is_dir():
        from .model_manager import is_transformers_whisper_directory

        if is_transformers_whisper_directory(legacy_transformers_path):
            print(
                f"偵測到 {legacy_transformers_path} 不是 CTranslate2 格式，"
                "不能直接供 Faster-Whisper 使用。請先轉換，或下載預轉換的 "
                f"CTranslate2 模型到：{model_path}",
                file=sys.stderr,
            )
            return 2
        print(
            f"找不到 large-v3 模型：{model_path}\n"
            "請先執行 download_model.ps1，或自行將模型放入該目錄。",
            file=sys.stderr,
        )
        return 2

    try:
        from .model_manager import validate_model_directory

        validate_model_directory(model_path)
        if arguments.device == "cuda":
            from .runtime import register_cuda_runtime

            register_cuda_runtime(root / "runtime" / "cuda12")
            import ctranslate2

            if ctranslate2.get_cuda_device_count() < 1:
                raise RuntimeError("CTranslate2 未偵測到可用的 NVIDIA CUDA 裝置")

        from .backend import FasterWhisperBackend
        from .normalizer import TraditionalChineseNormalizer
        from .service import TranscriptionService

        options = TranscriptionOptions(
            device=arguments.device,
            compute_type=arguments.compute_type,
            language=arguments.language,
            beam_size=arguments.beam_size,
            vad_filter=arguments.vad_filter,
            traditional=arguments.traditional,
            formats=parse_output_formats(arguments.formats),
            overwrite=arguments.overwrite,
        )
        service = TranscriptionService(
            FasterWhisperBackend(model_path), TraditionalChineseNormalizer()
        )
        output_root = arguments.output or root / "output"
        summary = run_batch(arguments.input, output_root, options, service)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"無法開始轉錄：{error}", file=sys.stderr)
        return 2

    for message in summary.messages:
        print(message)
    print(
        f"摘要：成功 {summary.succeeded}、跳過 {summary.skipped}、失敗 {summary.failed}"
    )
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
