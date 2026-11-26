from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .domain import OutputFormat, TranscriptionResult
from .formatters import format_srt_timestamp, format_txt_timestamp


def _render_txt(result: TranscriptionResult) -> str:
    return "".join(
        f"[{format_txt_timestamp(segment.start_seconds)} --> "
        f"{format_txt_timestamp(segment.end_seconds)}] {segment.text.strip()}\n"
        for segment in result.segments
    )


def _render_srt(result: TranscriptionResult) -> str:
    return "".join(
        f"{index}\n{format_srt_timestamp(segment.start_seconds)} --> "
        f"{format_srt_timestamp(segment.end_seconds)}\n{segment.text.strip()}\n\n"
        for index, segment in enumerate(result.segments, start=1)
    )


def _render_json(result: TranscriptionResult) -> str:
    payload = {
        "source": str(result.source),
        "language": result.language,
        "language_probability": result.language_probability,
        "model": result.model,
        "device": result.device,
        "compute_type": result.compute_type,
        "elapsed_seconds": result.elapsed_seconds,
        "segments": [
            {
                "start_seconds": segment.start_seconds,
                "end_seconds": segment.end_seconds,
                "text": segment.text,
            }
            for segment in result.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _atomic_write(target: Path, content: str) -> None:
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_file.write(content)
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, target)


def plan_output_paths(
    source: Path,
    output_directory: Path,
    formats: tuple[OutputFormat, ...],
    *,
    include_source_suffix: bool = False,
) -> tuple[Path, ...]:
    output_name = source.name if include_source_suffix else source.stem
    return tuple(
        output_directory / f"{output_name}.{output_format.value}"
        for output_format in formats
    )


def write_outputs(
    result: TranscriptionResult,
    output_directory: Path,
    formats: tuple[OutputFormat, ...],
    overwrite: bool,
    *,
    target_paths: tuple[Path, ...] | None = None,
) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    targets = target_paths or plan_output_paths(
        result.source, output_directory, formats
    )
    if len(targets) != len(formats):
        raise ValueError("輸出目標數量與格式數量不一致")
    conflicts = [target for target in targets if target.exists()]
    if conflicts and not overwrite:
        raise FileExistsError(f"輸出檔案已存在：{conflicts[0]}")

    renderers = {
        OutputFormat.TXT: _render_txt,
        OutputFormat.SRT: _render_srt,
        OutputFormat.JSON: _render_json,
    }
    for output_format, target in zip(formats, targets, strict=True):
        _atomic_write(target, renderers[output_format](result))
    return targets
