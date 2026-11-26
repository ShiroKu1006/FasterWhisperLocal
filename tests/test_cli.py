from pathlib import Path

import pytest

from faster_whisper_local.cli import (
    build_parser,
    discover_inputs,
    main,
    parse_output_formats,
    run_batch,
)
from faster_whisper_local.domain import (
    OutputFormat,
    TranscriptionOptions,
    TranscriptionResult,
    TranscriptionSegment,
)


def test_discover_inputs_recurses_and_ignores_unsupported_files(tmp_path: Path) -> None:
    nested = tmp_path / "week1"
    nested.mkdir()
    (tmp_path / "lecture.mp4").touch()
    (nested / "discussion.wav").touch()
    (nested / "notes.md").touch()

    assert discover_inputs(tmp_path) == (
        tmp_path / "lecture.mp4",
        nested / "discussion.wav",
    )


def test_discover_inputs_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="沒有可轉錄的影音檔案"):
        discover_inputs(tmp_path)


def test_parse_output_formats_removes_duplicates_in_user_order() -> None:
    assert parse_output_formats(["json", "txt", "json"]) == (
        OutputFormat.JSON,
        OutputFormat.TXT,
    )


def test_parser_defaults_to_accuracy_first_large_v3_settings() -> None:
    arguments = build_parser().parse_args(["lecture.mp4"])

    assert arguments.device == "cuda"
    assert arguments.compute_type == "float16"
    assert arguments.beam_size == 5
    assert arguments.vad_filter is True
    assert arguments.traditional is True
    assert arguments.formats == ["txt"]


def test_main_reports_missing_large_v3_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "lecture.wav"
    source.touch()

    exit_code = main([str(source)], project_root=tmp_path)

    assert exit_code == 2
    assert "models\\faster-whisper-large-v3" in capsys.readouterr().err


def test_main_rejects_incomplete_large_v3_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "lecture.wav"
    source.touch()
    (tmp_path / "models" / "faster-whisper-large-v3").mkdir(parents=True)

    exit_code = main([str(source)], project_root=tmp_path)

    assert exit_code == 2
    assert "模型目錄不完整" in capsys.readouterr().err


def test_main_explains_legacy_transformers_model_is_not_ct2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "lecture.wav"
    source.touch()
    transformers_directory = tmp_path / "models" / "whisper-large-v3"
    transformers_directory.mkdir(parents=True)
    (transformers_directory / "config.json").write_text(
        '{"architectures": ["WhisperForConditionalGeneration"]}', encoding="utf-8"
    )
    (transformers_directory / "pytorch_model.bin").touch()

    exit_code = main([str(source)], project_root=tmp_path)

    assert exit_code == 2
    message = capsys.readouterr().err
    assert "不是 CTranslate2 格式" in message
    assert "models\\faster-whisper-large-v3" in message


class PartiallyFailingService:
    def transcribe(self, source: Path, options: TranscriptionOptions, progress_callback=None):
        if source.stem == "bad":
            raise RuntimeError("decode failed")
        return TranscriptionResult(
            source=source,
            language="en",
            language_probability=1.0,
            segments=(TranscriptionSegment(0.0, 1.0, "ok"),),
            model=options.model,
            device=options.device,
            compute_type=options.compute_type,
            elapsed_seconds=0.1,
        )


class RecordingService:
    def __init__(self) -> None:
        self.transcribed_sources: list[Path] = []

    def transcribe(
        self, source: Path, options: TranscriptionOptions, progress_callback=None
    ) -> TranscriptionResult:
        self.transcribed_sources.append(source)
        return TranscriptionResult(
            source=source,
            language="en",
            language_probability=1.0,
            segments=(TranscriptionSegment(0.0, 1.0, "ok"),),
            model=options.model,
            device=options.device,
            compute_type=options.compute_type,
            elapsed_seconds=0.1,
        )


def test_batch_continues_after_failure_and_preserves_relative_directories(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "input"
    nested = input_directory / "week1"
    nested.mkdir(parents=True)
    (nested / "good.wav").touch()
    (input_directory / "bad.wav").touch()

    summary = run_batch(
        input_path=input_directory,
        output_root=tmp_path / "output",
        options=TranscriptionOptions(),
        service=PartiallyFailingService(),
    )

    assert summary.succeeded == 1
    assert summary.failed == 1
    assert summary.skipped == 0
    assert (tmp_path / "output" / "week1" / "good.txt").exists()
    assert summary.exit_code == 1


def test_batch_disambiguates_duplicate_stems_with_source_extensions(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    (input_directory / "lecture.mp4").touch()
    (input_directory / "lecture.wav").touch()
    service = RecordingService()

    summary = run_batch(
        input_path=input_directory,
        output_root=tmp_path / "output",
        options=TranscriptionOptions(
            formats=(OutputFormat.TXT, OutputFormat.SRT, OutputFormat.JSON)
        ),
        service=service,
    )

    assert summary.succeeded == 2
    assert summary.skipped == 0
    for source_name in ("lecture.mp4", "lecture.wav"):
        for output_format in ("txt", "srt", "json"):
            assert (tmp_path / "output" / f"{source_name}.{output_format}").exists()
    assert not (tmp_path / "output" / "lecture.txt").exists()


def test_batch_skips_existing_output_before_calling_transcription_service(
    tmp_path: Path,
) -> None:
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    source = input_directory / "lesson.wav"
    source.touch()
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    (output_directory / "lesson.txt").write_text("existing", encoding="utf-8")
    service = RecordingService()

    summary = run_batch(
        input_path=input_directory,
        output_root=output_directory,
        options=TranscriptionOptions(overwrite=False),
        service=service,
    )

    assert summary.succeeded == 0
    assert summary.skipped == 1
    assert service.transcribed_sources == []
