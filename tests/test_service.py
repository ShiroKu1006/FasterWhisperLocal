from pathlib import Path

import pytest

from faster_whisper_local.domain import TranscriptionOptions, TranscriptionSegment
from faster_whisper_local.service import BackendTranscription, TranscriptionService


class FakeBackend:
    def transcribe(
        self, source: Path, options: TranscriptionOptions
    ) -> BackendTranscription:
        return BackendTranscription(
            language="zh",
            language_probability=0.98,
            segments=(TranscriptionSegment(0.0, 1.0, "软件"),),
        )


class FakeNormalizer:
    def convert(self, text: str) -> str:
        return text.replace("软件", "軟體")


class FailingBackend:
    def transcribe(
        self, source: Path, options: TranscriptionOptions
    ) -> BackendTranscription:
        raise RuntimeError("decode failed")


def test_service_normalizes_chinese_and_reports_progress(tmp_path: Path) -> None:
    source = tmp_path / "lesson.wav"
    source.touch()
    stages: list[str] = []
    service = TranscriptionService(FakeBackend(), FakeNormalizer())

    result = service.transcribe(
        source,
        TranscriptionOptions(),
        lambda event: stages.append(event.stage),
    )

    assert result.language == "zh"
    assert result.segments[0].text == "軟體"
    assert result.model == "large-v3"
    assert result.compute_type == "float16"
    assert stages == ["file_started", "file_completed"]


def test_service_reports_error_event_when_backend_fails(tmp_path: Path) -> None:
    source = tmp_path / "broken.wav"
    source.touch()
    stages: list[str] = []
    service = TranscriptionService(FailingBackend(), FakeNormalizer())

    with pytest.raises(RuntimeError, match="decode failed"):
        service.transcribe(
            source,
            TranscriptionOptions(),
            lambda event: stages.append(event.stage),
        )

    assert stages == ["file_started", "error"]
