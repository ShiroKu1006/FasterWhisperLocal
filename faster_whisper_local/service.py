from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Protocol

from .domain import (
    ProgressEvent,
    TranscriptionOptions,
    TranscriptionResult,
    TranscriptionSegment,
)


@dataclass(frozen=True, slots=True)
class BackendTranscription:
    language: str
    language_probability: float
    segments: tuple[TranscriptionSegment, ...]


class TranscriptionBackend(Protocol):
    def transcribe(
        self, source: Path, options: TranscriptionOptions
    ) -> BackendTranscription: ...


class TextNormalizer(Protocol):
    def convert(self, text: str) -> str: ...


ProgressCallback = Callable[[ProgressEvent], None]


class TranscriptionService:
    def __init__(
        self,
        backend: TranscriptionBackend,
        normalizer: TextNormalizer,
    ) -> None:
        self._backend = backend
        self._normalizer = normalizer

    def transcribe(
        self,
        source: Path,
        options: TranscriptionOptions,
        progress_callback: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        if not source.is_file():
            raise FileNotFoundError(f"找不到輸入檔案：{source}")

        emit = progress_callback or (lambda _event: None)
        emit(ProgressEvent(stage="file_started", source=source))
        started_at = perf_counter()
        try:
            backend_result = self._backend.transcribe(source, options)
            segments = backend_result.segments
            if options.traditional and backend_result.language.lower().startswith("zh"):
                segments = tuple(
                    TranscriptionSegment(
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                        text=self._normalizer.convert(segment.text),
                    )
                    for segment in segments
                )
            result = TranscriptionResult(
                source=source,
                language=backend_result.language,
                language_probability=backend_result.language_probability,
                segments=segments,
                model=options.model,
                device=options.device,
                compute_type=options.compute_type,
                elapsed_seconds=perf_counter() - started_at,
            )
        except Exception as error:
            emit(ProgressEvent(stage="error", source=source, message=str(error)))
            raise
        emit(ProgressEvent(stage="file_completed", source=source, progress=1.0))
        return result
