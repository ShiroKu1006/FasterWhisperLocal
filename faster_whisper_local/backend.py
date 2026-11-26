from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .domain import TranscriptionOptions, TranscriptionSegment
from .service import BackendTranscription


ModelFactory = Callable[[str, str, str], Any]


def _default_model_factory(path: str, device: str, compute_type: str) -> Any:
    from faster_whisper import WhisperModel

    return WhisperModel(
        path,
        device=device,
        compute_type=compute_type,
        num_workers=1,
    )


class FasterWhisperBackend:
    def __init__(
        self,
        model_path: Path,
        model_factory: ModelFactory = _default_model_factory,
    ) -> None:
        self._model_path = model_path
        self._model_factory = model_factory
        self._model: Any | None = None
        self._model_key: tuple[str, str] | None = None

    def transcribe(
        self, source: Path, options: TranscriptionOptions
    ) -> BackendTranscription:
        model_key = (options.device, options.compute_type)
        if self._model is None or self._model_key != model_key:
            self._model = self._model_factory(
                str(self._model_path), options.device, options.compute_type
            )
            self._model_key = model_key

        segments_generator, info = self._model.transcribe(
            str(source),
            language=options.language,
            beam_size=options.beam_size,
            vad_filter=options.vad_filter,
        )
        segments = tuple(
            TranscriptionSegment(
                start_seconds=float(segment.start),
                end_seconds=float(segment.end),
                text=str(segment.text).strip(),
            )
            for segment in segments_generator
        )
        return BackendTranscription(
            language=str(info.language),
            language_probability=float(info.language_probability),
            segments=segments,
        )
