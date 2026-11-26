from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class OutputFormat(str, Enum):
    TXT = "txt"
    SRT = "srt"
    JSON = "json"


@dataclass(frozen=True, slots=True)
class TranscriptionOptions:
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    traditional: bool = True
    formats: tuple[OutputFormat, ...] = (OutputFormat.TXT,)
    overwrite: bool = False


@dataclass(frozen=True, slots=True)
class TranscriptionSegment:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    source: Path
    language: str
    language_probability: float
    segments: tuple[TranscriptionSegment, ...]
    model: str
    device: str
    compute_type: str
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: str
    source: Path | None = None
    message: str = ""
    progress: float | None = None
