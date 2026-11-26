from pathlib import Path
from types import SimpleNamespace

from faster_whisper_local.backend import FasterWhisperBackend
from faster_whisper_local.domain import TranscriptionOptions


class FakeModel:
    def transcribe(self, source: str, **kwargs: object):
        assert source.endswith("lesson.wav")
        assert kwargs == {
            "language": None,
            "beam_size": 5,
            "vad_filter": True,
        }

        def segments():
            yield SimpleNamespace(start=0.5, end=1.5, text=" hello ")

        return segments(), SimpleNamespace(language="en", language_probability=0.9)


def test_backend_consumes_non_batched_segment_generator(tmp_path: Path) -> None:
    model_path = tmp_path / "models" / "large-v3"
    created_with: list[tuple[str, str, str]] = []

    def model_factory(path: str, device: str, compute_type: str) -> FakeModel:
        created_with.append((path, device, compute_type))
        return FakeModel()

    backend = FasterWhisperBackend(model_path, model_factory=model_factory)
    result = backend.transcribe(tmp_path / "lesson.wav", TranscriptionOptions())

    assert created_with == [(str(model_path), "cuda", "float16")]
    assert result.language == "en"
    assert result.segments[0].text == "hello"
