from pathlib import Path

import json
import pytest

from faster_whisper_local.domain import (
    OutputFormat,
    TranscriptionResult,
    TranscriptionSegment,
)
from faster_whisper_local.outputs import write_outputs


def make_result(source: Path) -> TranscriptionResult:
    return TranscriptionResult(
        source=source,
        language="zh",
        language_probability=0.99,
        segments=(TranscriptionSegment(1.25, 4.8, "測試文字"),),
        model="large-v3",
        device="cuda",
        compute_type="float16",
        elapsed_seconds=2.5,
    )


def test_default_txt_output_contains_segment_timestamps(tmp_path: Path) -> None:
    source = tmp_path / "lesson.mp4"
    source.touch()

    written = write_outputs(
        make_result(source),
        output_directory=tmp_path / "output",
        formats=(OutputFormat.TXT,),
        overwrite=False,
    )

    assert written == (tmp_path / "output" / "lesson.txt",)
    assert written[0].read_text(encoding="utf-8") == (
        "[00:00:01.250 --> 00:00:04.800] 測試文字\n"
    )


def test_srt_and_json_outputs_preserve_transcription_metadata(tmp_path: Path) -> None:
    source = tmp_path / "lesson.mp4"
    source.touch()

    written = write_outputs(
        make_result(source),
        output_directory=tmp_path / "output",
        formats=(OutputFormat.SRT, OutputFormat.JSON),
        overwrite=False,
    )

    assert written[0].read_text(encoding="utf-8") == (
        "1\n00:00:01,250 --> 00:00:04,800\n測試文字\n\n"
    )
    metadata = json.loads(written[1].read_text(encoding="utf-8"))
    assert metadata["model"] == "large-v3"
    assert metadata["segments"] == [
        {"start_seconds": 1.25, "end_seconds": 4.8, "text": "測試文字"}
    ]


def test_output_collision_is_checked_before_any_file_is_written(tmp_path: Path) -> None:
    source = tmp_path / "lesson.mp4"
    source.touch()
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    (output_directory / "lesson.srt").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_outputs(
            make_result(source),
            output_directory=output_directory,
            formats=(OutputFormat.TXT, OutputFormat.SRT),
            overwrite=False,
        )

    assert not (output_directory / "lesson.txt").exists()
