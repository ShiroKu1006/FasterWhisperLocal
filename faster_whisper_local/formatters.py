from __future__ import annotations


def _timestamp_parts(seconds: float) -> tuple[int, int, int, int]:
    total_milliseconds = round(max(0.0, seconds) * 1000)
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return hours, minutes, whole_seconds, milliseconds


def format_txt_timestamp(seconds: float) -> str:
    hours, minutes, whole_seconds, milliseconds = _timestamp_parts(seconds)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def format_srt_timestamp(seconds: float) -> str:
    hours, minutes, whole_seconds, milliseconds = _timestamp_parts(seconds)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"
