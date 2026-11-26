from faster_whisper_local.formatters import format_srt_timestamp, format_txt_timestamp


def test_format_txt_timestamp_handles_hour_boundary_and_rounding() -> None:
    assert format_txt_timestamp(3661.9996) == "01:01:02.000"


def test_format_srt_timestamp_uses_comma_milliseconds() -> None:
    assert format_srt_timestamp(61.25) == "00:01:01,250"
