from faster_whisper_local.domain import OutputFormat, TranscriptionOptions


def test_accuracy_first_options_are_the_defaults() -> None:
    options = TranscriptionOptions()

    assert options.model == "large-v3"
    assert options.device == "cuda"
    assert options.compute_type == "float16"
    assert options.beam_size == 5
    assert options.vad_filter is True
    assert options.traditional is True
    assert options.formats == (OutputFormat.TXT,)
