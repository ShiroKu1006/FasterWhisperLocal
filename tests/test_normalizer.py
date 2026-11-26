from faster_whisper_local.normalizer import TraditionalChineseNormalizer


def test_opencc_uses_taiwan_phrase_conversion() -> None:
    normalizer = TraditionalChineseNormalizer()

    assert normalizer.convert("软件与鼠标") == "軟體與滑鼠"
