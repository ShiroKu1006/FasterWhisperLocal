from __future__ import annotations

from opencc import OpenCC


class TraditionalChineseNormalizer:
    def __init__(self) -> None:
        self._converter = OpenCC("s2twp")

    def convert(self, text: str) -> str:
        return self._converter.convert(text)
