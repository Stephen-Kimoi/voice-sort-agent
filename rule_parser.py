"""Turns a transcribed spoken sort rule into a structured bin assignment.

Deliberately deterministic (keyword + number matching), not an LLM call: the
vocabulary a sorting-line operator uses is small and fixed ("defective",
"anomalous", "bad", "good", "normal", bin numbers), so a regex-based parser is
both cheaper and more predictable than routing a five-word command through a
second model. Swap this for an LLM parser only if the vocabulary needs to grow
past what a fixed keyword list can cover.
"""

import re
from dataclasses import dataclass

_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

_ANOMALOUS_KEYWORDS = ("defect", "anomal", "bad", "damaged", "reject", "crack", "chip", "scratch")
_NORMAL_KEYWORDS = ("normal", "good", "clean", "pass")


@dataclass
class SortRule:
    normal_bin: int
    anomalous_bin: int
    raw_transcript: str

    def bin_for(self, is_anomalous: bool) -> int:
        return self.anomalous_bin if is_anomalous else self.normal_bin


def _number_after_keyword(text: str, keywords: tuple[str, ...]) -> int | None:
    for keyword in keywords:
        match = re.search(rf"{keyword}\w*[^.]*?\bbin\s+(\w+)", text)
        if not match:
            continue
        token = match.group(1)
        if token.isdigit():
            return int(token)
        if token in _WORD_TO_NUM:
            return _WORD_TO_NUM[token]
    return None


def parse_rule(transcript: str, default_normal_bin: int = 1, default_anomalous_bin: int = 2) -> SortRule:
    text = transcript.lower()

    anomalous_bin = _number_after_keyword(text, _ANOMALOUS_KEYWORDS)
    normal_bin = _number_after_keyword(text, _NORMAL_KEYWORDS)

    if anomalous_bin is None:
        anomalous_bin = default_anomalous_bin
    if normal_bin is None:
        normal_bin = default_normal_bin

    return SortRule(normal_bin=normal_bin, anomalous_bin=anomalous_bin, raw_transcript=transcript)
