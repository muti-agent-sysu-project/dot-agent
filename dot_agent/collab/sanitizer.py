"""One local registry per trajectory. The caller supplies sensitive spans."""
import re

from .types import Span

_TOKEN = re.compile(r"\[\[EC2:([A-Z][A-Z0-9_]*):([1-9][0-9]*)\]\]")


# Narrow defaults for unannotated input; callers can supply reviewed spans instead.
_DIRECT = re.compile(
    r"(?P<EMAIL>[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})|(?P<PHONE>(?<!\d)1[3-9]\d{9}(?!\d))"
)


class Registry:
    def __init__(self):
        self._tokens: dict[tuple[str, str], str] = {}
        self._originals: dict[str, str] = {}

    def mask(self, text: str, spans: list[Span] | None = None) -> str:
        if "[[EC2:" in text:
            raise ValueError("reserved placeholder prefix")
        if spans is None:
            spans = [Span(m.start(), m.end(), m.lastgroup) for m in _DIRECT.finditer(text)]
        ordered = sorted(spans, key=lambda span: span.start)
        end = 0
        for span in ordered:
            if not end <= span.start < span.end <= len(text):
                raise ValueError("overlapping or invalid span")
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", span.kind):
                raise ValueError("invalid entity type")
            end = span.end

        pieces = []
        end = 0
        for span in ordered:
            value = text[span.start:span.end]
            key = (span.kind, value)
            if key not in self._tokens:
                token = f"[[EC2:{span.kind}:{len(self._tokens) + 1}]]"
                self._tokens[key] = token
                self._originals[token] = value
            pieces.extend((text[end:span.start], self._tokens[key]))
            end = span.end
        return "".join(pieces) + text[end:]

    def restore(self, text: str) -> str:
        def original(match):
            if match[0] not in self._originals:
                raise ValueError("unknown placeholder")
            return self._originals[match[0]]
        return _TOKEN.sub(original, text)
