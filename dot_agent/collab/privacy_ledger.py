"""Local exact-match diagnostic over ordered cloud-bound payloads."""
from collections.abc import Mapping


class PrivacyLedger:
    def __init__(self, units: Mapping[str, str]):
        if any(not isinstance(v, str) or not v for v in units.values()):
            raise ValueError("each unit needs a nonempty literal")
        self._units = dict(units)
        self._tail = ""
        self._keep = max((len(v) - 1 for v in units.values()), default=0)
        self.exposed: set[str] = set()
        self.steps = 0

    def observe(self, cloud_text: str) -> frozenset[str]:
        """Return newly exposed unit IDs, including splits across adjacent calls."""
        window = self._tail + cloud_text
        hits = {key for key, value in self._units.items() if value in window}
        new = hits - self.exposed
        self.exposed.update(hits)
        self._tail = window[-self._keep:] if self._keep else ""
        self.steps += 1
        return frozenset(new)

    @property
    def exposure(self) -> float | None:
        return len(self.exposed) / len(self._units) if self._units else None
