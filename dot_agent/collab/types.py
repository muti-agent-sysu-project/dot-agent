from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    LOCAL = "Local-Exec"
    PROXY = "Minimal-Proxy-Cloud"
    ADVISE = "Local-Compute-Cloud-Advise"
    FULL = "Constrained-Full-Cloud"


@dataclass(frozen=True)
class Span:
    """Half-open Python character offsets in a locally annotated string."""
    start: int
    end: int
    kind: str
