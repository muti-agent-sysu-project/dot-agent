"""A declarative, non-executing skill with local state assertions."""
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    steps: tuple[str, ...]
    requires: tuple[tuple[str, str], ...]
    ensures: tuple[tuple[str, str], ...]

    def can_run(self, state: Mapping[str, str]) -> bool:
        return all(state.get(key) == value for key, value in self.requires)

    def verified(self, state: Mapping[str, str]) -> bool:
        return all(state.get(key) == value for key, value in self.ensures)


def compile_skill(steps: list[str], requires: Mapping[str, str],
                  ensures: Mapping[str, str]) -> Skill:
    if not steps or any(not step.strip() for step in steps) or not ensures:
        raise ValueError("steps and postconditions required")
    return Skill(tuple(steps), tuple(sorted(requires.items())), tuple(sorted(ensures.items())))
