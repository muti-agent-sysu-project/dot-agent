"""Deterministic foundation policy; eligibility is assessed locally."""
from .types import Action


def choose_action(*, local_ok: bool, proxy_sufficient: bool,
                  advice_sufficient: bool, full_allowed: bool) -> Action:
    if local_ok:
        return Action.LOCAL
    if proxy_sufficient:
        return Action.PROXY
    if advice_sufficient:
        return Action.ADVISE
    if full_allowed:
        return Action.FULL
    return Action.LOCAL
