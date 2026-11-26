import pytest

from eval import verify_theorems as theory


@pytest.mark.parametrize(("suite", "expected"), [
    (theory.verify_propositions, 42),
    (theory.verify_closure_separation, 128),
    (theory.verify_stream_separation, 173),
    (theory.verify_phase, 1413),
    (theory.verify_simulation, 15),
    (theory.verify_audit, 283),
    (theory.verify_linkage, 806),
])
def test_theory_suite(suite, expected):
    count = 0

    def check(condition, message):
        nonlocal count
        assert condition, message
        count += 1

    suite(check)
    assert count == expected
