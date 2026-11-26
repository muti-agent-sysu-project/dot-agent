import pytest

from eval import collab_experiment_runner as r


@pytest.mark.parametrize(("expression", "expected"), [
    ("(1234.5 - 1100) / 1100", 134.5 / 1100), ("-2 * (+3)", -6)])
def test_expression(expression, expected):
    assert r.arithmetic(expression) == expected


@pytest.mark.parametrize("expr", ["foo()", "2 ** 8", "1/0", "1e500", "1//2", "a.b", "1+1"*200])
def test_bad_expression_is_never_replaced_by_correct_answer(expr):
    assert r.resolve_output(r.encode(dict(expression=expr, unit="number", answer=42)).decode()) == ""


def test_unit_and_nonnumeric_contract():
    assert r.answer(r.resolve_output('{"expression":"100*(9-8)/8","unit":"percent","answer":0}'),
                    .125, percent_scale=.01)
    assert r.resolve_output('{"expression":null,"unit":"number","answer":"undefined"}') == "undefined"

