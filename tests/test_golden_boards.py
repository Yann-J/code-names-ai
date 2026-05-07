from codenames_ai.eval.golden import evaluate_golden, golden_pass_rate
from codenames_ai.eval.golden_boards import default_golden_boards, iter_golden_cases


def test_iter_golden_cases_all_match():
    results = [evaluate_golden(spy, g) for g, spy in iter_golden_cases()]
    assert all(r.matched for r in results)
    assert golden_pass_rate(results) == 1.0


def test_default_golden_boards_tuple():
    gold, spy = default_golden_boards()
    assert len(gold) == 3
    r0 = evaluate_golden(spy, gold[0])
    assert r0.matched
