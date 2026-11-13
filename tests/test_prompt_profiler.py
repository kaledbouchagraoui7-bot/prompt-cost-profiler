import pytest
from prompt_profiler import TokenProfiler

def test_token_estimation():
    p = TokenProfiler(cost_per_1k=0.002)
    res = p.estimate("hello world")
    assert res['tokens'] > 0

def test_cost_calculation():
    p = TokenProfiler(cost_per_1k=1.0)
    res = p.estimate("a")
    assert res['cost'] > 0

def test_empty_input():
    p = TokenProfiler()
    assert p.estimate("")['tokens'] == 0

def test_high_cost_scaling():
    p = TokenProfiler(cost_per_1k=100.0)
    res = p.estimate("long string " * 100)
    assert res['cost'] > 1.0

def test_json_structure():
    p = TokenProfiler()
    res = p.estimate("test")
    assert isinstance(res, dict)
    assert 'tokens' in res and 'cost' in res