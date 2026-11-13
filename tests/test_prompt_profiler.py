import pytest
import json
import subprocess
import sys
from pathlib import Path
from prompt_profiler import TokenProfiler, compact_prompt, recommend_model_route


def test_token_estimation():
    p = TokenProfiler(cost_per_1k=0.002)
    res = p.estimate("hello world")
    assert res["tokens"] > 0


def test_cost_calculation():
    p = TokenProfiler(cost_per_1k=1.0)
    res = p.estimate("a")
    assert res["cost"] > 0


def test_empty_input():
    p = TokenProfiler()
    assert p.estimate("")["tokens"] == 0


def test_high_cost_scaling():
    p = TokenProfiler(cost_per_1k=100.0)
    res = p.estimate("long string " * 100)
    assert res["cost"] > 1.0


def test_json_structure():
    p = TokenProfiler()
    res = p.estimate("test")
    assert isinstance(res, dict)
    assert "tokens" in res and "cost" in res


def test_compact_prompt_removes_boilerplate_and_duplicates():
    noisy = (
        "Could you please help me to summarize the weekly sales report?   \n\n\n"
        "Please kindly note that this dataset covers January metrics for our team.\n"
        "This dataset covers January metrics for our team and needs bullets.\n"
        "Please kindly note that this dataset covers January metrics for our team.\n"
        "Thank you very much in advance!\n"
    )
    res = compact_prompt(noisy, cost_per_1k=0.002)
    assert res["compacted_tokens"] < res["original_tokens"]
    assert res["pct_saved"] > 15.0
    assert res["duplicates_removed"] >= 1
    assert "thank you very much in advance" not in res["compacted_text"].lower()


def test_compact_prompt_preserves_code_fences_and_variables():
    code_prompt = (
        "In order to fix {user_id}, check this block:\n\n"
        "```python\ndef add(a, b):\n    # keep exact spacing\n    return a + b\n```\n"
        "Please make sure to keep {{API_VERSION}} untouched."
    )
    res = compact_prompt(code_prompt)
    assert "```python\ndef add(a, b):\n    # keep exact spacing\n    return a + b\n```" in res["compacted_text"]
    assert "{user_id}" in res["compacted_text"]
    assert "{{API_VERSION}}" in res["compacted_text"]


def test_recommend_model_route_simple_vs_complex():
    simple = recommend_model_route("Extract names and dates as JSON from this short bullet list.")
    deep = recommend_model_route(
        "Design a distributed microservice security audit architecture step by step, "
        "optimize SQL queries, debug traceback errors, and trace root cause:\n```python\nclass Lock:\n    pass\n```"
    )
    assert simple["recommended_tier"] == "budget"
    assert simple["savings_vs_frontier_pct"] > 90.0
    assert deep["recommended_tier"] in ("standard", "frontier")
    assert deep["complexity_score"] > simple["complexity_score"]


def test_profiler_methods_and_empty_compact():
    p = TokenProfiler(cost_per_1k=0.002)
    assert p.compact("")["tokens_saved"] == 0
    assert p.route("Format as CSV")["recommended_tier"] == "budget"


def test_cli_compact_and_route_flags():
    proc = subprocess.run(
        [
            sys.executable,
            "prompt_profiler.py",
            "--text",
            "Please kindly note that we need to extract email and date as JSON. Thank you in advance!",
            "--compact",
            "--route",
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert "compaction" in payload and "routing" in payload
    assert payload["routing"]["recommended_tier"] == "budget"
