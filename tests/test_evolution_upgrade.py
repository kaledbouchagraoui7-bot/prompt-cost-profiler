import pytest
from evolution_extensions import EvolutionExtensions

def test_sarif_structure():
    data = {"id": "test-123"}
    report = EvolutionExtensions.generate_sarif_report(data)
    assert "2.1.0" in report
    assert "test-123" in report

def test_security_score_range():
    score = EvolutionExtensions.calculate_security_score("Hello world")
    assert 0 <= score <= 1.0

def test_security_score_empty():
    assert EvolutionExtensions.calculate_security_score("") == 1.0

def test_sarif_tool_name():
    report = EvolutionExtensions.generate_sarif_report({}, tool_name="custom-tool")
    assert "custom-tool" in report