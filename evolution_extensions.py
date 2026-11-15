import json
import hashlib
from datetime import datetime
from typing import Any, Dict

class EvolutionExtensions:
    """Provides JSON/SARIF export and security scoring for prompt profiles."""
    
    @staticmethod
    def generate_sarif_report(profile_data: Dict[str, Any], tool_name: str = "prompt_cost_profiler") -> str:
        """Exports profiling data into a simplified SARIF-like JSON structure."""
        report = {
            "version": "2.1.0",
            "runs": [{
                "tool": {"driver": {"name": tool_name, "version": "1.0.0"}},
                "results": [{"message": {"text": f"Profile generated for {profile_data.get('id', 'unknown')}"}, "kind": "informational"}]
            }]
        }
        return json.dumps(report, indent=2)

    @staticmethod
    def calculate_security_score(prompt_text: str) -> float:
        """Grades prompt based on length and entropy as a proxy for injection risk."""
        if not prompt_text:
            return 1.0
        # Simple heuristic: longer prompts with high character variance score lower on safety
        length_factor = min(1.0, len(prompt_text) / 5000)
        entropy = len(set(prompt_text)) / (len(prompt_text) + 1)
        return round(1.0 - (length_factor * 0.5 + entropy * 0.5), 2)