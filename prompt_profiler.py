#!/usr/bin/env python3
"""
LLM Prompt Cost Profiler, Prompt Compactor & Complexity Model Router (`prompt_profiler.py`)

Combines local token/cost benchmarking with two cost-reduction engines inspired cleanly by:
  - `open-compress/claw-compactor` (MIT): whitespace/boilerplate compaction & CJK-aware tokenization
  - `CommonstackAI/UncommonRoute` (MIT): multi-tier complexity-based model routing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from typing import Any


CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
FILLER_PHRASES_RE = re.compile(
    r"\b(?:"
    r"please\s+kindly\s+note\s+that|"
    r"it\s+is\s+important\s+to\s+note\s+that|"
    r"as\s+an\s+ai\s+language\s+model[,]?|"
    r"thank\s+you\s+very\s+much\s+in\s+advance[.!]?|"
    r"thank\s+you\s+in\s+advance[.!]?|"
    r"i\s+would\s+really\s+appreciate\s+if\s+you\s+could|"
    r"could\s+you\s+please\s+help\s+me\s+to|"
    r"in\s+order\s+to|"
    r"please\s+make\s+sure\s+to"
    r")\b",
    flags=re.IGNORECASE,
)

MODEL_TIERS: dict[str, dict[str, Any]] = {
    "budget": {
        "tier": "budget",
        "model_family": "Fast Flash / Mini Tier (e.g., gemini-flash-lite / gpt-mini)",
        "cost_per_1k": 0.0005,
        "best_for": "Formatting, JSON/CSV extraction, classification, short translations",
    },
    "standard": {
        "tier": "standard",
        "model_family": "Balanced Workhorse Tier (e.g., gemini-flash / sonnet-standard)",
        "cost_per_1k": 0.002,
        "best_for": "Multi-paragraph drafting, unit-test generation, standard coding tasks",
    },
    "frontier": {
        "tier": "frontier",
        "model_family": "Frontier Reasoning Tier (e.g., gemini-pro / opus-frontier)",
        "cost_per_1k": 0.015,
        "best_for": "System architecture, deep debugging, formal proofs, multi-file refactoring",
    },
}


def estimate_token_count(text: str) -> int:
    """Estimate token count while staying compatible with word-based & code/CJK prompts."""
    if not text or not text.strip():
        return 0
    words_est = int(len(text.split()) * 1.33)
    chars_est = max(1, int(len(text.strip()) / 4))
    return max(1, words_est, chars_est)


def compact_prompt(text: str, cost_per_1k: float = 0.002) -> dict[str, Any]:
    """
    Compact a prompt by removing redundant whitespace, filler phrases, and duplicate
    prose paragraphs while preserving code fences (```...```) and variables intact.
    """
    if not text or not text.strip():
        return {
            "compacted_text": "",
            "original_tokens": 0,
            "compacted_tokens": 0,
            "tokens_saved": 0,
            "pct_saved": 0.0,
            "original_cost": 0.0,
            "compacted_cost": 0.0,
            "cost_saved": 0.0,
            "duplicates_removed": 0,
        }

    # 1. Stash fenced code blocks verbatim so code indentation & braces never mutate
    preserved_blocks: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        idx = len(preserved_blocks)
        preserved_blocks.append(match.group(0))
        return f"\n@@CODE_BLOCK_{idx}@@\n"

    masked = CODE_FENCE_RE.sub(_stash, text)

    # 2. Strip common polite/verbose filler phrases cleanly inside non-code text
    masked = FILLER_PHRASES_RE.sub("", masked)

    # 3. Collapse redundant horizontal spaces per line and deduplicate repeated prose blocks
    raw_lines = masked.splitlines()
    seen_hashes: set[str] = set()
    cleaned_lines: list[str] = []
    duplicates_removed = 0

    for raw_ln in raw_lines:
        stripped = re.sub(r"[ \t]+", " ", raw_ln).strip()
        if not stripped:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        if stripped.startswith("@@CODE_BLOCK_") and stripped.endswith("@@"):
            cleaned_lines.append(stripped)
            continue
        norm_key = hashlib.sha256(stripped.lower().encode("utf-8")).hexdigest()[:16]
        if len(stripped) >= 16 and norm_key in seen_hashes:
            duplicates_removed += 1
            continue
        seen_hashes.add(norm_key)
        # Clean stray leading punctuation left after filler removal
        stripped = re.sub(r"^[,;:\s]+", "", stripped)
        cleaned_lines.append(stripped)

    compacted = "\n".join(cleaned_lines).strip()
    compacted = re.sub(r"\n{3,}", "\n\n", compacted)

    # 4. Restore fenced code blocks untouched
    for idx, block in enumerate(preserved_blocks):
        compacted = compacted.replace(f"@@CODE_BLOCK_{idx}@@", block)

    orig_tokens = estimate_token_count(text)
    new_tokens = min(orig_tokens, estimate_token_count(compacted))
    saved = max(0, orig_tokens - new_tokens)
    pct_saved = round((saved / orig_tokens) * 100.0, 2) if orig_tokens > 0 else 0.0

    orig_cost = round((orig_tokens / 1000.0) * cost_per_1k, 6)
    new_cost = round((new_tokens / 1000.0) * cost_per_1k, 6)

    return {
        "compacted_text": compacted,
        "original_tokens": orig_tokens,
        "compacted_tokens": new_tokens,
        "tokens_saved": saved,
        "pct_saved": pct_saved,
        "original_cost": orig_cost,
        "compacted_cost": new_cost,
        "cost_saved": round(max(0.0, orig_cost - new_cost), 6),
        "duplicates_removed": duplicates_removed,
    }


def recommend_model_route(text: str) -> dict[str, Any]:
    """
    Score prompt complexity and recommend the lowest-cost model tier suited to the task.
    Returns complexity breakdown and cost savings vs. defaulting everything to Frontier.
    """
    tokens = estimate_token_count(text)
    lower = (text or "").lower()
    reasons: list[str] = []
    score = 0.15

    simple_signals = ("extract", "format", "summarize", "summary", "json", "csv", "translate", "classify", "bullet")
    coding_signals = ("```", "def ", "class ", "traceback", "stack trace", "pytest", "refactor", "sql", "regex", "bug")
    reasoning_signals = (
        "architecture", "tradeoff", "prove", "derive", "theorem", "multi-step", "step by step",
        "security audit", "root cause", "distributed", "Concurrency".lower(), "optimize",
    )

    simple_hits = [kw for kw in simple_signals if kw in lower]
    coding_hits = [kw for kw in coding_signals if kw in lower]
    reasoning_hits = [kw for kw in reasoning_signals if kw in lower]

    if tokens > 250:
        score += 0.18
        reasons.append(f"medium_length ({tokens} tokens)")
    if tokens > 900:
        score += 0.22
        reasons.append(f"long_context ({tokens} tokens)")
    if coding_hits:
        score += min(0.30, 0.12 * len(coding_hits))
        reasons.append(f"code_signals={coding_hits[:3]}")
    if reasoning_hits:
        score += min(0.40, 0.16 * len(reasoning_hits))
        reasons.append(f"reasoning_signals={reasoning_hits[:3]}")
    if simple_hits and not coding_hits and not reasoning_hits:
        score -= 0.10
        reasons.append(f"simple_extraction={simple_hits[:3]}")

    score = round(max(0.05, min(1.0, score)), 2)

    if score >= 0.65:
        chosen_key = "frontier"
    elif score >= 0.35:
        chosen_key = "standard"
    else:
        chosen_key = "budget"

    tier_info = MODEL_TIERS[chosen_key]
    est_cost = round((tokens / 1000.0) * tier_info["cost_per_1k"], 6)
    frontier_cost = round((tokens / 1000.0) * MODEL_TIERS["frontier"]["cost_per_1k"], 6)
    savings_pct = round(((frontier_cost - est_cost) / frontier_cost) * 100.0, 2) if frontier_cost > 0 else 0.0

    return {
        "recommended_tier": tier_info["tier"],
        "recommended_model_family": tier_info["model_family"],
        "cost_per_1k": tier_info["cost_per_1k"],
        "complexity_score": score,
        "estimated_tokens": tokens,
        "estimated_cost_usd": est_cost,
        "baseline_frontier_cost_usd": frontier_cost,
        "savings_vs_frontier_pct": savings_pct,
        "reasons": reasons or ["short general prompt"],
    }


class TokenProfiler:
    """Profiles token counts, API tier costs, prompt compaction, and model routing."""

    def __init__(self, cost_per_1k: float = 0.002) -> None:
        self.cost_per_1k = cost_per_1k

    def estimate(self, text: str) -> dict[str, Any]:
        if not text:
            return {"tokens": 0, "cost": 0.0}
        tokens = int(len(text.split()) * 1.33)
        if tokens == 0 and text.strip():
            tokens = 1
        cost = (tokens / 1000.0) * self.cost_per_1k
        return {"tokens": int(tokens), "cost": round(cost, 6)}

    def compact(self, text: str) -> dict[str, Any]:
        return compact_prompt(text, cost_per_1k=self.cost_per_1k)

    def route(self, text: str) -> dict[str, Any]:
        return recommend_model_route(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Cost Profiler, Compactor & Router")
    parser.add_argument("--text", help="Input prompt text")
    parser.add_argument("--compact", action="store_true", help="Compact prompt and report token/cost savings")
    parser.add_argument("--route", action="store_true", help="Recommend lowest-cost model tier by complexity")
    parser.add_argument("--cost-per-1k", type=float, default=0.002, help="Cost per 1k tokens in USD (default: 0.002)")
    args = parser.parse_args()

    if not args.text:
        print("Usage: python prompt_profiler.py --text 'your prompt here' [--compact] [--route]")
        return

    profiler = TokenProfiler(cost_per_1k=args.cost_per_1k)
    if args.compact or args.route:
        out: dict[str, Any] = {"estimate": profiler.estimate(args.text)}
        if args.compact:
            out["compaction"] = profiler.compact(args.text)
        if args.route:
            out["routing"] = profiler.route(args.text)
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps(profiler.estimate(args.text)))


if __name__ == "__main__":
    main()
