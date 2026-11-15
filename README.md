# `prompt-cost-profiler`

[![CI](https://github.com/kaledbouchagraoui7-bot/prompt-cost-profiler/actions/workflows/ci.yml/badge.svg)](https://github.com/kaledbouchagraoui7-bot/prompt-cost-profiler/actions/workflows/ci.yml)

Zero-dependency local LLM Token Profiler, Prompt Compactor, and Complexity-Based Model Router ([Live Zero-Install Browser Demo](https://kaledbouchagraoui7-bot.github.io/prompt-cost-profiler/)).

## Features
- **Token & Multi-Tier Cost Estimation:** Local token count and cost comparison across Budget, Standard, and Frontier model pricing tiers.
- **Prompt Compactor (`compact_prompt()`):** Strips boilerplate filler phrases, extra whitespace, and duplicate context blocks while preserving fenced code blocks (` ```...``` `) and `{variables}` — reporting exact `% saved`. Adapted from open-source compaction patterns (`open-compress/claw-compactor`, MIT).
- **Complexity-Based Model Router (`recommend_model_route()`):** Scores prompt complexity and selects the lowest-cost tier suited to the request. Adapted from open-source routing concepts (`CommonstackAI/UncommonRoute`, MIT).

## Quick Start
```bash
python prompt_profiler.py --text "Your prompt..." --compact --route
pytest -v
```
