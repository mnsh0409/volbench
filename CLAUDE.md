# volbench — guidance for Claude

Open-source framework for leakage-safe evaluation of probabilistic
volatility & tail-risk forecasts. Paper target: IJF (Open-Source
Forecasting special section). Never violate:

1. Temporal integrity: no code path may let information from t' > t
   influence a forecast for t. The RollingOriginSplitter
   (src/volbench/splitter.py) is the only sanctioned way to produce
   train/test indices; new evaluation code must consume it, never
   hand-rolled slices. tests/test_splitter.py is the contract — if a
   feature requires weakening those tests, the feature is wrong.
2. Probabilistic outputs are first-class: adapters return the
   Distribution object (src/volbench/dist.py), never bare point arrays.
   A model's predict() returns a distribution over the NEXT-PERIOD
   RETURN; its variance is the variance forecast. Daily units always,
   never annualized.
3. Determinism: every entry point takes a seed; results carry a config
   hash; `make reproduce` must stay green.

## Read these before designing anything

| File | What it settles |
|---|---|
| `docs/research_design.md` | hypotheses, data panel, model list, protocol, out-of-scope guard rail |
| `docs/metrics_reference.md` | exact metric definitions and notation — code and paper must match it |
| `docs/data_sources.md` | which sources may be used, their licences, redistribution rules |
| `docs/decisions.md` | settled decisions (D-001…) — do not relitigate without an explicit reopen |
| `docs/design.md` | component architecture and invariants |
| `docs/phase1_prompts.md` | the per-stream task prompts for this phase |

If a task would contradict any of those, STOP and report rather than
working around it. Those files are mirrors of the planning folder; treat
them as read-only here and flag drift instead of editing them — with two
exceptions, available only when a task explicitly instructs the edit:
appends to `docs/decisions.md` and updates to `docs/M2_NOTES.md`. Masters
stay on the planning machine, which reconciles anything appended here
(numbering included). Never edit the other mirrors, and never edit these
two uninstructed.

## Conventions

Python 3.11+, uv, ruff (line 100), mypy --strict on src, pytest (every
bug becomes a regression test). Public API changes update docs/design.md
in the same PR. Conventional commits. Data: only sources listed in
docs/data_licenses.md; never vendor non-redistributable data. Secrets
(e.g. TimeGPT keys) via env vars only — never in code, config, fixtures.

Run checks: `uv run pytest && uv run ruff check . && uv run mypy`.

## Skills

`.claude/skills/leakage-check` — run it over any change touching data
loading, splitting, features, scaling, refitting, caching, or evaluation.
`.claude/skills/ijf-writing` — for any prose destined for the manuscript.
