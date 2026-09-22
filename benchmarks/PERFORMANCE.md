# Preliminary Jericho decision-mode comparison

Tested decision implementation: `f51fa32` on `codex/single-forward-decisions`.

Single-token decisions reduced inference overhead, but did not improve game
scores in this small test. This is a workflow comparison, not evidence of
improved long-horizon reasoning or self-evolution.

## Setup

- OpenRouter `openai/gpt-4o-mini` for both modes, Zork1 and Library, seeds 0/1,
  at most 20 environment steps per run; eight scheduled runs.
- Actual `JitRLAgent.generate_action` and repository `JerichoEnv`, with real ROMs.
  Both modes received environment legal actions. No candidate pruning.
- Cross-episode memory and prompt evolution disabled in both modes to isolate
  the current decision workflows. This does not evaluate full JitRL adaptation.
- Legacy used verbalized confidence, three proposed actions, temperature 0,
  and its existing history-summary generation. The API decision mode used all
  legal actions, temperature 1 for probability extraction, and selected argmax.
  The sampled output letter was not used for execution.
- Game seeds were explicitly applied to the underlying Frotz environment (the
  repository wrapper stores the seed without applying it). Legacy action-list
  shuffling used a fixed step/seed value. API output is not guaranteed deterministic.

## Scores and reliability

- Zork1, seed 0: new mode stopped after 10 steps with an API read timeout,
  score 0; legacy completed 20 steps, score 0. Excluded from paired efficiency totals.
- Zork1, seed 1: both completed 20 steps, scores 0 / 0 (new / legacy).
- Library, seed 0: both completed 20 steps, scores 5 / 6.
- Library, seed 1: both completed 20 steps, scores 5 / 5.

All executed actions in the saved runs were in the legal list. Candidate sets
contained at most eight actions; no missing-label or >20-candidate failures
occurred. This does not establish API coverage for larger action sets.
Step-limit completion is not game victory, and no task-success-rate improvement
can be inferred from these scores.

## Efficiency on the three fully completed pairs only

Each mode executed 60 environment steps:

- Decision-path time: new **101.41 s** vs legacy **414.14 s**; mean **1.69 s** vs
  **6.90 s** per step, approximately **4.08x** faster.
- API-reported cost: new **$0.0079734** vs legacy **$0.0394110**, approximately
  **79.8%** lower.
- Successful API calls: **60** vs **117**. Legacy usually makes a history-summary
  call and an action-generation call; new mode makes one scoring call.
- Input tokens: **54,605** vs **204,418**. Output tokens: **60** vs **20,620**.

Decision-path time includes all API work performed inside `generate_action`,
including legacy summaries, but excludes environment work and environment/model
setup. These are network-observed timings, not GPU-forward benchmarks.
Costs are reported usage for returned responses, not a complete billing audit.

An initial harness run omitted legacy retrieval arguments and was discarded.
One later legacy seed-1 attempt stalled during a summary request and was manually
interrupted and rerun with bounded waits; it is not included in the paired totals.
The initial recorded new-mode timeout used a 60-second limit. After the stall,
remaining SDK and urllib requests were bounded to 30 seconds, without retries.
Completed pairs did not hit these limits. Runs were sequential, with new mode
first; provider routing, caching, and latency can affect the measured differences.

## Reproduction

Set `OPENROUTER_API_KEY` in your environment, then run from the repository root:

```bash
uv run --python 3.10 --with jericho --with 'numpy<2' --with 'openai<2' \
  --with tiktoken --with python-dotenv \
  python benchmarks/compare_decision_modes.py \
  --games zork1 library --seeds 0 1 --steps 20 \
  --output benchmarks/results/paired
```

Use `--prompt-key` for a hidden interactive key prompt and `--resume` to skip
already recorded runs with matching game, seed, mode, model, and step limit.
Results include per-step states, allowed actions, chosen actions, responses,
scores, timing and aggregate usage. Credentials are not serialized.
