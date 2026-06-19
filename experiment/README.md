# Agent for automatic optimizer synthesis

A small framework for generating, validating and benchmarking Python optimizers with an LLM.

The project has four main parts:

- `agent/` — generic agent, model clients, prompt strategy, runner, validation
- `tasks/` — task adapters (for example, COCO BBOB)
- `baselines/` — baseline optimizers and benchmarking helpers
- `experiments/` — runnable scripts for agent runs, baselines and result aggregation

The default experiment uses:

- a `CocoBbobAdapter`
- Python solutions executed by `PythonRunner`
- prompt generation via `DefaultPromptStrategy`
- either `OllamaClient` or `OpenRouterClient`
- checkpointed experiment runs in `results/...`

---

## 1. Repository layout

Typical layout:

```text
project/
  agent/
    __init__.py
    agent.py
    interfaces.py
    models.py
    prompts.py
    runner.py
    types.py
    validation.py

  tasks/
    coco_bbob.py

  baselines/
    opytimizer_bbob.py

  experiments/
    run_agent_bbob.py
    run_baselines_bbob.py
    aggregate_results.py
```

Run scripts from the **project root** with `-m`, for example:

```bash
uv run -m experiments.run_agent_bbob
```

---

## 2. What the agent does

The agent:

1. builds prompts from a task description
2. generates Python candidate solutions with an LLM
3. validates code (`validation.py`)
4. runs smoke tests and full tests
5. repairs runtime failures when possible
6. evolves solutions through:
   - generation
   - combine
   - mutate
7. keeps the best solution found

The core generic interface is intentionally small:

- `ProblemAdapter`
- `ModelClient`
- `Runner`
- `PromptStrategy`

That makes it possible to swap the task, model backend or execution runner with minimal changes.

---

## 3. Main scripts

### `experiments/run_agent_bbob.py`

Runs the full agent loop on the BBOB task.

It supports:

- checkpointed runs
- resuming with the same `--run-name`
- provider switch via `--provider`
- model switch via `--model`
- final holdout evaluation
- debug artifacts and logs

Default provider is `ollama`. The script also supports `openrouter`.

### `experiments/run_baselines_bbob.py`

Runs Opytimizer baselines on the same BBOB task.

It can evaluate either:

- `full` test set
- `final` test set

and optionally a subset of algorithms.

### `experiments/aggregate_results.py`

Collects agent and baseline runs from `results/...` and writes aggregate JSON/CSV summaries.

---

## 4. Default quick start

### 4.1. Run a smoke baseline

```bash
uv run -m experiments.run_baselines_bbob --run-name smoke_base --algorithms pso
```

This creates:

```text
results/baselines_bbob/smoke_base/
  config.json
  problem_text.txt
  raw_results.json
  summary.json
  unavailable_specs.json
```

### 4.2. Run a smoke agent experiment

```bash
uv run -m experiments.run_agent_bbob --run-name smoke_agent --iterations 2
```

This creates:

```text
results/agent_bbob/smoke_agent/
  config.json
  problem_text.txt
  state.json
  events.jsonl
  final_result.json
  best_solution.py
  solutions/
  debug/
```

### 4.3. Aggregate results

```bash
uv run -m experiments.aggregate_results --tag smoke
```

This writes summaries to:

```text
results/aggregate_bbob/
```

---

## 5. Recommended default workflow

If you just want the default end-to-end flow:

```bash
uv run -m experiments.run_baselines_bbob --run-name smoke_base --algorithms pso
uv run -m experiments.run_agent_bbob --run-name smoke_agent --iterations 2
uv run -m experiments.aggregate_results --tag smoke
```

For a less toy-like run:

```bash
uv run -m experiments.run_baselines_bbob --run-name base_exp1 --test-set final
uv run -m experiments.run_agent_bbob --run-name agent_exp1 --iterations 10
uv run -m experiments.aggregate_results --tag exp1
```

---

## 6. Providers and models

## 6.1. Ollama (default)

By default `run_agent_bbob.py` uses:

- `--provider ollama`
- `--model qwen2.5-coder:14b`

Example:

```bash
uv run -m experiments.run_agent_bbob \
  --run-name local_exp \
  --provider ollama \
  --model qwen2.5-coder:14b
```

## 6.2. OpenRouter

To use a remote model via OpenRouter:

```bash
export OPENROUTER_API_KEY=YOUR_KEY
uv run -m experiments.run_agent_bbob \
  --run-name or_exp \
  --provider openrouter \
  --model openai/gpt-oss-120b:free
```

If you want the script to prefer a free OpenRouter model, add `--prefer-free`.
When this flag is set:

- an explicit free model ID ending in `:free` is preserved
- otherwise the script uses `OPENROUTER_FREE_MODEL` if set
- and falls back to `cohere/north-mini-code:free` by default

If `--model` is missing, empty, or still set to the default Ollama placeholder
`qwen2.5-coder:14b`, the script will instead use `OPENROUTER_DEFAULT_MODEL` if set,
or `openai/gpt-oss-120b:free` by default.

Notes:

- `--provider openrouter` is required
- `OPENROUTER_API_KEY` must be set
- `--model` must be a valid OpenRouter model id unless `--prefer-free` is used

---

## 7. Minimal configuration model

`run_agent_bbob.py` stores a single run config in `config.json`.

Important fields:

- `provider`
- `model`
- `iterations`
- `temperature`
- `top_p`
- `num_ctx`
- `num_predict`
- `budget_multiplier`
- retry / timeout settings

In practice:

- for **Ollama**, `num_ctx`, `num_predict`, `think`, `raw` are relevant
- for **OpenRouter**, `num_predict` is used as `max_tokens`

---

## 8. Resume and fault tolerance

`run_agent_bbob.py` is designed to survive bad generations.

If the agent throws during generation or repair:

- the run state is saved
- the process waits and retries
- found solutions remain on disk

If you rerun the same command with the same `--run-name`, the script resumes from the saved run directory.

Example:

```bash
uv run -m experiments.run_agent_bbob --run-name agent_exp1 --iterations 10
```

If something went wrong, just run the same command again.

Useful files:

- `state.json` — current run state
- `events.jsonl` — event log
- `debug/agent.log` — text log
- `debug/outer_errors.log` — outer exceptions
- `solutions/` — all generated solution files

---

## 9. Typical output locations

### Agent run

```text
results/agent_bbob/<run_name>/
```

Important files:

- `best_solution.py`
- `final_result.json`
- `state.json`
- `events.jsonl`
- `solutions/*.py`
- `debug/*`

### Baseline run

```text
results/baselines_bbob/<run_name>/
```

Important files:

- `raw_results.json`
- `summary.json`

### Aggregation

```text
results/aggregate_bbob/
```

Important files:

- `agent_runs_<tag>.json`
- `agent_summary_<tag>.json`
- `baseline_runs_<tag>.json`
- `baseline_summary_<tag>.json`
- `comparison_<tag>.json`
- CSV versions of the same data

---

## 10. Common scenarios

## 10.1. Run just one baseline

```bash
uv run -m experiments.run_baselines_bbob --algorithms pso
```

## 10.2. Run several baselines

```bash
uv run -m experiments.run_baselines_bbob --algorithms pso de gwo
```

## 10.3. Run only the final baseline set

```bash
uv run -m experiments.run_baselines_bbob --test-set final
```

## 10.4. Run a very short local agent test

```bash
uv run -m experiments.run_agent_bbob --iterations 1
```

## 10.5. Run a verbose agent job

```bash
uv run -m experiments.run_agent_bbob --run-name debug_run --iterations 3 --verbose
```

## 10.6. Use OpenRouter without changing code

```bash
export OPENROUTER_API_KEY=YOUR_KEY
uv run -m experiments.run_agent_bbob \
  --provider openrouter \
  --model openai/gpt-oss-120b:free
```

---

## 11. Interpreting scores

The BBOB adapter reports normalized scores where **higher is better**.

Useful fields in agent results:

- `best_score_total_inner_loop`
- `best_score_mean_inner_loop`
- `final_holdout_total_score`
- `final_holdout_mean_score`

For comparison against baselines, prefer:

- `final_holdout_mean_score` for the agent
- `mean_score` for baselines

That is the quantity used by `aggregate_results.py`.

---

## 12. If something breaks

### Import errors
Make sure you launch from the project root with `-m`, for example:

```bash
uv run -m experiments.run_agent_bbob
```

### OpenRouter errors
Check:

- `OPENROUTER_API_KEY`
- `--provider openrouter`
- `--model ...` value

### Repeated bad generations
Inspect:

- `results/.../debug/agent.log`
- `results/.../debug/outer_errors.log`
- generated files in `solutions/`

### Baselines not found
Run:

```bash
uv run -m experiments.run_baselines_bbob --algorithms pso
```

and inspect `unavailable_specs.json`.

---

## 13. Extending the project

To use the agent on another task:

1. implement a new `ProblemAdapter`
2. provide:
   - `problem_text`
   - `build_smoke_tests()`
   - `build_full_tests()`
   - `evaluate_output(...)`
3. plug it into a runner script similar to `run_agent_bbob.py`

To add a new backend:

1. implement `ModelClient.generate(prompt) -> ModelResponse`
2. add a provider branch in `make_client(...)`

To add another execution mode:

1. implement a new `Runner`
2. pass it into `Agent(...)`

---

## 14. Smallest useful command set

If you want the shortest possible cheat sheet:

```bash
# baseline smoke run
uv run -m experiments.run_baselines_bbob --run-name smoke_base --algorithms pso

# agent smoke run
uv run -m experiments.run_agent_bbob --run-name smoke_agent --iterations 2

# aggregate everything
uv run -m experiments.aggregate_results --tag smoke
```
