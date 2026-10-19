<h1 align="center">AutoRLBench</h1>

<p align="center">
  <a href="https://thinking-space.github.io/AutoRLBench/"><img src="assets/logo.png" alt="AutoRLBench" width="360"></a>
</p>

<p align="center">
  <a href="https://thinking-space.github.io/AutoRLBench/"><img alt="Website" src="https://img.shields.io/badge/Website-thinking--space.github.io-171716?logo=googlechrome&logoColor=white"></a>
  <a href="https://thinking-space.github.io/AutoRLBench/#leaderboard"><img alt="Leaderboard" src="https://img.shields.io/badge/Leaderboard-0.1-171716?logo=trophy&logoColor=white"></a>
  <a href="https://thinking-space.github.io/AutoRLBench/#tasks"><img alt="Tasks" src="https://img.shields.io/badge/Tasks-6%20domains-171716?logo=stack&logoColor=white"></a>
  <a href="https://thinking-space.github.io/AutoRLBench/contribute.html"><img alt="Contribute" src="https://img.shields.io/badge/Contribute-a%20run-171716?logo=git-pull-request&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-171716?logo=apache&logoColor=white"></a>
  <a href="README_zh.md"><img alt="Chinese (Simplified)" src="https://img.shields.io/badge/Chinese-Simplified-171716"></a>
</p>

A benchmark and open call for valuable, reproducible RL runs. Each task is a real training run that went wrong. The agent has to find the root cause (**issue**) and, where a repaired run exists, fix it and rerun (**solution**).

## Leaderboard

AutoRLBench Leaderboard 0.1, issue evaluation. *First release; updated as new agents and tasks are added.*

| # | Model | Agent | Issue score | |
| ---: | --- | --- | ---: | :--- |
| 1 | GLM-5.3 | Claude Code | 0.92 | ███████████████████ |
| 2 | Claude-Fable-5.1 | Claude Code | 0.77 | ████████████████░░░ |
| 3 | Claude-Opus-5.5 | Claude Code | 0.77 | ████████████████░░░ |
| 4 | Kimi-K3 | Claude Code | 0.77 | ████████████████░░░ |
| 5 | DeepSeek-V4-Pro | Claude Code | 0.69 | ██████████████░░░░░ |
| 6 | GPT-6-Astra | Codex | 0.62 | █████████████░░░░░░ |
| 7 | GPT-6.1-Sol | Codex | 0.54 | ███████████░░░░░░░░ |

Solution evaluation (repair and rerun) is coming soon. Full results: [website](https://thinking-space.github.io/AutoRLBench/#leaderboard).

## Tasks

Six domains, one representative task each. The domains come from where in the RL pipeline the failure lives:

| Domain | Task | What goes wrong | Evaluations |
| --- | --- | --- | --- |
| **Configuration** | [AIME25 single-sample variance](tasks/configuration/verl_aime25_single_sample_protocol/) | One sample per question makes a 30-question score too noisy to judge the run. | issue · solution |
| **Reward** | [Length bonus reward hack](tasks/reward/slime_retool_length_reward_bonus/) | An unconditional length bonus teaches the policy to write longer, not better. | issue · solution |
| **Data** | [Training data is too hard](tasks/data/verl_taco_difficulty_mismatch/) | Train is TACO `VERY_HARD` while validation is `EASY`; held-out accuracy ends below its start. | issue |
| **Environment** | [Zero query vector in retrieval](tasks/environment/search_r1_zero_query_embedding/) | The retriever's query vector is all zeros, so every query gets the same documents. | issue · solution |
| **Algorithm** | [Ranking distillation collapse](tasks/algorithm/rank_distillation_collapse/) | The ranking loss improves while the student's generation quality collapses. | issue |
| **Infrastructure** | [Missing flash-attention hook](tasks/infrastructure/verl_missing_flash_attn_patch/) | A rewritten monkey patch silently drops the flash-attention hook. | issue · solution |

## How a task works

```text
prepare ──▶ agent ──▶ verifier
  │           │          │
  │           │          └─ issue:    LLM judge on agent_answer.md vs. hidden oracle
  │           │             solution: rerun, then a metric threshold
  │           └─ reads instruction.md and a writable copy of workspace/
  └─ fetches pinned models and datasets (the only stage with network)
```

Each stage runs in its own container from the same prebuilt image. Oracles and verifiers are mounted only into the verifier stage.

## Repository layout

| Path | Contents |
| --- | --- |
| [`tasks/`](tasks/) | Benchmark tasks, grouped by domain |
| [`autorlbench/`](autorlbench/) | Evaluation harness: task loader, validator, runtimes, agent adapters, CLI |
| [`docker/`](docker/) | Shared environment images and in-container tooling ([README](docker/README.md)) |
| [`skills/package-task/`](skills/package-task/) | Agent skill that turns a raw task into a task |
| [`tests/`](tests/) | Harness unit tests |
| [`page/`](page/) | Website source, deployed to GitHub Pages |

## Task layout

```text
tasks/<domain>/<task_id>/
├── README.md               # summary, difficulty, reference solution, verification
├── task.toml               # metadata and environment image
├── workspace/              # the run as the agent sees it: code, config, logs, metrics
├── environment/prepare.sh  # fetches pinned models and datasets (only stage with network)
└── evaluations/
    ├── issue/              # instruction.md · oracle.md · verifier/
    └── solution/           # instruction.md · oracle/ · verifier/
```

## Quick start

```bash
# Validate a task
python3 -m autorlbench validate tasks/reward/slime_retool_length_reward_bonus

# Run an agent on the issue level in Docker, then verify
python3 -m autorlbench evaluate tasks/reward/slime_retool_length_reward_bonus \
  --evaluation-level issue --backend docker \
  --agent codex --model <model> --judge-model <judge-model> --json

# Batch evaluation through Harbor (pip install harbor)
python3 -m autorlbench harbor-evaluate tasks/reward --agent codex --model <model>
```

Run `python3 -m autorlbench --help` for every option. The issue verifier calls an OpenAI-compatible judge through `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

## Contributing

We welcome both problematic runs and healthy reference runs. Bring the issue materials, or add the repaired run when it exists; an agent does the packaging. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [contribution guide](https://thinking-space.github.io/AutoRLBench/contribute.html).

## License

Apache-2.0. Each task's workspace keeps the license of its upstream code.
