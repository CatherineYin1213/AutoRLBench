
<p align="center">
  <a href="https://thinking-space.github.io/AutoRLBench/"><img src="assets/logo.png" alt="AutoRLBench" width="360"></a>
</p>

<p align="center">
  <a href="https://thinking-space.github.io/AutoRLBench/"><img alt="Website" src="https://img.shields.io/badge/Website-thinking--space.github.io-4285F4?logo=googlechrome&logoColor=white"></a>
  <a href="https://thinking-space.github.io/AutoRLBench/#leaderboard"><img alt="Leaderboard" src="https://img.shields.io/badge/Leaderboard-0.1-FFB000?logo=trophy&logoColor=white"></a>
  <a href="https://thinking-space.github.io/AutoRLBench/#tasks"><img alt="Tasks" src="https://img.shields.io/badge/Tasks-6%20domains-00C853?logo=stack&logoColor=white"></a>
  <a href="https://thinking-space.github.io/AutoRLBench/contribute.html"><img alt="Contribute" src="https://img.shields.io/badge/Contribute-a%20run-7B1FA2?logo=git-pull-request&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-D22128?logo=apache&logoColor=white"></a>
  <a href="README_zh.md"><img alt="Chinese (Simplified)" src="https://img.shields.io/badge/Chinese-Simplified-E74C3C"></a>
</p>

A benchmark and open call for valuable, reproducible RL runs. Each task is a real training run that went wrong. The agent has to find the root cause (**issue**) and, where a repaired run exists, fix it and rerun (**solution**).

<img src="assets/ai-trains-ai.jpeg" alt="AI Trains AI" width="480">

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

<table>
<tr>
<td width="50%" valign="top">
<a href="tasks/configuration/verl_aime25_single_sample_protocol/">
<img src="assets/task-configuration.jpeg" alt="Configuration" width="100%">
</a>
<strong>Configuration</strong> · issue · solution<br>
<a href="tasks/configuration/verl_aime25_single_sample_protocol/"><b>AIME25 single-sample variance</b></a><br>
One sample per question makes a 30-question score too noisy to judge the run.
</td>
<td width="50%" valign="top">
<a href="tasks/reward/slime_retool_length_reward_bonus/">
<img src="assets/task-reward.jpeg" alt="Reward" width="100%">
</a>
<strong>Reward</strong> · issue · solution<br>
<a href="tasks/reward/slime_retool_length_reward_bonus/"><b>Length bonus reward hack</b></a><br>
An unconditional length bonus teaches the policy to write longer, not better.
</td>
</tr>
<tr>
<td width="50%" valign="top">
<a href="tasks/data/verl_taco_difficulty_mismatch/">
<img src="assets/task-data.jpeg" alt="Data" width="100%">
</a>
<strong>Data</strong> · issue<br>
<a href="tasks/data/verl_taco_difficulty_mismatch/"><b>Training data is too hard</b></a><br>
Train is TACO <code>VERY_HARD</code> while validation is <code>EASY</code>; held-out accuracy ends below its start.
</td>
<td width="50%" valign="top">
<a href="tasks/environment/search_r1_zero_query_embedding/">
<img src="assets/task-environment.jpeg" alt="Environment" width="100%">
</a>
<strong>Environment</strong> · issue · solution<br>
<a href="tasks/environment/search_r1_zero_query_embedding/"><b>Zero query vector in retrieval</b></a><br>
The retriever's query vector is all zeros, so every query gets the same documents.
</td>
</tr>
<tr>
<td width="50%" valign="top">
<a href="tasks/algorithm/verl_grpo_uid_after_repeat/">
<img src="assets/task-algorithm.jpeg" alt="Algorithm" width="100%">
</a>
<strong>Algorithm</strong> · issue<br>
<a href="tasks/algorithm/verl_grpo_uid_after_repeat/"><b>GRPO uid assigned after rollout repeat</b></a><br>
The uid assignment lands after the rollout repeat step, making every GRPO group a singleton and silently degrading training to baseline-free REINFORCE.
</td>
<td width="50%" valign="top">
<a href="tasks/infrastructure/verl_missing_flash_attn_patch/">
<img src="assets/task-infrastructure.jpeg" alt="Infrastructure" width="100%">
</a>
<strong>Infrastructure</strong> · issue · solution<br>
<a href="tasks/infrastructure/verl_missing_flash_attn_patch/"><b>Missing flash-attention hook</b></a><br>
A rewritten monkey patch silently drops the flash-attention hook.
</td>
</tr>
</table>

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
