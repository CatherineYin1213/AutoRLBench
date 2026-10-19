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
  <a href="README.md"><img alt="English" src="https://img.shields.io/badge/English-README-171716"></a>
</p>

一个面向真实、可复现 RL 训练运行的 benchmark，也是一个开放征集。每个任务都来自一次真实跑坏的训练：agent 要找到根因（**issue**），有修复版本的任务还要改好并重跑（**solution**）。

## Leaderboard

AutoRLBench Leaderboard 0.1，issue 档成绩。*初版，会随新 agent 与新任务持续更新。*

| # | 模型 | Agent | Issue 分数 | |
| ---: | --- | --- | ---: | :--- |
| 1 | GLM-5.3 | Claude Code | 0.92 | ███████████████████ |
| 2 | Claude-Fable-5.1 | Claude Code | 0.77 | ████████████████░░░ |
| 3 | Claude-Opus-5.5 | Claude Code | 0.77 | ████████████████░░░ |
| 4 | Kimi-K3 | Claude Code | 0.77 | ████████████████░░░ |
| 5 | DeepSeek-V4-Pro | Claude Code | 0.69 | ██████████████░░░░░ |
| 6 | GPT-6-Astra | Codex | 0.62 | █████████████░░░░░░ |
| 7 | GPT-6.1-Sol | Codex | 0.54 | ███████████░░░░░░░░ |

solution 档（修复并重跑）即将开放。完整结果见[网站](https://thinking-space.github.io/AutoRLBench/#leaderboard)。

## 任务

六个 domain，每个放一个代表性任务。domain 按失败发生在 RL 流程的哪一环来划分：

| Domain | 任务 | 出了什么问题 | 评测 |
| --- | --- | --- | --- |
| **Configuration** | [AIME25 单样本方差](tasks/configuration/verl_aime25_single_sample_protocol/) | 每题只采样一次，30 题的分数噪声大到无法判断这次运行。 | issue · solution |
| **Reward** | [长度奖励劫持](tasks/reward/slime_retool_length_reward_bonus/) | 无条件长度奖励让策略学会写得更长，而不是写得更好。 | issue · solution |
| **Data** | [训练数据太难](tasks/data/verl_taco_difficulty_mismatch/) | 训练集是 TACO `VERY_HARD`、验证集是 `EASY`，held-out 准确率跌回起点以下。 | issue |
| **Environment** | [检索查询向量全零](tasks/environment/search_r1_zero_query_embedding/) | 检索器的 query 向量全是零，任何问题都返回同一批文档。 | issue · solution |
| **Algorithm** | [排序蒸馏崩溃](tasks/algorithm/rank_distillation_collapse/) | 排序 loss 在下降，学生的生成质量却在崩塌。 | issue |
| **Infrastructure** | [缺少 flash-attention hook](tasks/infrastructure/verl_missing_flash_attn_patch/) | 重写的 monkey patch 悄悄丢掉了 flash-attention hook。 | issue · solution |

## 一个任务怎么跑

```text
prepare ──▶ agent ──▶ verifier
  │           │          │
  │           │          └─ issue:    LLM judge 比对 agent_answer.md 与隐藏 oracle
  │           │             solution: 重跑后按指标门槛判定
  │           └─ 读取 instruction.md 与一份可写的 workspace/
  └─ 拉取钉死版本的模型与数据集（唯一联网的阶段）
```

三个阶段各自起容器，用的是同一个预构建镜像。oracle 与 verifier 只挂进 verifier 阶段。

## 目录

| 路径 | 内容 |
| --- | --- |
| [`tasks/`](tasks/) | 按 domain 分组的 benchmark 任务 |
| [`autorlbench/`](autorlbench/) | 评测框架：任务加载、校验、运行时、agent 适配器、CLI |
| [`docker/`](docker/) | 共享环境镜像与容器内工具（[说明](docker/README.md)） |
| [`skills/package-task/`](skills/package-task/) | 把 raw task 打包成任务的 agent skill |
| [`tests/`](tests/) | 框架单元测试 |
| [`page/`](page/) | 网站源码，部署到 GitHub Pages |

## 任务结构

```text
tasks/<domain>/<task_id>/
├── README.md               # 摘要、难度、参考解、判分方式
├── task.toml               # 元数据与环境镜像
├── workspace/              # agent 看到的那次运行：代码、配置、日志、指标
├── environment/prepare.sh  # 拉取钉死版本的模型与数据集（唯一联网阶段）
└── evaluations/
    ├── issue/              # instruction.md · oracle.md · verifier/
    └── solution/           # instruction.md · oracle/ · verifier/
```

## 快速开始

```bash
# 校验一个任务
python3 -m autorlbench validate tasks/reward/slime_retool_length_reward_bonus

# 在 Docker 里跑 issue 档，然后判分
python3 -m autorlbench evaluate tasks/reward/slime_retool_length_reward_bonus \
  --evaluation-level issue --backend docker \
  --agent codex --model <model> --judge-model <judge-model> --json

# 通过 Harbor 批量评测（pip install harbor）
python3 -m autorlbench harbor-evaluate tasks/reward --agent codex --model <model>
```

全部参数见 `python3 -m autorlbench --help`。issue 档的 verifier 通过 `OPENAI_BASE_URL` 与 `OPENAI_API_KEY` 调用 OpenAI 兼容的 judge。

## 贡献

有问题的 run 和健康的对照 run 都欢迎。带上 issue 所需材料即可，已有修复版本时再补上 solution 运行；打包由 agent 完成。详见 [CONTRIBUTING.md](CONTRIBUTING.md) 与[贡献指南](https://thinking-space.github.io/AutoRLBench/contribute.html)。

## 许可

Apache-2.0。各任务的 workspace 沿用其上游代码的许可。
