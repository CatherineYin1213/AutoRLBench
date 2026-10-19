# slime_retool_length_reward_bonus

| | |
| --- | --- |
| **Author** | chenyinghao |
| **Domain** | `reward` |
| **Tags** | `slime` `retool` `grpo` `reward-hacking` `length-bonus` `aime` |
| **Framework** | `slime`（Megatron train + SGLang rollout），`examples/retool/` |
| **Model** | Qwen3-4B（Instruct-2507 SFT → RL） |
| **Dataset** | DAPO-Math-17k（RL）+ ReTool-SFT（SFT warm-up） |
| **Environment** | `slime-v0.3.2`；`prepare.sh` 取四份资产 |
| **Resources** | 8 GPUs |
| **Recorded runs** | 缺陷侧一次历史运行（100 步）；修复侧参考运行 70 rollouts |
| **Recorded runtime** | 100 步（冻结证据） |

## Task abstract

奖励函数在正确性得分之上**无条件**加了 `5 * length_ratio`，于是策略靠生成更长的 rollout
劫持奖励。日志里的 `eval/aime` 一路涨到 **3.841**（超出 `[-1, 1]`），而真实 held-out
准确率同时崩到约 8.4%，eval 回复长度从约 4.1k 涨到 15.3k，截断率从约 0.8% 涨到 82.5%。
奖励在设计上就不认长度之外的东西。

## The defect

`examples/retool/generate_with_retool.py::reward_func`，在 math_dapo 精确匹配分
（`strict_box_verify=True`，`Answer: \boxed{...}`）与有界的工具调用 shaping 项之上，对
**每个**样本额外加：

```python
5 * min(sample.response_length / args.rollout_max_response_len, 1.0)
```

该项只依赖生成长度、不依赖正确性，所以"错而长"可以压过"对而短"。

### Why nothing catches it

- **日志指标看起来在进步。** `eval/aime` 是 `reward_func` 的均值，本身包含长度项，
  于是它**涨**——涨到 `[-1, 1]` 之外也不报错。
- **训练完全正常。** SFT → RL 配方原样复现，工具 rollback、token/log-prob 对齐、
  上下文预算都有断言，核心框架未改。
- **不寻常但正确的设置会误导。** `kl-loss-coef 0.00`、`--rotary-base 5000000`、
  `max_turns=16`、`tool_concurrency=32`、W&B debug 日志、`/root/…` 下的运行时路径，
  都是配方的正常属性，不是原因。
- **需要把日志指标翻译成真实准确率才看得见。** `reward_func` 把错误样本地板在
  `min(-0.6, …)` 而非 `-1`，所以均值分数不能按 `(score + 1) / 2` 换算成准确率。

## Evidence

- AIME-2024 的 11 个 eval 点，step 0 到 99
- 记录的 `eval/aime` 从约 +0.82 涨到 +3.841（超出 `[-1, 1]`）
- eval 回复长度 4.1k → 15.3k；eval 截断率 0.8% → 82.5%
- step 100 的训练回复长度约 15159 token，截断率 78.5%
- 去污染后的 held-out 分数从 step 0 的约 -0.417 掉到 step 99 的 -0.831，把精确匹配
  准确率上界压到约 29.2% 与 8.4%——**这是上界，不是点估计**

配方其余部分正确：多轮工具 rollout 的 `<interpreter>…</interpreter>` 观察以
`trainable=False` 追加；缺 `output_token_logprobs` 会中止样本；逐轮 token/log-prob
对齐有断言；重试时重置陈旧 rollout 状态；per-turn `max_new_tokens` 有上限；只用文档
里的 `--custom-generate-function-path` 与 `--custom-rm-path` 钩子。

## Reference answer

### `issue` — the diagnosis

根因目标：`reward_func` 里无条件的长度奖励项。

修复目标：删掉长度比例项，让奖励只由正确性加有界的工具 shaping 决定。

必须判不过的：把 verdict 翻成 healthy、把上涨的 `eval/aime` 当成 held-out 提升、或把
失败归给某个"不寻常但无害"的设置。

### `solution` — 补丁

删除长度项。删除后 `generate_with_retool.py` 与 `slime_retool` 的副本逐字节相同。

## Verification

`issue` 权重 0.2 / 0.6 / 0.2，通过线 0.8。

`solution`：修好后跑 `examples/retool/retool_qwen3_4b_rl.sh`，固定
`--num-rollout 70` / `--eval-interval 10`，得到 0, 9, …, 69。verifier 在 math_dapo
**score 空间** `[-1, 1]` 读 `eval/aime`（冻结 trace 记录的空间，`eval_reward_key:
score`），取 best，必须落在 `[-0.55, 0.5]`。

评这个通道才是判别量，因为缺陷就在里面：长度奖励还在时 `eval/aime` 会离开
`[-1, 1]`（历史上 step 0 的 +0.822 到 step 99 的 +3.841），每个 eval 点都触发上界；
删掉后它是正确性信号，不可能超过 +1。

两个界都对着修复参考运行标定（同配方去掉长度项、同镜像同资产、`--num-rollout 70`，
devspace `1177503:/user/chenyinghao/harbor-runs/manual-slime-retool`）：step 0 为
-0.5808，不过下界；最佳点在 step 59，-0.4246（余量 0.125）；最后一点 step 69 为
-0.4861。

**不使用准确率门槛。** `reward_func` 的错误地板是 `min(-0.6, …)`，均值分数无法换算成
准确率，且修复配方的准确率从未被测过。测了哪个通道就评哪个通道。

## Running it

镜像 `slime-v0.3.2`（`docker/images/slime-v0.3.2.Dockerfile`），与同一配方的
`slime_retool` 共用。没有按任务的 Dockerfile；`prepare.sh` 负责取资产。

`prepare.sh` 拉取与 `slime_retool` 相同的四份资产：DAPO-Math-17k jsonl、AIME-2024
jsonl、SFT 后的 HF checkpoint、以及它的 torch_dist 转换。**两个文件刻意保持一致**：
本任务是 `slime_retool` 的注入缺陷孪生，两者唯一的差别必须就是这个缺陷。改一份就改两份。

这条规则同样适用于配方本身。`workspace/examples/retool/` 与 `slime_retool` 的副本只差
`reward_func` 里那两行长度奖励；`retool_qwen3_4b_rl.sh` 与孪生逐字节相同。其余任何
改动要么向 agent 泄漏缺陷，要么破坏这一对存在的一曲线对比。

`workspace/wandb/` 与 `workspace/outputs/` 下 agent 会读的冻结证据也一样，已清除缺陷
痕迹：W&B group 是 `qwen3-4B-retool-multi-turn`（不是 `…-reward-hack`），记录的
`--save` 目标是 `qwen3-4b-sft-multi-turn`（不是 `…-hack`，那个还与 `config.yaml` 自己
的 `save:` 值矛盾），记录的主机路径和回执 `task_id` 写的是 `slime_retool` 而不是
`slime_retool_length_hack`。只有隐藏的 `evaluations/` 树和本文件点明它是缺陷孪生。

`evaluations/solution/solution/solve.sh` 是 `harbor run -a oracle` 的参考运行：删掉长度
奖励、套用 instruction 的预算（70 rollouts、不存 checkpoint），并把日志 tee 到
`experiment_logs/`。

workspace 代码与 ReTool 配方 Apache-2.0。
