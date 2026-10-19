# verl_aime25_single_sample_protocol

| | |
| --- | --- |
| **Author** | 王嘉泽 |
| **Domain** | `configuration` |
| **Tags** | `ppo-critic` `evaluation-protocol` `sampling` `aime25` `false-alarm` `stdlib-only` |
| **Framework** | `verl` PPO-Critic checkpoint，经 `run_eval.py` 评测 |
| **Model** | `Qwen3-4B-Base` |
| **Dataset** | AIME25（30 题） |
| **Environment** | `ppo-critic-vimpo`；`prepare.sh` 是已记录的空操作 |
| **Resources** | 无 GPU 需求（评分只用标准库） |
| **Recorded runs** | 同一 checkpoint 的 1-sample 与 32-sample 两轮重复评测 |
| **Recorded runtime** | 秒级（protocol check） |

## Task abstract

AIME25 表现出的"失败"是**评测协议采样不足**造成的假警报，不是 PPO 训练失败。
`config/eval_config.yaml` 设了 `evaluation.samples_per_question: 1`——在只有 30 题的
基准上，用 temperature 1.0 / top-p 1.0 采样一次。对**同一个 checkpoint** 做受控重复
评测，每轮 AIME25 准确率在 **6.67% 到 20%** 之间摆动，其中 **10% 出现在 32 轮里的
7 轮**。单次 30 题的实现是一个高方差估计量。

## The defect

```yaml
# config/eval_config.yaml
evaluation:
  samples_per_question: 1
```

`run_eval.py` 用 `--protocol-only` 报告当前协议（`expected_generations: 30`）。

### Why nothing catches it

- **没有任何训练或模型缺陷。** 坏的是测量协议，代码和权重都是好的。
- **一次低分看起来完全像训练失败。** 30 题上单次采样，运气差就低。
- **题量小放大了噪声。** AIME25 只有 30 题，单题波动直接是 3.3 个百分点。
- **随机解码。** `temperature: 1.0`、`top_p: 1.0` 让同一 checkpoint 每次结果都不同。

## Evidence

`workspace/ptdbench_case/original_eval/` 下保留了受控证据：

- `n1/`——每题一次随机采样，单个 30 题的实现
- `n32/`——每题 32 次采样，`notes/n32_epoch_breakdown.txt` 记录逐轮分布
  （6.67%–20%，10% 出现在 7/32 轮）
- `notes/experiment_design.txt` / `notes/final_comparison.txt`——同一 checkpoint 上
  1-sample 与 32-sample 协议的受控对比

checkpoint 本身是大型外部资源，不随包。

## Reference answer

### `issue` — the diagnosis

根因目标：`config/eval_config.yaml` 的 `evaluation.samples_per_question: 1`。

修复目标：把 `samples_per_question` 提到 `32` 并用 `mean` 聚合，保留 30 题基准与随机
生成（`temperature: 1.0`、`top_p: 1.0`）。修复后 `run_eval.py --protocol-only` 应报告
`samples_per_question: 32`、`expected_generations: 960`。

必须判不过的归因：PPO 训练失败、奖励缺陷、优化器问题。

### `solution` — 补丁

```diff
-  samples_per_question: 1
+  samples_per_question: 32
```

reducer 用 `mean`。

## Verification

`issue` 比对 `agent_answer.md` 与隐藏 `oracle.md`，权重 0.2 / 0.6 / 0.2，通过线 0.8。

**`solution` 是确定性的协议检查，不是指标评分——这是刻意保留的例外。** 与另外三个
ppo_critic 任务不同，本任务修的是**评测配置**，没有训练可重跑，也没有
`val-core` / `critic/rewards` 可重建。`evaluations/solution/verifier/test.sh` 改为：

1. 在修复后的 workspace 跑 `run_eval.py --protocol-only`，断言生效协议为
   `task == aime25`、`questions == 30`、`samples_per_question == 32`、
   `reducer == mean`、`temperature == 1.0`、`top_p == 1.0`、
   `expected_generations == 960`；
2. 用可信清单 `oracle/original_eval.sha256` 做 `sha256sum -c`，验证
   `workspace/ptdbench_case/original_eval/` 逐字节未变。

检查是精确的且完全可离线运行（无 GPU、无网络、无模型），因此**不需要未标定的门槛**。
`oracle/` 里带修复后的 `eval_config.yaml`、`expected.md` 和 `original_eval.sha256`。
verifier 读取 `$PTDBENCH_ORACLE`、`$PTDBENCH_WORKSPACE`、`$PTDBENCH_RESULTS`。

## Running it

`metadata.environment.image: ppo-critic-vimpo` → `docker/images/ppo-critic-vimpo.Dockerfile`，
与三个 ppo_critic 训练任务同一个镜像。本任务借它只是为了一个 Python 解释器：两个评分
步骤（`run_eval.py --protocol-only` 与 `sha256sum -c`）只用标准库，从不碰
rollout/eval 栈，受控证据随包在 `workspace/ptdbench_case/original_eval/`。

`environment/prepare.sh` 因此是**有记录的空操作**（不下载任何东西，并写明原因）：
两个评分步骤都不加载模型，证据也已内嵌。

可选的全量 AIME25 重跑通过 `CHECKPOINT_PATH` 接收一个外部 PPO-Critic checkpoint——那是
训练出来的 checkpoint 而非可下载的 HF 基座模型——由操作者提供，评分不依赖它。

**记录中的 AIME25 评测是在另一套栈上产生的**（CUDA 12.6.3 / torch 2.7.1 /
vLLM 0.10.1.1 + inspect-ai + lighteval，钉在 bundle 的
`tasks/PTDBench_candidate_bundle_20260821/…/environment/` 下）；那套栈刻意**不**在这里
重建，因为没有 checkpoint 随包、评分也只用标准库。

workspace 代码与 verl 快照 Apache-2.0，任务打包 MIT。
