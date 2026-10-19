# torch_functional

| | |
| --- | --- |
| **Author** | benchmark_evaluation |
| **Domain** | `infrastructure` |
| **Tags** | `verl` `log-prob` `causal-shift` `fsdp` `no-padding` `implementation-defect` |
| **Framework** | `verl`（FSDP / NO_PADDING 路径） |
| **Model** | Qwen2.5-1.5B-Instruct |
| **Dataset** | TACO_EASY（8368 train / 184 test） |
| **Environment** | `verl` 镜像；数据集与模型按 HF revision 钉死 |
| **Resources** | 8 × H100 |
| **Recorded runs** | 注入缺陷的一次 400 步崩溃运行；修复侧一次同栈参考运行 |
| **Recorded runtime** | 400 步 |

## Task abstract

`prepare_model_inputs` 构造 log-prob 重算标签时**没有做因果移位**：
`input_ids_rmpad_rolled = input_ids_rmpad`，而不是
`torch.roll(input_ids_rmpad, shifts=-1, dims=1)`。这批未移位的标签同时喂给
`logprobs_from_logits` 和融合 kernel 的 `shift_labels`；前者只做 log-softmax + gather，
模型 logits 也是标准的（`logits[i]` 预测 `token[i+1]`），于是每条重算出的回复
log-prob 都错位一位。后果：`rollout_corr/kl` 稳定在 20–22，`val-core` 在 150 步内从
0.071 掉到 0，训练奖励归零，回复长度从约 168 爆到约 2400。

## The defect

`verl/workers/engine/fsdp/transformer_impl.py::prepare_model_inputs`：

```diff
- input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
+ input_ids_rmpad_rolled = input_ids_rmpad
```

这是线上 FSDP/NO_PADDING 的 log-prob 路径：`old_log_prob`、`ref_log_prob`、actor
update 的 `log_prob` 都走它。

### Why nothing catches it

- **错位一位不报错。** 形状、dtype、长度全部合法，只是一个数值全错的向量。
- **只有 KL 和下游指标会说话。** 没有断言、没有异常，`rollout_corr/kl` 是唯一早期
  信号。
- **修复点容易找错。** 两个更早的框架都被推翻：
  `log_probs_from_logits_response` 在 verl 0.9.0.dev 里是死代码（零调用点）；
  `no_padding_2_padding` 的切片也只是**消费**一个已 roll 的数组，改动它无法破坏或
  恢复训练。devspace 上的 4-way 对照（Qwen2.5-1.5B）确认：只有
  `roll + no_padding_2_padding[-1]-slice` 能复现教科书式 log-prob（平均 |Δ| = 0.0000），
  去掉 roll 则 |Δ| ≈ 10 nats，与 padding 切片无关。

## Evidence

`workspace/logs` 里放着这次缺陷的真实 400 步崩溃运行
（`qwen2_5_1.5b_instruct_taco_fsdp_20260910_040027.log` 及配套的 offline W&B
`run-xr5gf82b.wandb`）。它跑的是线上 FSDP/NO_PADDING 路径
（`strategy=fsdp`、`use_remove_padding=True`、`_v1=False`）且 roll 被去掉：

- `rollout_corr/kl` 全程 ≈ 20–22
- `val-core/taco/acc/mean@1` 在 150 步内 0.071 → 0.0
- `critic/rewards/mean` 归零
- `response_length/mean` 从约 168 膨胀到约 2400

这是给 agent 的诊断证据，不是健康基线。

## Reference answer

### `issue` — the diagnosis

根因目标：`prepare_model_inputs` 里被丢掉的因果 roll。

修复目标：恢复
`input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)`。

必须澄清的归因：

- `padding.py::no_padding_2_padding` 保留健康的左移切片——它是已 roll 数组的消费者，
  不是缺陷
- `taco.py` 与健康 scorer 一致，不构成因果
- parquet、启动预算、1.5B checkpoint 都是固定的

### `solution` — 补丁

把 roll 加回来，其余不动。

## Verification

`issue` 比对 `agent_answer.md` 与隐藏 `oracle.md`，权重 0.2 / 0.6 / 0.2，通过线 0.8。
本质根因就是丢掉的 roll。

`solution` 按加权计分：

- validation：`val-core/taco/acc/mean@1` 的有限峰值 ≥ `0.10` 记 1
- training reward：训练步上 `critic/rewards/mean` 的均值严格 > `0.10` 记 1
- `score = 0.8 * validation + 0.2 * training reward`

门槛对着本引擎上的修复参考运行标定（devspace、8×H100、2026-10-05）：roll 恢复后
`val-core` 从 step-0 的 0.054 涨到 step 150 的峰值 0.125（终值 0.109），
`critic/rewards/mean` 平均约 0.15–0.20，`rollout_corr/kl` 保持约 0。roll 去掉时
两者都归零。`0.10 / 0.10` 落在两个 regime 中间。

### 重新定向记录（2026-10-05）

先前两版定位在本引擎上是惰性的，已回退：原始注入点在
`torch_functional.py::log_probs_from_logits_response`（死代码），第二次改到
`no_padding_2_padding`（只是消费者）。缺陷现位于线上 `prepare_model_inputs` 的 roll，
`torch_functional.py` 与 `padding.py` 已恢复上游健康形态。一次 roll 复原的完整 400 步
训练跑得健康（`rollout_corr/kl` ≈ 0，val 峰值 0.125）；去掉 roll 则 val 与奖励归零。

## Running it

`metadata.environment.image: verl` → `docker/images/verl.Dockerfile`
（`verlai/verl:vllm017.latest` 按 digest 钉死）。

启动脚本：`examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh`。指定实验
8 卡、400 步、大 shared-memory。

数据仓库 `LIF1014/ptdbench-verl-implementation-torch-functional-dataset`，revision
`83da7ccf4a74543f5aa3e46191c4916d7029564b`，Apache-2.0，与 `task_function_call` 同一份
TACO EASY 快照：

- `workspace/taco_verl/test.parquet`，61,082,676 bytes，sha256 `e364fa69d973f8e8644b2fa4d7dc6ec405d5b4adff64002c6ad52e7c787c6eaa`
- `workspace/taco_verl/train.parquet`，606,016,648 bytes，sha256 `7f25abdfa9e0f5baa173d153a32973fbe2cc8736503a35408b8ebb8ae2bc5706`

模型仓库 `LIF1014/ptdbench-verl-implementation-torch-functional`，revision
`36a4c226ef70aef2c38df173a7cf86a47ad3ad00`，Qwen2.5-1.5B-Instruct，`model.safetensors`
sha256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`（与 `task_function_call` 的字节相同）。

镜像在 build 时拉取不可变 revision、校验产物，只在 agent 阶段 hydrate workspace。

用 Docker agent backend；local backend 不构建镜像、也不 hydrate 外部资产。

workspace 代码与 verl 快照 Apache-2.0。
