# verl_missing_flash_attn_patch

| | |
| --- | --- |
| **Author** | benchmark_evaluation |
| **Domain** | `infrastructure` |
| **Tags** | `llama` `dapo-math` `monkey-patch` `flash-attn` `zero-advantage` `verl` |
| **Framework** | `verl` / GRPO |
| **Model** | Llama-3.2-1B-Instruct |
| **Dataset** | DAPO-Math-17k |
| **Environment** | `verl` 镜像；数据集按 HF revision 钉死 |
| **Resources** | 2 GPUs |
| **Recorded runs** | 缺陷 / 修复各一次，2026-10-03 同节点 200 步 |
| **Recorded runtime** | 约 29 min（200 步） |

## Task abstract

被重写的 `apply_monkey_patch` 不再安装基线里的 flash-attention hook，held-out 比同集群
零差对照低 43%（0.0389 vs 0.0687），70.5% 的训练步是零优势。同时
`TRAIN_BATCH_SIZE=4` 配约 3–5% 的稀疏 0/1 奖励，使大部分 GRPO 组全错、优势为零 —
两者**各自都足以**解释曲线。修复只恢复 monkey patch 即可过 solution 档。

## The defect

`verl/models/transformers/monkey_patch.py::apply_monkey_patch` 被重写，丢掉了
`use_remove_padding` 或 SP>1 时使用的 flash-attention hook。

同一份 run 里还有第二个可选归因：
`examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh::TRAIN_BATCH_SIZE=4`。

### Why nothing catches it

- **丢了 hook 不报错。** 模型仍能前向，只是走的不是基线路径。
- **曲线低得不夸张。** held-out 停在 3–4%，看起来像"难任务 + 小模型"，而不是坏的
  patch。
- **两个原因纠缠。** 200 步里有 59 步 `actor/grad_norm` 为零，零优势步占 70.5%；
  单看 batch size 也能解释，所以诊断要能分辨。
- **`repeat_kv` 不是原因。** 本 run 用 SP=1，该 helper 不在执行路径上。

## Evidence

- val-core 末三点均值 0.0389；比同集群零差对照 0.0687 低 43%
- `actor/entropy` 尾部 2.98（对照 1.36）；策略几乎不收敛
- `response_length/mean` 376（对照 563）
- 零优势步 70.5%（对照 56%）；有奖励步 29.5%（对照 44%）
- `TRAIN_BATCH_SIZE=4`、`ROLLOUT_N=8`、200 步；held-out
  `val-core/math_dapo/acc/mean@1` 停在约 3–4%（终值 0.0402），多数训练步梯度为零
  （59/200 步 `actor/grad_norm` 非零）

被保留的源指标是 `val-core/math_dapo/acc/mean@1`；冻结 workspace 是证据包，完整重跑
不在 verifier 范围内。

### Held-out results

同节点、同 2 卡、同 200 步、同 `TRAIN_BATCH_SIZE=4`、同数据集摘要（test split 1807 行），
两臂只在 `monkey_patch.py` 上不同：

| 臂 | step 200 | 峰值 |
| --- | ---: | ---: |
| 暂存缺陷（未装 flash-attn hook） | 0.04759 | 0.06862 @100 |
| oracle 修复（装上 hook） | 0.07360 | 0.07471 @180 |

协议基线（来自 source、**未在当前栈复现**）：健康 run 在 200 步到约 8.4%，1000 步到
约 11.8%。这两个数不要用来推门槛——2026-10-03 的 oracle 修复在本栈上 step 200 只到
7.36%（峰值 7.47%）。

## Reference answer

### `issue` — the diagnosis

两个根因目标，**任一个都是正确归因**：

- `monkey_patch.py::apply_monkey_patch`（恢复完整基线，含 flash-attn hook）
- `run_llama3_2_1b_instruct_dapo_math_fsdp.sh::TRAIN_BATCH_SIZE`（增大）

`repeat_kv` 的布局不是被评分的根因（SP=1，不在执行路径）。

### `solution` — 补丁

恢复 `apply_monkey_patch` 完整基线。**这一项单独就足够过 solution 档**——清过门槛的
标定 run 只做了这一个修复，没有动 `TRAIN_BATCH_SIZE`。

## Verification

`issue` 权重 0.2 / 0.6 / 0.2，通过线 0.8。两个根因各自独立充分，点出其一并给出正确的
因果即可拿满 `root_cause_score`。

`solution`：修好后跑指定启动脚本，step 200 的有限
`val-core/math_dapo/acc/mean@1` ≥ `0.060`。只评 step 200。

`0.060` 取两臂中点，比缺陷臂高约 2.2 SE、比修复臂低约 2.4 SE。旧协议评
**所有步的峰值**、门槛 0.07——未修复的 workspace 峰值 0.06862 = 124/1807，差三个
正确答案就能过，而且延长 run 或提高 `TEST_FREQ` 就能在没修的情况下抬高分数。
`historical_final_baseline` 也从 `0.04022346368715084`（72/1790，已废弃的 test split）
改为缺陷臂实测的 0.04759（86/1807）。

**未决：** 修复臂只有一次运行，step-200 的 run-to-run 方差未测；再跑一对（各约 29 min）
能把门槛钉牢。

## Running it

`metadata.environment.image: verl` → `docker/images/verl.Dockerfile`
（`verlai/verl:vllm017.latest` 按 digest 钉死）。

数据仓库
`LIF1014/ptdbench-llama-dapo-implementation-task-monkey-patch-011-dataset`，revision
`c3a84bacaadea32cc7a2b9d81162678ac2cb684c`，Apache-2.0：

- `workspace/DAPO-Math-17k_verl/test.parquet`，675,271 bytes，sha256 `f860d1c3bfb0ce4a937b5e91bb43549a0a7ded51b1725c58c9c10f9f075d2aeb`
- `workspace/DAPO-Math-17k_verl/train.parquet`，5,885,886 bytes，sha256 `2897263b99e9b4af7523edeb5ec1ab9cac6e1bba63f0bfa78d4cd9356426ca5b`

镜像在 build 时拉取 revision、校验产物，只在 agent 阶段 hydrate workspace。

2 卡，由 `metadata.yaml` 的 `environment.env` 声明。

### agent 可见面上不得出现叶子 task id（2026-10-15 修复）

叶子 id `task_monkey_patch_011` 直接点出缺陷文件
（`verl/models/transformers/monkey_patch.py`）。两个面带着它：

- **`workspace/` → `/workspace`**：16 个文件、131 处——冻结训练日志、每个 `wandb/`
  的 `config.yaml` / `output.log` / `debug.log`、`outputs/2026-07-20/` 下的 hydra
  配置、以及 5 个 `outputs/*.json` 回执，都带
  `trainer.experiment_name=task_monkey_patch_011` 或含它的源路径。全部改为
  `__BENCH_TASK_ID__`。这不是随便的占位符：启动脚本默认
  `EXPERIMENT_NAME=${EXPERIMENT_NAME:-${BENCH_TASK_LEAF_ID:-__BENCH_TASK_ID__}}`，
  没有 harness 代码设置 `BENCH_TASK_LEAF_ID`，所以 agent 新跑的 run 恰好输出同一个串，
  快照与它自洽。
- **`environment/` → `/ptdbench/environment`**：按设计只读挂进 agent 阶段。本任务的
  `prepare.sh` 在注释里违约：头部有叶子任务路径、legacy HF 仓库名
  `…-task-monkey-patch-011-dataset`，更糟的是引用了
  `evaluations/solution/verifier/resources/grader_spec.json`、它的
  `historical_final_baseline` 和 `required_final_score "0.07"` 的通过线。全部删除；
  可执行内容逐字节不变（已 diff 非注释行核对）。

任何重新生成之后复查：

```bash
grep -rIilE "monkey_patch_011|monkey-patch-011|required_final_score|grader_spec" \
  workspace environment evaluations/*/instruction.md
```

必须没有输出。两处近似命中要保留：符号 `apply_monkey_patch` 是真实的 verl 源码，
`0.04022346368715084` 是历史 run 自己记录的指标，也就是 agent 该读的证据。

未改的 harness 级问题：`TaskEvaluator._run_directory_prefix` 把 task id 放进**宿主**的
run 目录名。Docker backend 下不可见（agent 只看到 `/workspace`、`/task`、
`/results`），local backend 下会可见——本任务不用 local backend。

### 从 `prepare.sh` 归档

数据集 recipe 与兄弟 DAPO-Math 任务共享
（`docker/ptdbench_datasets/dapo_math_17k.py`）；上游 parquet 是 17917 题重复 100 次。

legacy Dockerfile 会从上面那个 HF 数据仓库拉
`artifacts/workspace-assets.tar.gz`，revision `c3a84bacaadea32cc7a2b9d81162678ac2cb684c`，sha256 `e6ff5139b9fd4a4c9cca8d29d1d50faaea2220098e9d97829a737fb50fd314db`
（5,676,512 bytes），解包到 `/opt/ptdbench-assets` 再由 `hydrate-assets.sh` 灌进
`/workspace`。该 tarball 从未在本地下载，内容也从未核验。
