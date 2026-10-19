# task_019

| | |
| --- | --- |
| **Author** | benchmark_evaluation |
| **Domain** | `configuration` |
| **Tags** | `rlve_hyper` `hyperparameter` `lr-warmup` `schedule` `slime` `grpo` `decoy` |
| **Framework** | RLVE / slime + GRPO |
| **Model** | Qwen2.5-0.5B-Instruct |
| **Dataset** | RLVE held-out environments |
| **Environment** | `slime-rc0`；数据集资产按 HF revision 钉死 |
| **Resources** | 4 × H100 |
| **Recorded runs** | 缺陷 / 修复各一次，2026-10-15 端到端重跑 |
| **Recorded runtime** | 修复侧 5h28m（约 65 min / 20 步） |

## Task abstract

`--lr-warmup-iters=2000` 让 100 步的预算全部耗在 warmup 里：学习率停在 1e-7，永远
到不了 2e-6 的基线，eval 提升低于健康 p10。同一条命令里的 `--lr-decay-iters=10000`
是**非因果诱饵**——在 `--lr-decay-style constant` 下调度器 warmup 一结束就直接返回
`max_lr`，从不读 `lr_decay_iters`。step 99 上缺陷侧 -0.5321，修复侧 -0.1021。

## The defect

`scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh`：

- `--lr-warmup-iters 2000`（根因，必须降到 0 或很小的值）
- `--lr-decay-iters 10000`（诱饵，在 constant 衰减下完全惰性）

### Why nothing catches it

- run 正常完成、数值有限，只是"涨得少"，看起来像预算不足或数据难。
- 诱饵和真因写在同一个脚本里，**看着同样可疑**。删掉诱饵不改变任何行为；不改 warmup
  则什么都没修好。
- 判据要落到实际学习率上：`train/lr-pg_0` 在 100 步内只到 1e-7。

## Evidence

- `train/lr-pg_0` 最终只有 1e-7，从未到达基线 2e-6（2000 步 warmup 里只走了 100 步）
- raw_reward -0.599 → -0.531（+0.068）
- eval -0.620 → -0.522，提升低于健康 p10

被保留的源指标是 `eval/HELD-OUT_ENVIRONMENTS_128`；冻结 workspace 是证据包。

### Held-out results

两侧在同一 devspace（`1177503`）上重跑过本暂存 workspace；两棵树仅差 `rlve.sh` 里的
oracle 修复，其他跟踪文件 md5 全同。

| step | 缺陷侧（warmup 2000） | 修复侧（warmup 0） |
| ---: | ---: | ---: |
| 0 | -0.6195 | -0.5767 |
| 19 | -0.6149 | -0.2533 |
| 39 | -0.5635 | -0.2384 |
| 59 | -0.6305 | -0.1355 |
| 79 | -0.6525 | -0.1419 |
| **99** | **-0.5321** | **-0.1021** |
| 平均 `train/lr-pg_0` | 5.05e-8 | 2.0e-6 |

缺陷侧 step 99 与冻结的全量 `eval_last`（-0.5221）吻合，说明缩小规模的暂存仍可用
历史基线。两臂在 step 0 评的是同一个 checkpoint，差 0.042，据此单个 eval 点的 sigma
约 0.03。

健康协议基线：起点约 -0.62 ± 0.03，eval 提升至少达健康 p10 的约 +0.21。

## Reference answer

### `issue` — the diagnosis

- `--lr-warmup-iters`：**唯一**本质根因，降到 0 或任意合理的小 warmup。
- `--lr-decay-iters`：**非本质**，constant 衰减下惰性，可选清理。**漏掉它不得扣分**；
  正确论证它惰性的答案应得分。

期望数值只是参考，judge 可以接受任何能恢复到基线学习率的合理值。

弱 reward/eval 移动是学习率调度被截断的症状，不是独立根因。

### `solution` — 补丁

改 `rlve.sh` 的 `--lr-warmup-iters` 为 0，跑完 100 步。

## Verification

`issue`：权重 0.2 / 0.6 / 0.2，通过线 0.8。`--lr-warmup-iters` 是唯一本质根因，点出它
并说明因果即可拿满 `root_cause_score`。

`solution` 按加权计分：

- validation：step 99 的 `eval/HELD-OUT_ENVIRONMENTS_128` 必须存在且落在
  `[-0.40, 0.30]`，记 1
- learning rate：`train/lr-pg_0` 必须跨至少 95 步被记录、step 99 存在，且均值 ≥ `8e-7`
- `score = 0.8 * validation + 0.2 * learning rate`

2026-10-15 重新标定，改了什么以及为什么：

- **评分点从"任意步的峰值"移到 `final_step = 99`。** 旧规则会让一个在 eval 39 就被
  截断的修复运行拿 1.0（其 -0.2384 以 0.0016 越过旧门槛 -0.24），等于 60% 的算力
  预算可以省掉；峰值还会随 `--num-rollout` 增长，固定步索引不会。
- `final_step` 从 100 改为 99。任何日志里都没有第 100 步
  （`--num-rollout 100` 给出 0..99，`--eval-interval 20` 给出 0/19/39/59/79/99），
  旧值一旦被 key 上就会把完美运行判 0。
- `required_final_score` 从 -0.24 改为 -0.40，取缺陷侧 step 99（-0.5321）与修复侧
  训练后最差点（step 19 的 -0.2533）的中点 -0.393，两边各约 0.13 余量。修复侧训练后
  的点会摆动 0.15，所以门槛对着它最弱的 checkpoint 而非 step 99 设。
- 新增 `required_final_score_max = 0.30`。RLVE reward 有界于 `[-1, 1]`，修复侧收在
  -0.102，而 grader 本来接受 `experiment_logs` 下任意日志里的任意行——伪造一行
  `eval 99: {... 0.9}` 曾能通过。
- 新增预算覆盖：要求 step-99 的 eval 点与至少 95 个 `train/lr-pg_0` 步。

**残留且已接受：** eval 与 lr 点仍按 step 在 `experiment_logs` 下所有日志间取每步
最大值合并，所以重复运行在 step 99 上是 best-of-N，而 `oracle/task.yaml` 的
`max_runs: 1` 在容器内没有被强制执行。lr 分量也可以靠在保留 warmup 的同时抬高
`--lr` 满足（均值恢复、constant 调度没恢复），但那样的 run 仍须过 step-99 eval 门槛。
部分修复（如 `--lr-warmup-iters 100`）会让均值 lr 约 1e-6、过 lr 分量，其 step-99
eval 可能落在两臂之间——**尚未测量**，位置相对 -0.40 未知。

## Running it

`metadata.environment.image: slime-rc0` → `docker/images/slime-rc0.Dockerfile`。

数据集资产：

- repo `LIF1014/ptdbench-rlve-hyper-task-019-dataset`，已发布
- revision `235103f0bdca99ca0c86da0b79ac1de92acb52c5`，MIT
- `workspace/data/HELD-OUT_ENVIRONMENTS/test_128.json`，110,359 bytes，
  sha256 `3ebd3497305f75cb0712c4a7dd5105b3af0c32b1cdc85f2bdf1a6913e6db4018`
- `artifacts/workspace-assets.tar.gz`，9,207,578 bytes，sha256 `d7164e8c27586efae4daeb972f858092390c475938fcb848e213612bb4190a9e`

`timeout_minutes` 只作文档：harbor 不产生 `[agent] timeout_sec`，agent 阶段不限时。
实测修复侧重跑 5h28m，source 的 180 不准。

### 任务名脱敏（2026-10-15 审计）

agent 实际能读的是 `workspace/`、`instruction.md` 和 `environment/`
（docker backend 挂在 `/ptdbench/environment`，cybertron backend 挂在
`/workspace/.ptdbench-environment`）；`metadata.yaml`、本文件、
`evaluations/*/oracle/`、`evaluations/*/verifier/` 在 agent 阶段都不挂载，而
`PTDBENCH_TASK_ID` 只是叶子 id。

修掉的三处：

- `environment/prepare.sh:2` 以 `# rlve_hyper/task_019` 开头，点明了"缺陷是超参数"。
  现为 `__BENCH_TASK_ID__`，并注明该文件 agent 可见。顺带清掉了一处按名字引用兄弟任务
  的交叉引用。
- `workspace/historical_evidence/manifest.json` 带着
  `source_run: run_48_warmup_iters_2000_lr_warmup_decay=10000`，把两个 buggy key 和
  它们的值都写了出来。该字段从 agent 可见副本中删除（`visible_manifest`），保留在
  `evaluations/solution/oracle/historical_evidence_manifest.json` 作溯源。
- `workspace/scripts/.../rlve.sh:14` 的 `RUN_NAME` 默认值是 `rlve_task_019`，会给
  agent 自己的 wandb group、run 目录和日志路径打标签。现为
  `${2:-${BENCH_TASK_LEAF_ID:-__BENCH_TASK_ID__}}`，没有 harness 代码设置
  `BENCH_TASK_LEAF_ID`，新 run 会输出占位符。

**刻意保留、不算泄漏：** `historical_evidence/{manifest,hparams_effective}.json` 里的
`lr_warmup_iters: 2000`。`rlve.sh` 本身就传 `--lr-warmup-iters 2000`，冻结的 hparams
只是复述一个可见配置值——那正是 issue 档期望 agent 找到的证据。

**未决：** `convert_rlve_hyper.py` 会从不在本仓库的 `Benchmark_Evaluation` 树重新
复制 `rlve.sh` 和 `environment/`，完整重转换会重新引入 `RUN_NAME` 默认值与
`prepare.sh` 头部；只有 `source_run` 的删除由转换器自己处理。

`harbor_evaluator.py` 会把 `metadata.yaml` 的 title 写进 `task.toml` 的
`[task] description`；Harbor 用 `instruction.md` 而非 description 构造 agent prompt，
故未按容器内泄漏处理，但这是所有任务共有的 harness 级表面。
