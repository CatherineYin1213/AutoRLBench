# task_001

| | |
| --- | --- |
| **Author** | benchmark_evaluation |
| **Domain** | `configuration` |
| **Tags** | `rlve_hyper` `hyperparameter` `attention-dropout` `grad-norm` `slime` `grpo` |
| **Framework** | RLVE / slime + GRPO |
| **Model** | Qwen2.5-0.5B-Instruct |
| **Dataset** | RLVE held-out environments |
| **Environment** | `slime-rc0`；数据集资产按 HF revision 钉死 |
| **Resources** | 4 × H100 |
| **Recorded runs** | 每侧一次，端到端在同一套 live protocol 下 |
| **Recorded runtime** | 标定运行 5h04m（02:29:03 → 07:33:36） |

## Task abstract

`--attention-dropout=0.5` 让 held-out eval 朝错误方向移动（-0.611 → -0.639），同时
grad_norm 从 2258 爆到 19760、熵从 4.04 掉到 3.52。健康运行在同样预算下 raw reward
仍有约 +0.36 的增益。两条曲线在 step 99 分开：修复侧 -0.170，缺陷侧 -0.634。

## The defect

`scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh` 把 `--attention-dropout` 设成 `0.5`。
修复是把该值降到 `0`。这是单一超参数缺陷，配方其余部分与健康运行一致。

### Why nothing catches it

- 训练不报错，8 项指标全部有限，run 正常退出。
- 症状（高熵、爆炸的 grad_norm）看起来像优化器或数值问题，其实是 dropout 的下游
  效应——它们是**症状不是根因**，把它们当成独立根因会分散归因。
- eval 起点本身就在 -0.6 附近，一次低分不显眼；要靠与健康运行的增益对比才看得出。

## Evidence

- runaway/raw_reward：-0.650 → -0.594（+0.056）；健康运行最差也有 +0.36
- `train/entropy_loss`：4.04 → 3.52；健康上界 1.40
- `train/grad_norm`：2258 → 19760；健康运行始终 ≤ 9.3
- `eval/HELD-OUT_ENVIRONMENTS_128`：-0.611 → -0.639（方向错误）

被保留的源指标是 `eval/HELD-OUT_ENVIRONMENTS_128`。冻结的 workspace 就是证据包；
完整训练重跑不在 verifier 范围内。

### Held-out results

两侧都在 live protocol 下端到端跑完：

| step | 缺陷侧 | 修复侧 |
| ---: | ---: | ---: |
| 99 | -0.634 | -0.170 |

门槛 `-0.50` 落在两者之间。标定表与"为何不放中点"的理由见
`evaluations/solution/verifier/resources/README.md`。

健康协议基线（沿袭自 source）：Qwen2.5-0.5B-Instruct 在同一组 RLVE held-out 环境上，
健康运行的 `eval/HELD-OUT_ENVIRONMENTS_128` 起点约 -0.62 ± 0.03，eval 提升至少达到
健康 p10 的约 +0.21。本任务预算 100 rollouts。

## Reference answer

### `issue` — the diagnosis

根因目标：

- `--attention-dropout`

修复目标：

- `rlve.sh::--attention-dropout`：降低到 `0`。

判卷时，高熵和爆炸的 grad_norm 应被认定为 dropout 的症状，而非独立根因。

### `solution` — 补丁

把 `rlve.sh` 里的 `--attention-dropout` 改为 `0`，然后原样跑指定启动脚本到 step 99。

## Verification

`issue` 比对 `$PTDBENCH_RESULTS/agent_answer.md` 与隐藏 `oracle.md`，judge 按
verdict / root-cause / evidence 打分（权重 0.2 / 0.6 / 0.2），加权 ≥ 0.8 通过。

`solution` 不评结构化诊断。修好暂存 workspace、跑完指定脚本，第 99 步必须产生有限的
`eval/HELD-OUT_ENVIRONMENTS_128`，且 ≥ `-0.50`。通过记 1，否则 0。缺 step 99 直接
失败，所以必须跑完；峰值会被报告但不参与评分。

**门槛的标定：** 两侧端到端各跑一次，修复侧 step 99 为 -0.170，缺陷侧 -0.634，
门槛取 `-0.50`。

`required_final_score` 的两个旧值都不能再用：source 里 `timeout_minutes: 90` 比实测
少约 3.4 倍（实际 5h04m），而 `-0.24` 是同族三个任务逐字共享的常量、从未由运行推导。
`timeout_minutes` 也没有任何东西强制执行：`_write_task_config()` 从不输出
`[agent] timeout_sec`，agent 阶段实际不限时，按约 5h  provisioning 即可。

## Running it

`metadata.environment.image: slime-rc0` → `docker/images/slime-rc0.Dockerfile`。

数据集资产：

- repo `LIF1014/ptdbench-rlve-hyper-task-001-dataset`，已发布
- revision `1c33b31d0632b6a5508b8f1b88d3c98f1cf4e2e7`，MIT
- `workspace/data/HELD-OUT_ENVIRONMENTS/test_128.json`，110,359 bytes，
  sha256 `3ebd3497305f75cb0712c4a7dd5105b3af0c32b1cdc85f2bdf1a6913e6db4018`
- `artifacts/workspace-assets.tar.gz`，9,207,578 bytes，sha256 `d7164e8c27586efae4daeb972f858092390c475938fcb848e213612bb4190a9e`

镜像在 build 时拉取不可变 revision、校验产物，并只在 agent 阶段把 workspace 挂进去。

用 Docker agent backend。local backend 不构建任务镜像、也不 hydrate 外部资产。模型
权重与历史训练 checkpoint 不是数据集资产。

`environment/` 以只读挂在 `/ptdbench/environment`，所以 `prepare.sh` 也不得出现任务名
或类别名。`readme.md`、`metadata.yaml` 和 `evaluations/*/oracle/` 不会被复制进
Harbor 任务目录。

### 名字泄漏审计

类别名 `rlve_hyper` 直接指向缺陷类型，不能到 agent 手里。它不会从 harness 泄漏：
`TaskPackage.task_id` 取的是叶子目录名 `task_001`，因此 `PTDBENCH_TASK_ID`、Harbor
任务名（`ptdbench/task_001-solution`）、trial 名和 compose project 名都只带数字叶子。
录制的 `agent_trace.json` 里 `hyper` 出现零次，包括 agent 启动时跑的
`env | grep PTDBENCH`。

发现并修掉的两处：

| 泄漏 | 位置 | 处理 |
| --- | --- | --- |
| `rlve_hyper/task_001` 出现在头部注释 | `environment/prepare.sh:2` | 注释不再命名任务或类别 |
| `"source_run": "run_11_attention_dropout_0.5"` | `workspace/historical_evidence/manifest.json` | 从 agent 可见副本中删除；`oracle/historical_evidence_manifest.json` 保留 |

第二处更严重：run 标签同时写出了根因和它的值，而录制里的 agent 确实读过这个文件。
转换器原本把同一个 manifest dict 写进可见路径和 oracle 路径，现在
`convert/convert_rlve_hyper.py` 让可见副本走 `visible_manifest()`。
`attention_dropout: 0.5` 仍出现在 `historical_evidence/hparams_effective.json` 和启动
脚本里——那是**预期证据，不是泄漏**。剩下的 `task_001` 字符串（`rlve.sh` 的
`RUN_NAME` 默认值、manifest 的 `task_id`）是纯数字，不指认任何东西。
