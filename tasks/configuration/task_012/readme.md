# task_012

| | |
| --- | --- |
| **Author** | benchmark_evaluation |
| **Domain** | `configuration` |
| **Tags** | `rlve_hyper` `hyperparameter` `hidden-dropout` `kl` `slime` `grpo` |
| **Framework** | RLVE / slime + GRPO |
| **Model** | Qwen2.5-0.5B-Instruct |
| **Dataset** | RLVE held-out environments |
| **Environment** | `slime-rc0`；数据集资产按 HF revision 钉死 |
| **Resources** | 4 × H100 |
| **Recorded runs** | 每侧一次端到端，另有一次未修复重跑 |
| **Recorded runtime** | 100 rollouts / 4 卡 |

## Task abstract

`--hidden-dropout=0.5` 让训练**冻结**：raw_reward 在 -0.604 原地不动，同时熵（6.00 →
6.26）和 PPO KL（12.3）远高于健康运行。grad_norm 从 264452 掉到 727。held-out
停在 -0.614，修复侧在 step 99 到 -0.166。

## The defect

`scripts/training/Qwen2.5-0.5B-Instruct/rlve.sh` 里的 `--hidden-dropout` 设为 `0.5`，
应降到 `0`。

### Why nothing catches it

- run 正常完成，没有异常，没有 NaN。
- reward 曲线是**平的**，不是崩的——"不涨"比"跌"更像数据或预算问题，容易误判。
- 极端的 KL 和 grad_norm 是 hidden dropout 0.5 的**症状**，把它们当成独立根因会
  偏离真正的开关。

## Evidence

- raw_reward -0.604 → -0.604，完全不动
- `train/entropy_loss` 6.00 → 6.26；健康上界 1.40
- `train/ppo_kl` 12.3；健康运行约 0
- `train/grad_norm` 264452 → 727

被保留的源指标是 `eval/HELD-OUT_ENVIRONMENTS_128`。冻结 workspace 是证据包，完整
重跑不在 verifier 范围内。

### Held-out results

| step | 未修复重跑 | 修复侧 |
| ---: | ---: | ---: |
| 99 | -0.5898 / -0.6910 | -0.1661 |

修复侧的尾部准确率 0.3495，未修复重跑 0.047。两个门槛都落在这两个群体之间，不是对
某个换算过或历史上的数字。

健康协议基线：同一组 RLVE held-out 环境、健康运行下起点约 -0.62 ± 0.03，eval 提升
至少达到健康 p10 的约 +0.21。本 case 预算 100 rollouts。

## Reference answer

### `issue` — the diagnosis

根因目标：`--hidden-dropout`。修复目标：降到 `0`。

极端 KL 与 grad_norm 是症状，不是独立根因。

### `solution` — 补丁

把 `rlve.sh` 的 `--hidden-dropout` 改为 `0`，跑完指定脚本。

## Verification

`issue` 比对 `agent_answer.md` 与隐藏 `oracle.md`，权重 0.2 / 0.6 / 0.2，通过线 0.8。

`solution` 按加权计分。**评分绑定到单次 launch**：verifier 选出同时报告 held-out eval
和离线 W&B 目录的最新文本日志，两个分量都只从这一次 launch 读。因此日志根目录里留
一个调试用的小 run 会被忽略，而不是被并进分数。

- validation：最终 eval step 的 `eval/HELD-OUT_ENVIRONMENTS_128` 必须在
  `[-0.38, 0.10]`，且最后一个 eval step 必须是 99（跑短了不算）；满足记 1，否则 0
- training reward：最后一次 launch 的 `RLVE/*/accuracy` 末 25% 的均值 ≥ `0.18` 记 1
- `score = 0.8 * validation + 0.2 * training reward`

两项都过才算通过（`score == 1.0`）；只过一项时框架仍记录加权浮点值。

文本日志不被单独采信：elected 日志报的每个 eval step，若同一 launch 的 `.wandb`
二进制流里也有，必须一致，且至少要有 `3` 个点互证。最终 eval（step 99）只进文本
日志、从不进二进制流——每个参考 run 都是如此——所以互证要求落在重叠区间而不是评分点。

门槛来自 devspace `1107990`（`/user/chenyinghao/harbor-runs/task_012__stage__env`）
的参考 run：修复侧收在 -0.1661、尾准确率 0.3495；未修复重跑收在 -0.5898 / -0.6910、
尾准确率 0.047。

## Running it

`metadata.environment.image: slime-rc0` → `docker/images/slime-rc0.Dockerfile`。

数据集资产：

- repo `LIF1014/ptdbench-rlve-hyper-task-012-dataset`，已发布
- revision `7a0f045d575b0826fb96f1cfe38c90e121d88ef0`，MIT
- `workspace/data/HELD-OUT_ENVIRONMENTS/test_128.json`，110,359 bytes，
  sha256 `3ebd3497305f75cb0712c4a7dd5105b3af0c32b1cdc85f2bdf1a6913e6db4018`
- `artifacts/workspace-assets.tar.gz`，9,207,578 bytes，sha256 `d7164e8c27586efae4daeb972f858092390c475938fcb848e213612bb4190a9e`

镜像在 build 时拉取该 revision、校验产物，只在 agent 阶段 hydrate workspace。

用 Docker agent backend。local backend 不构建镜像、不 hydrate 外部资产。模型权重与
历史 checkpoint 不是数据集资产。
