# verl_taco_difficulty_mismatch

| | |
| --- | --- |
| **Author** | benchmark_evaluation |
| **Domain** | `data` |
| **Tags** | `taco` `difficulty` `data-mismatch` `verl` `issue-only` |
| **Framework** | `verl` |
| **Model** | Qwen2.5-1.5B-Instruct |
| **Dataset** | TACO `VERY_HARD` 训练集 + TACO `EASY` 验证集（184 行） |
| **Environment** | `verl` 镜像；`prepare.sh` 拉取模型与数据归档 |
| **Resources** | 无 GPU 声明（issue-only，不跑训练栈） |
| **Recorded runs** | 一次历史训练，400/400 步跑完 |
| **Recorded runtime** | 约 8 h 41 min（78.10 s / 步） |

## Task abstract

训练集太难，且难度与验证集不匹配：`workspace/taco_verl/train.parquet` 只含 TACO
`VERY_HARD` 行，而 `test.parquet` 是既有的 TACO `EASY` 验证集。held-out 准确率因此
**低于起点**——最后一步 0.0380（7/184），step 0 是 0.0489（9/184），峰值在 step 50 的
0.0978。训练奖励从 step 201 起稳定在约 0.15，而验证准确率不再恢复。

这是 **issue-only** 任务，没有 solution 档。

## The defect

数据选择本身：训练集只取 TACO `VERY_HARD`，验证集是 `EASY`。代码、启动超参、模型都
是健康的，不构成根因。

### Why nothing catches it

- **代码是健康基线。** 没有注入缺陷，workspace 就是正常实现的快照。
- **训练奖励看起来在涨。** `critic/rewards/mean` 从 0.008 一路升到约 0.15 并稳住。
- **只有 held-out 在跌，且跌得不夸张。** 0.0489 → 0.0380 的回落很容易被当成噪声。
- **难度的因果需要看分布。** 要比较 train/test 的难度构成才能下结论。

## Evidence

日志：`workspace/logs/qwen2_5_1.5b_instruct_taco_fsdp_20261010_073442.log`。
W&B：`workspace/logs/wandb/offline-run-20261010_073718-kovw4pvq`。

run 完成 400/400 步，进度条约 8:40:39。每 epoch 145 步，`total_training_steps` 400，
所以步数上限让 job 停在第三个 epoch 中途：epoch 0 覆盖 1–145，epoch 1 从 146 起，
epoch 2 从 291 起。

held-out 为 184 行 TACO `EASY`，计数为 `round(acc * 184)`：

| Step | 全对 | acc | pass_rate |
| ---- | ---- | --- | --------- |
| 0 | 9/184 | 0.0489 | 0.1967 |
| 50 | 18/184 | 0.0978 | 0.2887 |
| 100 | 17/184 | 0.0924 | 0.2918 |
| 150 | 16/184 | 0.0870 | 0.3050 |
| 200 | 6/184 | 0.0326 | 0.1167 |
| 250 | 1/184 | 0.0054 | 0.0649 |
| 300 | 4/184 | 0.0217 | 0.1149 |
| 350 | 5/184 | 0.0272 | 0.1433 |
| 400 | 7/184 | 0.0380 | 0.1351 |

`critic/rewards/mean` 在 1–400 步平均 0.0938，前 100 步平均约 0.0082；400 步中有 96 步
为 0，最大值 0.4375（step 299），step 400 为 0.1250。五十步均值：

| Steps | 平均训练奖励 |
| ----- | ------------ |
| 1–50 | 0.0081 |
| 51–100 | 0.0083 |
| 101–150 | 0.0188 |
| 151–200 | 0.0934 |
| 201–250 | 0.1789 |
| 251–300 | 0.1467 |
| 301–350 | 0.1470 |
| 351–400 | 0.1495 |

101–150 的均值恰为 0.01875。从 step 201 起五十步均值稳定在约 0.15，而 held-out 不再
恢复。

## Reference answer

### `issue` — the diagnosis

两种表述都算完整根因：

- 训练集太难
- 训练集与验证集难度不匹配

代码、启动超参和模型都不是根因。**不需要**代码缺陷。

### `solution`

不存在（issue-only）。

## Verification

`issue` 比对 `$PTDBENCH_RESULTS/agent_answer.md` 与隐藏 `oracle.md`，judge 按
verdict / root-cause / evidence 打分，权重 0.2 / 0.6 / 0.2，通过线 0.8。只要点出
"训练集太难"或"训练与验证难度不匹配"，`root_cause_score` 即满分。

## Running it

`metadata.environment.image: verl` → `docker/images/verl.Dockerfile`
（`verlai/verl:vllm017.latest`，按 digest 钉死），与其他 verl + TACO 任务同一个镜像。

启动脚本：`examples/grpo_trainer/run_qwen2_5_1.5b_instruct_taco_fsdp.sh`。指定实验使用
8 卡、400 步预算。issue 评测不启动训练。

`environment/prepare.sh` 是唯一联网阶段，拉取模型与数据归档：

- 数据仓库 `LIF1014/ptdbench-data-task-train-too-hard-dataset`，revision
  `22a1fee0ea2527f514c4115f6c792216a99d7e6f`，Apache-2.0
- `workspace/taco_verl/test.parquet`，61,082,426 bytes，sha256 `0593078d24134d204b305dd0373acfc33daed9a2c73fdad7304a2fde191daea3`
- `workspace/taco_verl/train.parquet`，168,834,942 bytes，sha256 `8640b3e6c7a5ff3d5b51aa46e74f74147f3c565cd89b73676d5113f8e546d573`
- 归档 `artifacts/workspace-assets.tar.gz`，167,612,568 bytes，sha256 `eddeece8c2828e21b581d5168dd4604470165cb061a53e68d50b26d49ce6d916`，
  含 2329 行 `VERY_HARD` 训练数据（全为 `stdin`）与未改动的 184 行 `EASY` 验证集

`VERY_HARD` 没有 call-based 题。转换器对着既有 EASY `train.parquet` 校验过：5996 行
`stdin`、2372 行 `call_based` 匹配。超过 2048 prompt token 的行被丢弃，因此丢了 4 行
`VERY_HARD`。agent 可见的 parquet 与归档 revision 都不存 `extra_info.difficulty`；
prompt、`reward_model`、`index`、`split`、`code_type`、`fn_name` 均未改动。

模型仓库 `LIF1014/ptdbench-verl-implementation-torch-functional`，revision
`36a4c226ef70aef2c38df173a7cf86a47ad3ad00`，`model.safetensors` sha256
`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`——与 `torch_functional` 共享，不专属本任务。prepare 拉到
`$PTDBENCH_WORKSPACE/Qwen2.5-1.5B-Instruct`，即启动脚本的 `MODEL_PATH` 默认值。

**刻意不声明 GPU。** `metadata.environment.env` 为空，与其他 verl 任务不同。本任务只有
issue 档：agent 读历史快照、verifier 评 `agent_answer.md`，两个阶段都不跑训练栈。
声明 `NVIDIA_VISIBLE_DEVICES` 会让 docker runtime 给 agent 与 verifier 容器加
`--gpus`，在无 GPU 的宿主上直接失败。要复跑那 8 卡实验，在宿主设
`PTDBENCH_DOCKER_GPUS`；Harbor 侧重跑还需兄弟任务那种 `environment/docker-compose.yaml`
的 GPU/shm 覆盖。

**不装 `pyext`。** `verl/utils/reward_score/taco.py` 尝试
`from pyext import RuntimeModule`，`ImportError` 时回退到自带的
`_FallbackRuntimeModule`，而所有行走的 stdin 路径都经过它，所以装不装会改变奖励。
录制运行的冻结依赖清单没有 `pyext`——那才是复现这份运行的依据，而不是 legacy
Dockerfile。`reward_design/task_evaluator` 装了它，因为那份清单里有 `pyext==0.7`。
不要改 `docker/images/verl.Dockerfile`，那是所有 verl 任务共享的。

### 从 `prepare.sh` 移出的说明

`prepare.sh` 以只读挂进 agent 容器，所以不得携带点明缺陷或评分门槛的维护者背景。

**迁移：** 2026-10-16 用 `prepare.sh` 替换了 `environment/{Dockerfile,hydrate-assets.sh}`。
legacy Dockerfile 从 `verlai/verl:vllm017.latest` 构建专用镜像、装补丁版 `pyext`、
build 时下载模型与数据归档，并在 agent 阶段从 `/opt/ptdbench-assets` hydrate
`/workspace`。仓库、revision、字节数与 SHA-256 逐字搬过来，hydrate 目标路径不变。
归档布局已对着已发布的 revision 复核：恰好只有
`workspace/taco_verl/train.parquet` 与 `test.parquet`。

**任务名泄漏（未修）：** 任务目录名与已发布的数据仓库名
（`...-task-train-too-hard-dataset`）都写着答案，而 agent 两个都能看到——
`PTDBENCH_TASK_ID` 会导出到 agent 阶段，`prepare.sh` 在
`/ptdbench/environment` 可读。关闭它需要任务改名并重新发布一个中性名字的数据仓库，
所以记在这里而不是在 `prepare.sh` 里绕过。
