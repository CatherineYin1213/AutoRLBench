# task_function_call

| | |
| --- | --- |
| **Author** | benchmark_evaluation |
| **Domain** | `reward` |
| **Tags** | `reward_design` `taco` `function-call` `call-based` `verl` `injected-defect` |
| **Framework** | `verl` |
| **Model** | Qwen2.5-1.5B-Instruct |
| **Dataset** | TACO_EASY（41 道 call-based + 143 道 stdin 的 held-out） |
| **Environment** | `verl` 镜像；数据集随预构建镜像烘焙 |
| **Resources** | 8 GPUs |
| **Recorded runs** | 修复 / 缺陷各一次 400 步，同主机；另有历史 run |
| **Recorded runtime** | 修复 9h37m、缺陷 8h44m |

## Task abstract

暂存的 `taco.py` 忽略 `fn_name`，把所有样本都塞进 stdin 路径，于是**函数调用题永远拿
零分**，而 stdin 题照常得分。held-out 里 41 道 call-based 题即使给出正确的函数解也
被压到零，聚合验证却仍然非零（0.0217 → 0.0543，峰值 0.0870）——因为 143 道 stdin 题
还在得分。

## The defect

`verl/utils/reward_score/taco.py::run_test` 删掉了 call-based 分支。

### Why nothing catches it

- **聚合指标非零。** 143 道 stdin 题撑起了分数，整体看起来"只是偏低"。
- **不报错。** 走 stdin 路径对 call-based 题是一组合法的（错误的）判定。
- **题目数据本身标记得很清楚。** parquet 里 `fn_name` 和 `Use Call-Based format` 都还在，
  缺陷只在 scorer。
- **`prime_code` 与 agent-loop 工具解析器不参与**指定 TACO 启动，不是因果。

## Evidence

- 脏验证从 0.0217 涨到 0.0543，step 300 峰值 0.0870
- 官方函数解在训练与验证上都得 0 分，而官方 stdin 解仍得 1 分
- held-out 含 41 道 call-based、143 道 stdin，所以每道 call-based 行即使函数解正确也
  被压在零

### Held-out results

修好后的 400 步参考运行在 devspace `1236983` 上跑过，同主机还有一次缺陷重跑，门槛由
这两条曲线标定（表见 `evaluations/solution/oracle/README.md`）：

| | 修复侧 | 缺陷侧 |
| --- | ---: | ---: |
| 修复参考终值 | 0.0978 | — |
| 标定值（validation / reward） | 0.1098 / 0.1838 | 0.0783 / 0.1003 |

早前版本用"健康 scorer 到 step 400 是 0.139"作参照点，那属于
`tasks_coding_pass_01`——另一个实验，其栈包含 `pyext`。本任务的 `prepare.sh` 刻意
不装 `pyext`，修复运行收在 0.0978。

## Reference answer

### `issue` — the diagnosis

根因目标：`taco.py::run_test`。

修复目标：当 `fn_name` 存在时，把候选按 call-based 代码编译、需要时实例化
`Solution`、调用 `method(*inputs)` 并比较返回值；stdin 路径保持不变。

必须澄清的归因：

- 数据与 prompt 仍通过 `fn_name` 与 `Use Call-Based format` 标识 call-based 题，
  缺陷只在 scorer
- 保留的 stdin 路径解释了为什么所有 call-based 行都失败时聚合验证仍非零
- `prime_code` 与 agent-loop 工具解析器不是因果

### `solution` — 补丁

恢复 call-based 执行分支，stdin 路径不动。

## Verification

`issue` 权重 0.2 / 0.6 / 0.2，通过线 0.8。忽略 `fn_name`、删掉 call-based 执行路径是
本质根因。

`solution`：修好后跑指定脚本 400 步：

- validation：`val-core/taco/acc/mean@1` 在固定步集 `200, 250, 300, 350, 400` 上的
  均值必须落在 `[0.095, 0.35]`；五步都必须存在（`TEST_FREQ=50` 的 400 步运行正好产出）
- training reward：训练步 `1..400` 的 `critic/rewards/mean` 均值严格 > `0.14` 且
  ≤ `0.6`；400 步都必须存在
- `score = 0.8 * validation + 0.2 * training reward`

固定步集而不是取峰值，是因为 `TEST_FREQ` 与 `TOTAL_TRAINING_STEPS` 只是启动脚本里的
普通环境变量覆盖，而修复运行的峰值（0.125）只出现一次。两个上界用来挡"把 scorer 改成
返回常数"——两个评分量都出自 agent 被要求修的那个文件。

## Running it

`metadata.environment.image: verl` → `docker/images/verl.Dockerfile`
（`verlai/verl:vllm017.latest` 按 digest 钉死）。

数据**未发布**，随预构建镜像烘焙；当前 Dockerfile 不下载也不校验外部 dataset revision：

| Runtime path | Size | SHA-256 |
| --- | ---: | --- |
| `workspace/taco_verl/test.parquet` | 61,082,676 bytes | `e364fa69d973f8e8644b2fa4d7dc6ec405d5b4adff64002c6ad52e7c787c6eaa` |
| `workspace/taco_verl/train.parquet` | 606,016,648 bytes | `7f25abdfa9e0f5baa173d153a32973fbe2cc8736503a35408b8ebb8ae2bc5706` |

预构建镜像内嵌快照，只在 agent 阶段 hydrate。local backend 不 hydrate 烘焙资产。

8 卡、400 步、大 shared-memory。`verify.timeout_minutes` 为 660，而修复参考运行本身
就用了 9h37m（缺陷 8h44m——恢复 call-based 执行会更慢），agent 的诊断与修复阶段只有
不到 1h20m 余量。

模型权重与历史训练 checkpoint 不是外部数据集资产。

### 从 `prepare.sh` 移出的说明

`prepare.sh` 以只读挂进 agent 容器的 `/ptdbench/environment`，所以不得携带点明缺陷、
兄弟任务或评分门槛的维护者背景。

**叶子 id 脱敏。** 原实验名 `taco_qwen2_5_1.5b_function_call` 直接点明缺陷，在训练日志、
离线 W&B 的 `config.yaml` / `output.log` / `debug.log` / `run-*.wandb`、以及 `outputs/`
下的 hydra 配置里共出现 30 次。`workspace/` 内的每一处现为 `__BENCH_TASK_ID__`，启动
脚本读 `EXPERIMENT_NAME=${EXPERIMENT_NAME:-${BENCH_TASK_LEAF_ID:-__BENCH_TASK_ID__}}`；
没有 harness 代码设置 `BENCH_TASK_LEAF_ID`，所以新 run 输出同一个串、快照保持自洽。
**不要恢复原名。** 二进制 `run-*.wandb` 是逐记录重新生成的（长度、CRC、32KB 分块
重新计算）而非原位打补丁；`oracle/task.yaml` 仍记录历史名，因为 oracle 树对 agent
不可见。

**迁移。** `prepare.sh` 于 2026-09-23 取代
`environment/{Dockerfile,hydrate-assets.sh}`。legacy Dockerfile 只装补丁版 `pyext` 并把
`workspace/` 烤进镜像——既不下载模型也不下载数据，而两者都不在 git 中，所以由它构建
的容器会在 `MODEL_PATH` 和 `TRAIN_FILE` 上立刻失败。两份资产是照着运行时的路径、
字节数和 SHA-256 重建的。

**兄弟任务。** `reward_design/task_evaluator` 出自同一 recipe——对同一源文件的两种注入。
资产**内容**逐字节相同，但各自在自己的 revision 上发布各自的 HF 仓库，所以这里改动通常
要同步审那边。两份 `prepare.sh` **刻意不同**：只有兄弟装了 `pyext`。

**为什么不装 `pyext`。** scorer 尝试 `from pyext import RuntimeModule`，`ImportError`
时回退到自带的 `_FallbackRuntimeModule`，而 stdin 路径每个样本都会走到回退，所以装与
不装会落在奖励上，不是装饰性选择。决定跟随本任务自己的冻结 run manifest：
`workspace/logs/wandb/run-20260821_153903-js8e27l6/files/requirements.txt` 是 282 行、
没有 `pyext`；兄弟的冻结 manifest 是 283 行、有 `pyext==0.7`。逐行排序后只差这一个包。
两者并不等价：`pyext` 的 `from_string` exec 进临时 globals 再拷到新模块，函数的
`__globals__` 与模块 `__dict__` 是两个对象；回退直接 exec 进 `module.__dict__`。这里
唯一的 `getattr(mod, 'code')` 用法结果相同，语义不同——这正是每个任务要单独对齐的原因。
要恢复它，从兄弟 `prepare.sh` 尾部搬一段（装进 `$PTDBENCH_PREPARED/pysite` 并把
`PYTHONPATH` 追加到 `$PTDBENCH_PREPARED/env`）。**不要**改
`docker/images/verl.Dockerfile`，那是所有 verl 任务共享的。

**已知残留泄漏。** 两个 HF 仓库名叫
`LIF1014/ptdbench-verl-coding-tasks-function-call` 和 `…-function-call-dataset`。
`prepare.sh` 必须写出它们才能拉取，所以这个串在包里无法消除——关闭它意味着用中性名
重新发布两个仓库并更新此处的 revision。
