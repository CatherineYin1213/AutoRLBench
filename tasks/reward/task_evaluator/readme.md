# task_evaluator

| | |
| --- | --- |
| **Author** | benchmark_evaluation |
| **Domain** | `reward` |
| **Tags** | `reward_design` `taco` `code-execution` `comparison` `verl` `injected-defect` |
| **Framework** | `verl` |
| **Model** | Qwen2.5-1.5B-Instruct |
| **Dataset** | TACO_EASY（400 步训练预算） |
| **Environment** | `verl` 镜像；数据集随预构建镜像烘焙 |
| **Resources** | 8 GPUs |
| **Recorded runs** | 缺陷侧一次 400 步运行；健康 scorer 同设置对照 |
| **Recorded runtime** | 400 步（约 8 小时量级） |

## Task abstract

暂存的 `taco.py` 去掉了列表解包和 stdin 的模糊回退，于是**只有字面输出相等**才算对。
`[1, 2]` 与 `[[1, 2]]`、空白归一化后的等值文本、近似相等的浮点、等价集合全部被判错。
held-out 从 step 0 到 300 一直是 **0.0**，而同一设置下健康 scorer 在 step 0 就到 0.0598、
step 400 到 0.139。

## The defect

`verl/utils/reward_score/taco.py::run_test` 删掉了 call-based 的元组强制转换与
`output == expected or expected[0]`，以及 stdin 的 `custom_compare_`、
`output == [ground_truth]`、strip、float、set 回退。

### Why nothing catches it

- **代码执行本身没坏。** parquet 数据、prompt、call-based 执行路径都没变，缺陷只限于
  结果比较。
- **精确比较仍然会通过。** 窄的 smoke test 用完全相等的输出就能过，缺陷因此被掩盖。
- **验证恒零但训练不报错。** 看起来像模型学不会，而不是评分器不给分。
- **`prime_code` 与 agent-loop 的工具解析器不参与**，指定 TACO 启动用不到它们，不构成
  因果。

## Evidence

- 脏验证从 step 0 到 300 一直 0.0；同设置下健康 scorer 在 step 0 是 0.0598、step 400
  是 0.139
- `[1, 2]` 对 `[[1, 2]]`：暂存 judge 给 0，健康 judge 给 1；而完全相等的列表仍给 1
- stdin 输出带多余空格：暂存 judge 给 0，健康 strip 回退给 1；完全相等的 stdin 行仍给 1

被保留的源指标是 `val-core/taco/acc/mean@1`。冻结 workspace 是证据包；solution 评分
要求一次新的 400 步训练，只读它的实验日志。

## Reference answer

### `issue` — the diagnosis

根因目标：`taco.py::run_test`。

修复目标：恢复 call-based 的元组强制转换与 `output == expected or expected[0]`，以及
stdin 的 `custom_compare_`、`output == [ground_truth]`、strip、float、set 回退。

必须澄清的归因：

- 数据、prompt、call-based 执行路径未变，缺陷只在结果比较
- 精确比较仍通过，所以窄 smoke test 会掩盖它
- `prime_code` 与 agent-loop 工具解析器不是因果

### `solution` — 补丁

从健康 scorer 恢复上述比较逻辑，其余不动。

## Verification

`issue` 权重 0.2 / 0.6 / 0.2，通过线 0.8。严格的 judge 行为及其对 held-out 奖励的
因果影响是本质根因。

`solution`：修好后跑指定启动脚本 400 步，评分加权：

- validation：有限的 `val-core/taco/acc/mean@1` 峰值 ≥ `0.110`（11.0%）记 1
- training reward：训练步上 `critic/rewards/mean` 的均值严格 > `0.14` 记 1
- `score = 0.8 * validation + 0.2 * training reward`

## Running it

`metadata.environment.image: verl` → `docker/images/verl.Dockerfile`
（`verlai/verl:vllm017.latest` 按 digest 钉死）。

数据**未发布**，随预构建镜像烘焙；当前 Dockerfile 不下载也不校验外部 dataset revision：

| Runtime path | Size | SHA-256 |
| --- | ---: | --- |
| `workspace/taco_verl/test.parquet` | 61,082,676 bytes | `e364fa69d973f8e8644b2fa4d7dc6ec405d5b4adff64002c6ad52e7c787c6eaa` |
| `workspace/taco_verl/train.parquet` | 606,016,648 bytes | `7f25abdfa9e0f5baa173d153a32973fbe2cc8736503a35408b8ebb8ae2bc5706` |

预构建镜像内嵌该快照，只在 agent 阶段 hydrate 到挂载的 workspace。local backend 不会
hydrate 烘焙的模型与数据资产。

指定实验用 8 卡、400 步预算、大 shared-memory 分配。用 Docker agent backend 与任务
专用预构建镜像。

模型权重与历史训练 checkpoint 不是外部数据集资产。
