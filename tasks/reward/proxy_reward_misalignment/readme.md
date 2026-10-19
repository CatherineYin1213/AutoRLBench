# proxy_reward_misalignment

| | |
| --- | --- |
| **Author** | 王嘉泽 |
| **Domain** | `reward` |
| **Tags** | `ppo-critic` `vimpo` `proxy-reward` `reward-hacking` `brevity` `verl` |
| **Framework** | `verl`（VIMPO PPO-Critic recipe） |
| **Model** | `Qwen3-4B-Base` |
| **Dataset** | GuruMath fixed-16 controlled set（RL prompts）+ AIME-32 / MATH-500-32 held-out |
| **Environment** | `ppo-critic-vimpo`；`prepare.sh` 拉取 Qwen3-4B-Base |
| **Resources** | 4 GPUs |
| **Recorded runs** | 缺陷侧一次受控运行（15 步）；修复侧尚无真实重跑 |
| **Recorded runtime** | 15 步 controlled smoke |

## Task abstract

自定义奖励函数把**简洁度**当成训练分数：

```text
proxy_reward = max(0.0, min(1.0, 1.0 - n_chars / 4000.0))
```

回复越短奖励越高，与数学正确性无关。于是训练奖励一路上涨（约 0.52 → 0.67），而
held-out 准确率同时**下跌**（0.78125 → 0.59375），平均回复长度从约 677 掉到 474。
这是代理目标错位，不是奖励信号死掉。

## The defect

`run_buggy.sh` 接上
`reward.custom_reward_function.path=ptdbench_case/type3_length_reward_v2_fn.py`。
对训练源（`data_source == "ptdbench_type3_proxy_train_v2"`），`compute_score` 返回的
`score` 是 `1 - n_chars/4000`，不是正确性。

验证（`data_source != TRAIN_SOURCE`）仍然评真实正确性，所以损失与评测指向不同的东西。

### Why nothing catches it

- **函数是诚实的。** 它带着真实的 `acc` / `true_math_score` 字段，错位因此**可观测**：
  奖励涨而 `true_math_score` 不涨。
- **PPO 完全正常。** 梯度、critic 更新、MFU 都健康，15 步跑完——缺陷在**目标**里，
  不在优化器里。
- **训练指标看起来最好。** `critic/rewards/mean` 是高且上升的，按训练日志判断会得出
  "一切正常"的结论。
- **早停点会骗人。** held-out 准确率在 **step 0** 最高（0.78125），之后一路跌；
  任何取峰值的评分方式都会让未修复的重跑通过。

## Evidence

来自 `workspace/ptdbench_case/logs/observed_buggy_run.log`（15 个 PPO step，
`test_freq=5`，`val_before_train=True`）：

- `critic/rewards/mean`（代理）高且上升：约 `0.52 → 0.67`，均值约 `0.54`
- held-out `val-core/math__math/acc/mean@1` 下跌：`0.78125`（step 0）→ `0.6875`（5）
  → `0.65625`（10）→ `0.59375`（15）
- `response_length/mean` 下降（约 `677 → 474`）：策略正在学会变短，正是代理奖励的
- 辅助的真值跟踪 `true_accuracy/score_mean` 约 `0.52` 且不改善，确认代理增益不反映
  正确性

## Reference answer

### `issue` — the diagnosis

根因目标：`type3_length_reward_v2_fn.py::compute_score` 对训练源使用简洁度代理。

修复目标：训练源按数学正确性评分——短而错得 0 分，长短正确的 `\boxed{}` 答案都得 1 分，
回复长度不影响训练奖励；保持 `compute_score(...)` 签名。`oracle/fixed_reward_fn.py`
把训练源路到 `math__math` 上的 `default_compute_score`。

必须判不过的归因：奖励死掉、解析器不匹配、actor LR 崩溃、启动失败。

### `solution` — 补丁

让训练源走正确性评分，长度不再进入奖励。

## Verification

`issue` 比对 `agent_answer.md` 与隐藏 `oracle.md`，权重 0.2 / 0.6 / 0.2，通过线 0.8。

`solution` 确定性、只看指标：`score = 0.8 * validation + 0.2 * training_reward`，
两项都过才 `score == 1.0`。

**奖励劫持的特殊处理（维护者必读）：这是四个任务里最不自标定的一个。**

- validation 评的是**终值**而非峰值。缺陷 run 的准确率在 step 0 见顶，取峰值会让
  未修复的重跑过关；所以 verifier（`grade_submission.py::select_final`）刻意取
  ≤ `final_step` 的最后一步。
- `critic/rewards/mean` 在缺陷 run 里按构造就高（就是那个代理，约 0.54），
  **不能**区分修没修。真正的判别量是终值验证准确率；`required_training_reward`
  保持低地板（`0.05`）只为挡住彻底崩掉的重跑。
- `historical_final_baseline` 是缺陷终值（`0.59375`）。`required_final_score = 0.65`
  是**未标定的保守草稿**，放在劫持终值之上、step-0 基线之下，要求修复能止住下跌。
  它**尚未**用真实正确性修复的重跑验证，可能需调整；`task.yaml` 带
  `needs_manual_review: true`。

## Running it

`metadata.environment.image: ppo-critic-vimpo` →
`docker/images/ppo-critic-vimpo.Dockerfile`（Python 3.12、CUDA 12.8、torch 2.8.0、
vLLM 0.11.0、Ray 2.56.0、transformers 4.57.6、flash-attn 2.8.3.post1）。

刻意不是共享 `verl` 镜像：workspace 自带面向 vLLM 0.11 的 veRL + VIMPO recipe，grader
重建的基线就在这个栈上产出。veRL 不烤进镜像；`PYTHONPATH=/workspace` 让
`import verl` 解析到挂载的 workspace。

`prepare.sh` 把 Qwen3-4B-Base 拉到 `/workspace/Qwen3-4B-Base`，revision 钉在
`906bfd4b4dc7f14ee4320094d8b41684abff8539`，三个 shard 带 sha256 与字节数校验。

受控训练集（`train_proxy_fixed16_x64.parquet`）与 held-out
（`aime32.parquet`、`math500_32.parquet`）很小，随包在
`workspace/ptdbench_case/assets/`，prepare 不重取。

recipe 需要 **4 张卡**；`MODEL_PATH` 指向本地 Qwen3-4B-Base checkpoint。

workspace 代码与 verl 快照 Apache-2.0，任务打包 MIT。
