# reward_parser_mismatch

| | |
| --- | --- |
| **Author** | 王嘉泽 |
| **Domain** | `reward` |
| **Tags** | `ppo-critic` `vimpo` `reward-function` `parser` `zero-reward` `verl` |
| **Framework** | `verl`（VIMPO PPO-Critic recipe） |
| **Model** | `Qwen3-4B-Base` |
| **Dataset** | GuruMath controlled smoke（RL prompts）+ AIME-32 / MATH-500-32 held-out |
| **Environment** | `ppo-critic-vimpo`；`prepare.sh` 拉取 Qwen3-4B-Base |
| **Resources** | 4 GPUs |
| **Recorded runs** | 缺陷侧一次受控运行（5 步）；修复侧尚无真实重跑 |
| **Recorded runtime** | 5 步 controlled smoke |

## Task abstract

自定义奖励函数用 XML 标签正则提取答案（`<answer>…</answer>`），而 GuruMath 的
prompt 与标准答案用的是 `\boxed{...}`。**没有任何回复能匹配**，于是每条 rollout 都得
`0.0`，`grade_answer` 根本走不到。奖励信号恒为零——但训练照常跑完，梯度、
critic 更新、MFU 全部健康，什么都不报错。

## The defect

`run_buggy.sh` 接上 `reward.custom_reward_function.path=ptdbench_case/buggy_reward_fn.py`，
其中：

```python
re.search(r"<answer>\s*(.*?)\s*</answer>", ...)   # 永远匹配不上 \boxed{...}
```

每个样本都返回 `{"score": 0.0, "acc": False, "parser_matched": False}`。

### Why nothing catches it

- **不抛异常。** 正则没匹配到就返回 0 分，这是一条合法路径。
- **奖励函数确实在运行。** `timing_s/reward` ≈ 0.0002 s，非零——它只是从不命中。
- **训练指标全绿。** 梯度、critic 更新、MFU 都正常，PPO 完成全部步数，只是优势在零
  附近抖动，策略什么也没学到。
- **单看验证指标像"模型不行"。** `val-core` 和 `critic/rewards/mean` 全是 0，很容易
  被归因为优化器或策略崩溃。

## Evidence

来自 `workspace/ptdbench_case/logs/observed_buggy_run.log`（5 个 PPO step，
`test_freq=1`，`val_before_train=True`）：

- `critic/rewards/mean` 与 `critic/score/mean` 每一步都是 `0.0`
- `val-aux/math__math/parser_matched/mean@1` 为 `0.0`（没有回复匹配上）
- `val-core/math__math/acc/mean@1` 与 `val-core/math__aime_repeated_8x/acc/mean@1`
  恒为 `0.0`
- `timing_s/reward` 非零（约 0.0002 s），证明奖励函数确实被调用

## Reference answer

### `issue` — the diagnosis

根因目标：`ptdbench_case/buggy_reward_fn.py::compute_score` 里的 XML 标签解析器。

修复目标：改为提取 `\boxed{...}`（取最后一处）并据此评分，同时保持
`compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)`
签名和 `{score, acc, parser_matched}` 返回契约。

必须判不过的归因：优化器、策略崩溃、基础设施启动失败。

### `solution` — 补丁

把答案提取换成 `\boxed{...}` 的最后一处匹配，再交给 `grade_answer`。

## Verification

`issue` 比对 `agent_answer.md` 与隐藏 `oracle.md`，权重 0.2 / 0.6 / 0.2，通过线 0.8。

`solution` 确定性、只看指标：修好奖励函数、真实重跑启动脚本，verifier 从
`$PTDBENCH_RESULTS/experiment_logs` 重建 `val-core/math__math/acc/mean@1` 与
`critic/rewards/mean`。`score = 0.8 * validation + 0.2 * training_reward`，两项都过
才 `score == 1.0`。

**门槛的标定状态：** `grader_spec.json` 里历史基线是缺陷 run 的真实终值（两次都是
`0.0`）。两个 `required_*` 是保守草稿，**尚未用真实健康重跑标定**；
`task.yaml` 带 `needs_manual_review: true`。用 oracle 修复版跑一次再启用评分。

## Running it

`metadata.environment.image: ppo-critic-vimpo` →
`docker/images/ppo-critic-vimpo.Dockerfile`（Python 3.12、CUDA 12.8、torch 2.8.0、
vLLM 0.11.0、Ray 2.56.0、transformers 4.57.6、flash-attn 2.8.3.post1）。

刻意不是共享 `verl` 镜像：workspace 自带的 veRL + VIMPO recipe 面向 vLLM 0.11，且
grader 重建的基线就在这个栈上产出。veRL 不烤进镜像；workspace 挂在 `/workspace`，
`PYTHONPATH=/workspace` 让 `import verl` 解析过去。

`prepare.sh` 把 Qwen3-4B-Base 拉到 `/workspace/Qwen3-4B-Base`，revision 钉在
`906bfd4b4dc7f14ee4320094d8b41684abff8539`，三个 shard 带 sha256 与字节数校验。

受控资产随包在 `workspace/ptdbench_case/assets/*.parquet`。

recipe 需要 **4 张卡**；`MODEL_PATH` 指向本地 Qwen3-4B-Base checkpoint。

workspace 代码与 verl 快照 Apache-2.0，任务打包 MIT。
