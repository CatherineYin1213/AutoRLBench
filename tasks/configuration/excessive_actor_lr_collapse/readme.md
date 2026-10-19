# excessive_actor_lr_collapse

| | |
| --- | --- |
| **Author** | 王嘉泽 |
| **Domain** | `configuration` |
| **Tags** | `ppo-critic` `vimpo` `learning-rate` `policy-collapse` `verl` |
| **Framework** | `verl`（VIMPO PPO-Critic recipe） |
| **Model** | `Qwen3-4B-Base` |
| **Dataset** | GuruMath controlled smoke（RL prompts）+ AIME-32 / MATH-500-32 held-out |
| **Environment** | `ppo-critic-vimpo`；`prepare.sh` 拉取 Qwen3-4B-Base |
| **Resources** | 4 GPUs |
| **Recorded runs** | 缺陷侧一次冻结运行（8 步）；修复侧尚无真实重跑 |
| **Recorded runtime** | 8 步 controlled smoke |

## Task abstract

`run_buggy.sh` 导出 `LR=1e-2`，recipe 把它当
`actor_rollout_ref.actor.optim.lr="${LR:-1e-6}"` 传给 actor 优化器——比预期的 1e-6
大 **10,000 倍**。策略随之退化：验证准确率从非零一路掉到 0，回复撑满 1024 token 上限，
grad_norm 与熵剧烈震荡。recipe 其余部分与健康对照逐字节相同，critic LR（1e-5）也没动。

## The defect

```bash
# run_buggy.sh
export LR=1e-2
```

消费点：

```bash
# recipe/vimpo/run_vimpo_critic.sh
actor_rollout_ref.actor.optim.lr="${LR:-1e-6}"
```

修复有两种等价写法：把 `run_buggy.sh` 里改成 `export LR=1e-6`，或删掉这行覆盖、让
recipe 默认值生效。

### Why nothing catches it

- **所有值都是合法的。** `1e-2` 是一个完全正常的浮点数，脚本不校验范围，配置也不
  报错。
- **启动阶段完全正常。** job 初始化、跑完全部 8 步（不是启动或基础设施失败）。
- **指标先好后坏。** `critic/rewards/mean` 起步约 0.03–0.06 才归零，验证集也是先非零
  再归零——看起来像训练后期不稳定，而不是一开始就错了。
- **偏差被 smoke 配置掩盖。** 4 卡、8 步、batch 16 的受控缩小是 harness 属性，不是
  原因，但它让"崩溃"发生在很短的时间窗里。

## Evidence

来自 `workspace/ptdbench_case/logs/observed_buggy_run.log`（8 个 PPO step，
`test_freq=1`）：

- job 初始化并跑完 8 步（非启动期失败）
- `critic/rewards/mean` 从 ≈0.03–0.06 起步，随后跌到 `0.0`
- 验证在 `val-core/math__math` 和 `val-core/math__aime_repeated_8x` 上都从非零归零
- 回复长度打满 1024 token 的 `MAX_RESPONSE_LENGTH`
- `actor/grad_norm` 与 actor 熵高度不稳定

这是 actor 学习率过大导致策略崩溃的经典特征。

## Reference answer

### `issue` — the diagnosis

根因目标：`run_buggy.sh` 中膨胀的 actor 学习率 `LR=1e-2`。

修复目标：在真实的 launcher→optimizer 路径上把 actor LR 落到 `1e-6`。

必须判不过的归因：基础设施启动失败、critic LR、某个"不寻常但无害"的设置。

### `solution` — 补丁

```diff
-export LR=1e-2
+export LR=1e-6
```

或直接删除该行，让 `${LR:-1e-6}` 生效。

## Verification

`issue` 比对 `$PTDBENCH_RESULTS/agent_answer.md` 与隐藏 `oracle.md`。judge 按
verdict / root-cause / evidence 打分（权重 0.2 / 0.6 / 0.2），加权 ≥ `0.8` 通过。

`solution` 是确定性、只看指标的，形状与 `verl_implementation/torch_functional` 相同：
修 LR、真实重跑启动脚本，verifier 从 `$PTDBENCH_RESULTS/experiment_logs` 下的日志
重建指标。`score = 0.8 * validation + 0.2 * training_reward`，两项都过才
`score == 1.0`。细节见 `evaluations/solution/verifier/resources/README.md`。

**门槛的标定状态（维护者注意）：** `grader_spec.json` 里的历史基线是缺陷 run 的真实
终值（`0.0`）。`required_final_score` / `required_training_reward` 是基于证据的保守
草稿，**尚未用真实健康重跑标定**；`task.yaml` 带 `needs_manual_review: true`。在用于
评分前，先用 oracle 修复版（`LR=1e-6`）标定。

## Running it

`metadata.environment.image: ppo-critic-vimpo` → `docker/images/ppo-critic-vimpo.Dockerfile`
（Python 3.12、CUDA 12.8、torch 2.8.0、vLLM 0.11.0、Ray 2.56.0、transformers 4.57.6、
flash-attn 2.8.3.post1）。

这个栈刻意**不是**共享的 `verl` 镜像（vLLM 0.17 / torch 2.10 / CUDA 12.9）：workspace
自带一份针对 vLLM 0.11 写的 veRL + VIMPO recipe，solution grader 重建的 val-core /
critic-reward 基线也是在这个栈上产出的。veRL 不烤进镜像；harness 把 workspace 挂在
`/workspace`，`PYTHONPATH=/workspace` 让 `import verl` 解析到它。

`environment/prepare.sh` 把 Qwen3-4B-Base 拉到 `/workspace/Qwen3-4B-Base`（即
`solution/instruction.md` 里有效运行使用的 `MODEL_PATH`）。`--revision` **钉死**在
`Qwen/Qwen3-4B-Base` 2026-08-18 的 main（commit
`906bfd4b4dc7f14ee4320094d8b41684abff8539`，录制运行时用的快照，已本地全量下载并逐个
shard 核对），三个权重 shard 另带 `--expect` sha256 与字节数校验。
`QWEN3_4B_BASE_REPO` / `QWEN3_4B_BASE_REVISION` 仍可覆盖（例如集群上 Qwen 的 xet-only
托管被封时指向镜像站；两个要一起覆盖，镜像站的 commit 不同）。

受控训练集与 held-out 资产很小，直接随包放在 `workspace/ptdbench_case/assets/*.parquet`，
prepare 不重取。

recipe 需要 **4 张卡**（`run_buggy.sh`：`CUDA_VISIBLE_DEVICES=0,1,2,3`、
`N_GPUS_PER_NODE=4`）；`MODEL_PATH` 必须指向本地 Qwen3-4B-Base checkpoint。

workspace 代码与 verl 快照 Apache-2.0，任务打包 MIT。
