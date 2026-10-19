# slime_search_r1_nomask

| | |
| --- | --- |
| **Author** | chenyinghao |
| **Domain** | `reward` |
| **Tags** | `slime` `search-r1` `grpo` `loss-mask` `observation-tokens` `nq` |
| **Framework** | `slime`（Megatron train + SGLang rollout），`examples/search-r1/` |
| **Model** | Qwen2.5-3B |
| **Dataset** | Search-R1 `nq_hotpotqa_train`（NQ + HotpotQA 合并） |
| **Environment** | `slime-v0.3.2`；`prepare.sh` 取约 90 GB 资产 |
| **Resources** | 8 GPUs |
| **Recorded runs** | 冻结运行一次；另有 2 次缺陷侧重跑与 2 次修复侧重跑 |
| **Recorded runtime** | 200 步（冻结证据） |

## Task abstract

检索回来的 `<information>` 观察 token 被算进了损失（`loss_mask=1`）。策略于是学会
**不要检索**：训练回复长度从 step 0 的约 1751 token 塌到 step 100 的约 9 token，
冻结的 `eval/nq_test` 在 200 步只到 **16.5%**，而健康协议基线是 **40.8%**。

## The defect

`examples/search-r1/generate_with_search.py::generate` 在追加检索到的
`<information>` token 时给了 `loss_mask=1`，而 `append_response_tokens` 仍把它们标成
`trainable=False`；本地列表随后被赋给 `sample.loss_mask`。

### Why nothing catches it

- **形状和契约全对。** token、mask、长度都合法，训练不报错。
- **训练奖励也在涨。** `rollout/raw_reward` 看起来正常，不构成 held-out 健康的证据。
- **曲线不像是坏掉，像是"模型选择直接答题"。** 回复长度塌缩本身就能被解释成效率提升。
- **不是截断造成的。** 截断率从 step 0 的约 38% 降到 step 100 的约 0%，所以 held-out
  的差距与截断无关。

## Evidence

- 21 个重建的 `eval/nq_test` 点（到 `eval/step` 199，即 `--eval-interval 10` 下 200 步
  前最后一次 eval）从约 4.8% 涨到 16.5%（`eval_last` 0.1647；峰值 16.6% 在 step 189）
- 健康的 Qwen2.5-3B Search-R1 GRPO 运行（`tasks/slime_search_r1`）在同一 horizon 到约
  **40.8%**，step 99 时约 37.3%
- 训练回复长度：step 0 约 1751 token → step 100 约 9 token
- 截断率：约 38% → 约 0%

### Held-out results

修复重跑预算刻意短于 200 步的冻结运行，评分停在 step 99：

| 量 | 修复侧 | 缺陷侧 |
| --- | ---: | ---: |
| best（step 0..99） | 0.3733 / 0.3570 | 0.2237 / 0.1587 |
| step 99 | 0.3733 | 0.1543 |

未训练的 step 0 约 0.045–0.051。

**两界由四次实测运行标定**（devspace `1107990`、`/user/chenyinghao/harbor-runs/`），
门槛取 `[0.28, 1.0]`，最薄余量 **0.056**，在缺陷侧。`0.28` 与健康孪生
`slime_search_r1` 用的是同一个数，所以门槛本身不泄露任务在配对里的哪一侧。

**为什么重跑预算比冻结运行短：** 这个缺陷是高方差的，且方差随步数增长。两次实测的
缺陷运行代码、参数、检索服务完全相同，一次在 step 20 就塌到约 9 token 的回复、
平台在 0.1657，另一次保持约 650 token 的回复、被 kill 时（step 119）还在往上爬
（0.2713）。在 step 99 两侧都完整观测过，所以 `[0.2237, 0.3570]` 的间隔是实测的；
在 step 199 缺陷侧是被截断的，那里没有任何门槛能被证明排除它。**不要在没有先跑一次
带缺陷的 200 步运行前把评分 horizon 延到 199。**

## Reference answer

### `issue` — the diagnosis

根因目标：`generate_with_search.py::generate` 里的 `observation_loss_mask`。

修复目标：把观察 token 排除出损失：

```python
loss_mask += [0] * len(obs_tokens_ids)
```

指出观察 loss mask 的正确因果角色即为完整归因。

### `solution` — 补丁

把观察 token 的 `loss_mask` 置 0，跑 `examples/search-r1/run_qwen2.5_3B.sh`。

## Verification

`issue` 权重 0.2 / 0.6 / 0.2，通过线 0.8。

`solution`：修好后跑 `examples/search-r1/run_qwen2.5_3B.sh`，固定
`--num-rollout 100` / `--eval-interval 10`，得到 0, 9, …, 99。`score_mode: best`，
最高的有限 `eval/nq_test` 必须落在 `[0.28, 1.0]`。

冻结 W&B 重建用 `eval/step` 199 作为 200 步 horizon，因为 slime 记录
`eval/step = rollout_id`，在 0、9、…、199 评测。

## Running it

镜像 `slime-v0.3.2`（`environment: image: slime-v0.3.2`），即上游
`slimerl/slime:v0.3.2-cu130` 按 digest 钉死——CUDA 13.0、sglang 0.5.15.post1。
没有按任务的 Dockerfile。

`prepare.sh` 先把 `faiss-gpu-cuvs` 装进 `$PTDBENCH_PREPARED`（不装进镜像，四个 slime
任务共享它）并通过 `PYTHONPATH` 导出。`-cuvs` 是 CUDA 13 的构建，正对本环境；
普通 `faiss-gpu` 是 CUDA 12 的。

然后把重跑要用的五份资产放进 `$PTDBENCH_WORKSPACE`：`Qwen2.5-3B` checkpoint 及其
`torch_dist` 转换、`e5-base-v2` 查询编码器、
`nq_hotpotqa_train/{train,test}.parquet`（`ptdbench_dataset --recipe
search_r1_nq_hotpotqa`）、`wiki-18/{e5_Flat.index,wiki-18.jsonl}`
（`--recipe wiki18_e5`）。约 90 GB，缓存在 `$PTDBENCH_DATASETS` / `$PTDBENCH_MODELS`
并逐 run 硬链。文件正文与 `slime_search_r1` 的逐字节相同。

`metadata.yaml` 的 `environment.env` 刻意不写 `HF_HUB_OFFLINE` /
`TRANSFORMERS_OFFLINE`：框架按阶段设置，任务级副本会连 prepare 的下载一起堵死。

用 Docker agent backend。

issue 诊断只用冻结快照。solution 重跑还需要 `:8000` 上的检索服务。`prepare.sh`
不能保活它——prepare 阶段是围在脚本前后的一个容器——所以
`run_qwen2.5_3B.sh` 在端口没人监听时自己拉起并等待就绪。`prepare.sh` 提前把语料的
`datasets` arrow 缓存在 `$PTDBENCH_WORKSPACE/.hf_datasets_cache`（经 `prepared/env`
导出为 `HF_DATASETS_CACHE`；持久副本在 `$PTDBENCH_DATASETS`）。冷启动仍要把 60 GiB
索引读进 GPU 内存，耗时数分钟。

模型权重与历史训练 checkpoint 不是数据集资产。

workspace 代码与 Search-R1 配方 Apache-2.0。
