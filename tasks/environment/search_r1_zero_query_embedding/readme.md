# search_r1_zero_query_embedding

| | |
| --- | --- |
| **Author** | chenyinghao |
| **Domain** | `environment` |
| **Tags** | `search-r1` `grpo` `retrieval` `faiss` `e5` `hotpotqa` `device-transfer` `silent-corruption` |
| **Framework** | Search-R1 + 内置 veRL `598e61bd1d36895726d28a8d06b3a15bed19f5d3`，FSDP + vLLM 0.6.3 |
| **Model** | Qwen2.5-1.5B-Instruct |
| **Dataset** | HotpotQA distractor，1024 训练 / 128 验证，12468 段语料 |
| **Environment** | `search-r1-verl`（**未构建**）；`prepare.sh` 拉模型并重建索引 |
| **Resources** | 1 × A100-40G |
| **Recorded runs** | 缺陷 / 对照各一次，同一主机并发，同 seed 同数据 |
| **Recorded runtime** | 缺陷侧 3393 s、对照侧 3056 s（仅计训练步） |

## Task abstract

`Encoder.encode` 返回的查询向量**全为零**：它返回的那个 CPU 零数组从未被写入。内积
索引下每篇文档得分都是 `0.0`，FAISS 对任何问题都返回同一组任意 top-3——形状合法、
dtype 合法，无异常、无警告。策略照常训完，只是读到的文本与问题无关。held-out
`val/em` 停在 **15.625%**，同配方把向量拷回主机的一组到 **32.813%**。

## The defect

`Search-R1/search_r1/search/retrieval_server.py::Encoder.encode`，117–120 行：

```python
response_buffer = np.zeros(tuple(query_emb.shape), dtype=np.float32)
writer = torch.as_tensor(response_buffer, device=query_emb.device)
writer.copy_(query_emb)
query_emb = response_buffer
```

`response_buffer` 在主机内存里，`query_emb` 在 GPU 上（`load_model` 无条件
`model.cuda()`）。`torch.as_tensor(cpu_array, device=cuda)` **不会**让 `writer` 与原数组
共享存储，而是拷贝出一个新的 GPU tensor；`writer.copy_()` 忠实地写进这个随即被丢弃的
临时量，函数返回的仍是不曾改动的零数组。

### Why nothing catches it

- **同设备时是对的。** 若 buffer 与 `query_emb` 在同一设备上，`as_tensor` 会别名，
  代码就正常——这决定了它能通过 review。
- **零向量是合法输入。** `METRIC_INNER_PRODUCT` 下零查询对所有文档得分 `0.0`，索引
  正常返回格式良好的 top-3，不报错、不改变形状、不产生 NaN。
- **训练和验证共用一个检索服务**，两条路径都受影响。
- **故障组仍在学习。** `val/searched` 在五个验证点全为 1.0，`val/format` ≥ 0.977，
  EM 从 3.125% 涨到 15.625%。说"训练没有提升"是**说反了现象**。

## Evidence

`workspace/` 保留了启动入口 `run.py`（它同时管检索服务与 trainer）、检索包装、
带缺陷的完整 Search-R1 + veRL 源码、随包数据，以及那次缺陷运行自己的记录：
`train.log`、`launch.log`、`retrieval_server.log`、`experiment.json`、`retrieval.jsonl`
节选、全部 32 个 `rollouts/metrics_*.json`、4 个 `rollouts/step_*.jsonl` 样本、
`metrics_summary.json`，以及瘦身后的离线 W&B（无 `.wandb` 二进制历史）。

- 五个 held-out 点（step 0/10/20/30/32）：`val/em` 3.125% → 4.688% → 11.719% →
  13.281% → 15.625%，`val/f1` 8.970% → 24.703%
- 缺陷侧 `retrieval.jsonl` 里所有文档得分都是 `0.0`，同一 batch 内每个 query 返回同样的
  三篇文档：`"Scott Derrickson nationality"` → `"Sheep–goat hybrid"`。交付包的审计在
  完整 trace 上数到 13959 个零点、恰好一种文档组合，对照组是 14193 个非零点、
  3285 种组合
- 训练前用问题本身作 query 的 held-out 检索预检：supporting-title recall@3 为 `0.0`、
  any-support hit@3 为 `0.0`，对照组 `0.809` / `0.977`
- 搜索协议健康：`val/searched` 1.0、`val/format` ~1.0、`env/number_of_valid_search`
  0.93–1.02
- 回复偏长：`response_length/mean` 约 670 token，对照组约 380
- 训练奖励也在涨（`critic/score/mean` 0.016 → 0.164），不构成 held-out 健康的证据
- **修复在树内可发现**：`search_r1/search/retrieval.py:127-128` 是同一段
  `Encoder.encode` 的另一个副本（无人服务），结尾两行是正确的；
  `index_builder.py:237` 对文档也是这么做的。两份编码器互相矛盾是最干净的诊断路径，
  issue oracle 认可它为证据

### Held-out results

对照组与暂存侧只差那四行（两棵 `Search-R1/` 树 `diff` 只有一个文件不同），同主机、
同 seed、同划分、同语料、同索引、同编码器、同预算，32 步到 **32.813%**。暂存侧的
15.625% 比它低 17.2 个点，其中 10.2 个点在 step 0 就已存在。

两次运行都正常结束（`exit_code: 0`）。日志记录的 step 时间 3393 s / 3056 s，即单卡
A100-40G 上约一小时训练，另加检索与 vLLM 启动。两臂在同一台 8 卡机器上**并发**运行，
时间含资源争用；`oracle/recorded_comparison/ISOLATION_AUDIT.md` 是交付包自己记录
"共享可能影响什么、不可能影响什么"的文件，并明确不宣称完全隔离。

单 seed、128 题验证集的成对短程实验，不构成统计显著性声明。

## Reference answer

### `issue` — the diagnosis

根因目标：`Encoder.encode` 里的 `retriever_query_device_copy`。

修复目标：在 FAISS 看到向量之前拷回主机——`query_emb.detach().cpu().numpy()`，再
`astype(np.float32, order="C")`。

必须判不过的归因：

- "运行没有学习"——实测涨了 12.5 个 EM 点。oracle 与 `standard_answer.txt` 都把这条
  列为 0 分归因
- 只说"编码器在 GPU、索引在 CPU"，而没指出没被写入的 buffer 才是数据从未跨设备的原因
- `max_turns=2`、`topk=3`、`answer_only_reward`、`state_masking`、FSDP offload、
  `gpu_memory_utilization=0.35`、单卡配置、32 步预算、1024/128 划分——都是配方配置

### `solution` — 补丁

```diff
-        response_buffer = np.zeros(tuple(query_emb.shape), dtype=np.float32)
-        writer = torch.as_tensor(response_buffer, device=query_emb.device)
-        writer.copy_(query_emb)
-        query_emb = response_buffer
+        query_emb = query_emb.detach().cpu().numpy()
+        query_emb = query_emb.astype(np.float32, order="C")
```

起作用的是 `.cpu()`。

## Verification

`issue` 比对 `$PTDBENCH_RESULTS/agent_answer.md` 与隐藏 `oracle.md`，judge 按
verdict / root-cause / evidence 打分（0.2 / 0.6 / 0.2），加权 ≥ 0.8 通过。`judge.py`、
`judge_prompt.md`、`test.sh` 与 `tasks/reward/slime_search_r1_nomask` 的逐字节相同。

`solution` 不评诊断。修好后按 `config.json` 里的预算重跑：

```bash
python3 run.py --run-name graded --runs-dir "$PTDBENCH_RESULTS/experiment_logs"
```

32 步、每 10 步评测，得到 0/10/20/30/32 五个点。`score_mode: best`，最高的有限
`val/em` 必须落在 `[0.24, 1.0]`。

`0.24` 比缺陷侧上限（0.15625）高 0.084、比对照组（0.32813）低 0.088，并且高于对照组
未训练的 step-0 值（0.13281），所以"改对代码但没跑完"同样不过。

**标定只有每侧一次运行。** `slime_search_r1_nomask` 用四次实测运行定门槛、能报最薄
余量，本任务不能。这里没有任何东西能建立 seed 间方差，而缺陷侧恰恰属于"上限取决于
策略多快学会忽略无用观察"的那类：

- 评分结果落在 0.24 附近的，属于未测区间；无论判通过与否，先读
  `experiment_logs/graded/train.log` 与 `retrieval.jsonl`
- 在同一预算下再跑一对，是支撑收紧、放宽或移动门槛的后续工作
- 不指望重跑与记录曲线逐位一致：FAISS 索引在 prepare 阶段重建而非随包，GPU 也有
  不确定性

## Running it

镜像 `search-r1-verl`（`docker/images/search-r1-verl.Dockerfile`），即
`results/baseline32/experiment.json` 记录的那套栈——Python 3.10、torch 2.4.0、
vLLM 0.6.3、flash-attn 2.6.3、transformers 4.46.3、ray 2.40.0、faiss-cpu 1.14.2。
它由交付包自己的 `uv.lock` 安装，lock 暂存在 `docker/search-r1-runtime/`；**那个 lock
才是 pin，不是 Dockerfile**。共享的 `verl` 镜像（vLLM 0.17 / torch 2.10）比本
workspace 面向的 API 高了五个 vLLM 次版本。

> **镜像从未构建过。** 它写在一台无 GPU、也访问不到 Docker Hub / GitHub 的机器上，
> 所以 `FROM` 按 tag 而非 digest 钉死，预编译 flash-attn wheel 的 URL 未经验证
> （404 时回退编译 lock 里的 sdist）。首次构建时按 Dockerfile 里的命令把 digest 钉上。

`environment/prepare.sh`：

- 用 `environment/data_manifest.json` 校验三个随包数据文件
- 拉取 `Qwen2.5-1.5B-Instruct`（revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，`model.safetensors` sha256
  `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`，即录制运行记下的哈希）与 `intfloat/e5-base-v2`（revision
  `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`，sha256 `d0d559c47d5f71b1d280b13b62a2657f3e3bc70c0786f9ab91a36545e6a8f693`，与交付包副本及 `slime_search_r1_nomask` 钉的同一份，
  两个任务共用一条缓存）
- 用 `environment/build_index.py` 重建 `shared/data/e5_Flat.index`

约 3.5 GB 权重缓存在 `$PTDBENCH_MODELS`；索引缓存在 `$PTDBENCH_DATASETS` 并逐 run
硬链。

索引重建而非随包，因为它是随包 6.6 MB 语料的纯函数，而二进制本身 37 MB。
`build_index.py` 精确复现上游的索引方式（`passage: ` 前缀、mean pooling、L2 归一化、
`Flat` + 内积、max_length 256），prepare 没有 GPU 时退回 CPU。它的建后断言**只做
文档侧**——条数、维度、有限性、单位范数——因为 `environment/` 对 agent 可读，在那里
断言查询侧检索质量等于直接指出缺陷位置。两份文档向量完全相同，所以文档侧是可以免费
断言的。

`metadata.yaml` 的 `environment.env` 刻意不写 `HF_HUB_OFFLINE` /
`TRANSFORMERS_OFFLINE`：框架按阶段设置它们，而任务级副本会连 prepare 自己的下载一起
堵死。

单卡。`run.py` 拒绝在已占用的卡上启动、要求 10 GiB 空闲磁盘，并自己拉起检索服务、
退出时回收，所以没有哪个阶段需要跨容器保活一个服务。

用 Docker agent backend。

workspace 代码与 Search-R1 配方 Apache-2.0。
