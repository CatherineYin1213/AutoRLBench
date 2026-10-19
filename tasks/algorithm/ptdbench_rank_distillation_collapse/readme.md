# ptdbench_rank_distillation_collapse

| | |
| --- | --- |
| **Author** | Yaxuan Li |
| **Domain** | `algorithm` |
| **Tags** | `distillation` `ranking` `grpo` `verl` `loss-design` `degeneration` `issue-only` |
| **Framework** | verl，全参数 AdamW |
| **Model** | DeepSeek-R1-Distill-Qwen-1.5B（学生）/ DeepSeek-R1-Distill-Qwen-7B（教师） |
| **Dataset** | DAPO-Math-17k；AIME25 / AMC23 / AIME26 验证 |
| **Environment** | `rank-distillation-cpu`；uv 管理 Python 3.12.12，无 Docker/conda |
| **Resources** | 8 × A800 80GB（完整训练 profile） |
| **Recorded runs** | A、B 各一次真实训练，每次约 11 小时 |
| **Recorded runtime** | 约 11 h / 侧 |

## Task abstract

On-policy 排序蒸馏把教师的候选偏好顺序迁移给学生。目标函数是相邻候选对的 logistic
约束，对候选集合的**共同 logit 平移不敏感**——只把候选 logits 统一减一个常数，loss
不变，而学生给候选集的总概率可以趋近于零。于是排序 loss 在降、实际生成分布却在退化。
两次真实训练里，教师 top-8 的学生总概率从约 0.93 掉到 0.0007 / 0.0004，输出熵从约
1.0 涨到 11.8，三个验证集在 step 140 全为 0%。

这是个 **issue-only** 任务：只诊断，不评分修复。

## The defect

排序目标

$$L=\frac{1}{|B|}\sum_{(x,y)\in B}\frac{1}{|y|(K-1)}
\sum_{t,i}\operatorname{softplus}\left(z_\theta(v_{t,i+1}\mid x,y_{<t})-
z_\theta(v_{t,i}\mid x,y_{<t})\right)$$

只约束候选**之间**的相对顺序，对它们的共同偏移没有约束。因此 loss 改善与生成分布
退化可以同时发生。

### Why nothing catches it

- **数学上的不变性是可证的，因果上的必然性不是。** 在独立 logit 坐标下集外直接梯度
  为零，但这不代表共享参数的网络在更新中其集外输出不变。
- **没有单因素消融**把全部失败唯一归因于全参数更新、长 rollout 或某个超参数。
  oracle 只奖励准确的机制解释和证据，不要求 agent 宣称完成了这种因果证明。
- B 组把 teacher tie rate 降到百万分之一级，失稳依旧发生——排除了"教师精度是主因"。
- 固定前缀的小型 CPU 观测（3 条正常轨迹、48 个位置，FP32）单独复现：排序 loss
  0.9054→0.6575 的同时，候选质量 0.9912→0.7747、熵 0.2490→2.7804。这只是检查，
  不作为完整 benchmark 结果。

## Evidence

| Artifact | What it answers |
| --- | --- |
| `workspace/` | 完整 verl 源码、训练入口、配置、公开测试 |
| `workspace/` 日志 | A/B 两组完整脱敏日志，401 条 step 指标 |
| 生成片段 | 退化后的实际输出 |
| 固定前缀原始指标 | 上述 48 位置的 CPU 探针结果 |
| `environment/source_manifest.json` | 来源与封装前文件哈希 |

| 观测 | A（BF16 教师头） | B（FP32 教师头 + 同分屏蔽） |
| --- | ---: | ---: |
| teacher top-8 学生总概率，step 1→200 | 0.93194 → 0.000667 | 0.93266 → 0.000378 |
| 输出熵，step 1→200 | 1.02176 → 11.75231 | 0.99482 → 11.80022 |
| AIME25，step 100 | 0% | 0% |
| 三个验证集，step 140 | 全 0% | 全 0% |

配置：DAPO-Math-17k、lr=1e-6、batch=32、K=8、seed=42、训练回复上限 7168、200 步。

`workspace/` 不含权重、数据集、缓存、隐藏测试、oracle 或参考修复；原始路径与追踪
服务账号已脱敏，训练数值未修改。

**一处必须保留的复现声明差异：** 未保存精确的启动时源码，`environment/
source_manifest.json` 记录的是封装前文件哈希。另外历史 top-K/top-1 全词表一致率有
独立的仪表错误，已在 agent 文档中明确标为不可用，不把它当作本任务的第二个隐藏问题。

## Reference answer

### `issue` — the diagnosis

完整机制解释必须同时给出：排序目标对候选共同平移的不变性、该方向未被惩罚、以及
共享参数网络下集外输出并非逐坐标独立。

以下四类**不应通过**：

- 泛泛的"缺正则"
- 把问题归因于教师精度
- 只归因于监控/仪表
- 过度因果断言（声称已证明优化器必然选择退化方向）

### `solution`

不存在。本任务不含 solution 目录，不要求参考修复，也不按"某个补丁能否训练达标"
评分。评分不包含修复建议。

## Verification

`evaluations/issue/verifier/test.sh` 是唯一正式入口，复用既有 PTDBench 的 `judge.py`
与 `judge_prompt.md`，按 verdict / root_cause / evidence 三维评分，权重
**0.2 / 0.6 / 0.2**，通过阈值 **0.8**。

隐藏的 `oracle.md` 才是共享 judge 实际读取的标准答案；`gold.yaml` 是同内容的结构化
摘要。评分上限同时写进 oracle，避免只改 YAML 而代码不读。**不评分修复建议。**

判分依赖可信的 LLM 通道：需要 `PTDBENCH_JUDGE_MODEL`、`OPENAI_BASE_URL`、
`OPENAI_API_KEY`，agent 不得获得这些凭据。缺少配置时写
`passed=false, score=null, status=unavailable`，不伪造通过。verifier 所在环境禁止
网络时，平台须在可信宿主侧运行 judge。

**本次未做 LLM-judge 验收**（用户明确要求）。因此没有虚构的 agent 推理轨迹或语义
评分，也不宣称"正确答案通过、错误答案失败"已被真实 LLM 验证。mock judge 只验证
输出结构、失败行为和加权阈值。未运行的 LLM 端到端评测、干净 GPU 复现与因果消融
已在 `validation/status.json` 中标记为未运行。

本地实际结果：schema 严格校验通过；任务契约/数学/证据测试 **14/14**；CPU uv prepare
的首次、重复与 `UV_OFFLINE=1` 模式均通过；A/B launcher dry-run 通过；源数据下载、
离线缓存校验、17917/30 行语义校验与学生模板重建通过。

现有框架测试 **59/60**。唯一失败是 `test_docker_stream_process_writes_live_stage_logs`：
本机 `/etc/profile` 无条件执行磁盘配额查询，向 `sh -lc` 的 stdout 插入内容，导致严格
字符串断言失败。这是已定位的宿主干扰，未修改测试来掩盖。

## Running it

prepare 是唯一允许下载依赖、模型和数据的阶段。

```bash
export PTDBENCH_ENVIRONMENT="$TASK/environment"
export PTDBENCH_WORKSPACE="$STAGED_WORKSPACE"
export PTDBENCH_PREPARED="$PREPARED_DIR"
bash "$PTDBENCH_ENVIRONMENT/prepare.sh"
```

环境传递文件为 `$PTDBENCH_PREPARED/env`，重复 prepare 会替换而非追加。

**默认 issue profile** 只装带哈希锁定的 CPU PyTorch 及其依赖，agent 可离线读代码和
日志、直接导入损失文件跑张量探针。不下载约 18.8 GB 权重，也不装 GPU 训练依赖。

**完整训练 profile**：

```bash
PTDBENCH_PREPARE_PROFILE=training bash "$PTDBENCH_ENVIRONMENT/prepare.sh"
cd "$PTDBENCH_WORKSPACE"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash opd_train/run_recorded.sh A
# B 用相同配置与初始模型，切换教师精度和 tie 处理
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash opd_train/run_recorded.sh B
```

训练 profile 需要 Linux x86_64、兼容 CUDA 13 的驱动、CUDA 工具链和足够磁盘内存。
`requirements-training.txt` 是记录环境的冻结快照，用 `--no-deps` 安装，源码走
`PYTHONPATH`。**本次未对该 GPU profile 做干净环境重训验收。**

prepare 期间一并处理三件容易被忽略的事：DAPO 提示结尾的 `\\boxed{{}}` 还原为
`\\boxed{}`（17917 行经 canonical JSON SHA-256 验证）；AIME26 由固定 JSONL 重建并
验证 30 行；学生 tokenizer 追加 `<think>\n`，上游原件另存为
`tokenizer_config.upstream.json`，不改动已校验的下载原件，token→ID 映射不变。

下载器对逐文件做 SHA-256 与字节数双重校验，失败立即退出、不降级。缓存只用于下载
复用，交付文件复制到 `workspace/prepared`，不用指向宿主缓存的软链接。uv 的依赖
缓存与解释器须单独保留，只有模型缓存不足以重建 Python 环境。

本地 `--backend local` 不会自动执行 prepare，也不提供文件系统/网络隔离；正式评测
必须由外层平台保证阶段隔离。uv 只管理 Python，不承担沙箱职责。

### 授权

任务代码 Apache-2.0。**本包不重新分发数据或权重**，19 份外部文件的固定 URL、
revision、字节数与 SHA-256 全在机器可读的 `environment/assets.json` 里（它也逐条记录
license）；CPU 依赖的所有允许发行物哈希在 `environment/requirements.txt`。

涉及的 license 共五类，其中三类需要留意——数据集的上游仓库/card 授权与题目本身的原始
权利是分开的：

| license | 适用 |
| --- | --- |
| MIT | 两个 DeepSeek-R1-Distill-Qwen 模型的全部文件 |
| BSD-3-Clause | flash-attn 2.8.3.post1 源码包 |
| Apache-2.0（上游 BytedTsinghua-SIA/DAPO-Math-17k） | `dapo-math-17k-processed.parquet`；**处理后的转换版本未单独声明授权** |
| Apache-2.0（dataset card） | AIME26 源 JSONL；**底层竞赛题目保留原始权利** |
| n/a | AIME25 / AMC23 的 `test.parquet`——托管快照没有独立的数据集授权，**竞赛题面保留原始权利** |
