---
name: package-task
description: >-
  Packages a buggy training workspace into a AutoRLBench task with environment
  (shared image + prepare.sh), issue and solution evaluations, workspace,
  metadata.yaml, and readme.md.
  Use when the user asks to package, build, or convert a workspace into the
  task format under tasks/, for any model, framework, or dataset.
---

# Package a training workspace into a AutoRLBench task

把有问题的训练工作区收成 `tasks/<family>/<task_id>/`（新任务也可以直接放 `tasks/<task_id>/`）。适用所有模型、框架与数据集。不产生 `agent_trace.json`。

环境走**共享环境镜像**：任务不自带 Dockerfile，`metadata.environment.image` 写 `docker/images/` 下某个 `<name>.Dockerfile` 的名字，`environment/` 只有 `prepare.sh`（唯一联网阶段，下模型下数据）。

细节模板见 [reference.md](reference.md)。填好的输入示例见 [examples.md](examples.md)。仓库侧的权威说明是 `README.zh.md`（元数据 schema、prepare 阶段）、`docker/README.zh.md`（环境镜像）和 `autorlbench/tasks/validator.py`（校验规则）。

## 用户必填输入

缺一项就停。

- 来源形态（见下节「三种来源形态」）：`source_workspace` + `buggy_commit` / `baseline_commit`，或 `delivery_dir`，或 `source_workspace` + `bug_root_cause`
- 目标目录、`task_id`、title
- 模型名称、数据集名称、框架、算法
- 启动脚本相对 `workspace/` 的路径、位置参数、必要环境变量
- 容器内模型路径、训练集路径、验证集路径
- 失败运行 logs 的路径；来源有 wandb 则保留
- 来源里有没有**对照侧**（同一次对照实验里那份正确实现的代码与结果）。有则它整份进 oracle，不进 workspace
- 环境镜像名（`docker/images/` 下 `<name>.Dockerfile` 的 `<name>`，现有 `verl`、`slime-rc0`、`slime-v0.3.2`、`rank-distillation-cpu`）
- 任务要用的卡（写进 `metadata.environment.env` 的 `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES`）
- 每份资产：HF repo、40 位 commit sha revision、`--allow` 清单、需校验文件的 SHA-256 与字节数、`--dest` 落点；用内置配方时给 recipe 名与校验值
- 评测集键、出现位置（文字 log、离线 wandb 或两者）、一条含该键与数字的 log 样例
- `final_step`、`historical_final_baseline`、`required_final_score`
- 门槛的**校准依据**：这个数是从几次运行上量的，至少一侧几次。只有每侧一次时要按单次标定写（见「门槛的校准依据」）

评分方式可省略。省略时分数全部看评测集键。若指定，则列出每一项参考分的键名、门槛、权重、出现位置、log 样例，以及及格规则；这些参考分不限于训练分。

`score_mode`（`best` 或 `final`）与 `required_final_score_max` 可省略，省略时按 `best` 且不设上限。

## 三种来源形态

先判断来源是哪一种，后面的流程差别都从这里来。

### 1. 一个 git 仓库的两个 commit

最常见。`git diff <buggy> <baseline>` 给出根因，workspace 对齐 buggy 那一次。做法见第 1 步。

### 2. 一份交付包：两个代码副本，各自带一次真实运行

`raw_tasks/` 下的交付包常是这种形态：`workspace_broken/` 与 `workspace_fixed/` 并列，各有一份完整代码和一份跑完的 `results/`，共享目录（`shared/data/`、`shared/runtime/`）只有一份。**没有 git 历史**，根因靠两份代码互 diff。

先量清楚两侧差多少：

```bash
diff -rq workspace_broken workspace_fixed          # 先看文件清单差异
diff -u workspace_broken/<改动的文件> workspace_fixed/<改动的文件>
```

差异往往只有一两个文件、几行代码（另加两侧各自的日志与结果，那些不是根因）。把代码差异从日志差异里分出来后，按形态 1 的第 3 步办。

**只发布有缺陷的那一侧。** 两侧并列放进 `workspace/` 等于把答案摆在同一层目录里，`diff -r` 就能拿到。照下面这样分：

| 交付包里的东西 | 去处 |
| --- | --- |
| 缺陷侧的完整代码 + 启动脚本 + 数据 | `workspace/` |
| 对照侧的代码（尤其被改动那个文件） | `evaluations/solution/oracle/{code_target,format_gold}/` |
| 对照侧的运行结果与指标曲线 | `oracle/dirty_reference/`、`oracle/recorded_comparison/` |
| 两侧的对照说明、隔离审计、检索/质量预检 | `oracle/recorded_comparison/` |
| 交付包级的脚本（`setup_env.sh`、`download_models.sh`、`SHA256SUMS`、包 README） | 丢掉，改写成 `prepare.sh` 与 `metadata.yaml`；它们带的 revision 与 sha 搬进 `prepare.sh` 和 manifest |
| 两侧共用的运行环境锁文件（`pyproject.toml` / `uv.lock`） | 进新镜像的构建上下文，见「Environment」 |

对照侧是这份任务最值钱的东西：它让门槛不必靠协议估计，而是有实测的对侧值。别把它当参考文档摆在 `readme.md` 里，要作为资产封印进 oracle。

### 3. 一份历史快照，没有对照侧，也没有注入缺陷

仓库里已有这种任务（`tasks/task_train_too_hard`、`tasks/ptdbench_rank_distillation_collapse`）。没有 diff 可依赖，根因由用户给出（`bug_root_cause`），用户没给就要他确认，不能自己从日志倒推一个。这类任务常常只发 issue 档——solution 档要有一条「修好之后指标应到多少」的线，没有对照侧就没有这条线。

## 从来源里挖根因

拿到 diff 或两侧代码后，根因的判据不只有代码差异。交付包里的这些文件能直接证实「缺陷真的生效了」，写进 oracle 的 Observable impact 比任何推断都硬：

- **检索/输入侧的记录**（`retrieval.jsonl` 之类）：如果缺陷影响的是模型的输入，这里能看到症状本身——例如所有返回文档得分都是 0、每个 query 返回同一组文档
- **预检报告**：交付包里常有对照两侧输入质量的脚本产物（recall@k、命中率）。缺陷侧 `0.0` 对对照侧 `0.809` 这种对比，是最短的一条证据链
- **逐步指标目录**（`rollouts/metrics_*.json`、`outputs/`）：曲线之外还有 `response_length/mean`、`env/*` 这类旁证。它们是「协议正常、只是输入没用」还是「整个训练坏了」的分水岭
- **隔离审计**：并行跑的两个 run 之间有没有串线。这份文件也是 `readme.md` 里「哪些外部因素被排除过」的出处

有一类结论要格外小心：**缺陷组也涨了**。上面那类交付里的缺陷组常常仍然从 3.1% 涨到 15.6%。根因是「比对照低」而不是「没有学习」。oracle 里要明写这一点，并在 `standard_answer.txt` 里把「训练没有提升」列为错误归因——不然 judge 会给一个说反了现象的答案满分。

## 派生资产与随包资产

数据、索引、权重怎么进任务，按「能不能从别的东西算出来」分：

| | 进 git（`workspace/`） | prepare 阶段现建 |
| --- | --- | --- |
| 判据 | 别处拿不到，或重建会改变对照 | 由已随包的东西 + 钉死的模型算得出来 |
| 例 | 训练/验证 parquet、语料 jsonl | FAISS 索引、torch_dist 权重、arrow 缓存 |
| 依据 | 放进 `environment/data_manifest.json`，`prepare.sh` 校验它 | 由 `environment/<脚本>.py` 生成，产物进 `$PTDBENCH_DATASETS` 缓存 |

判断标准是性价比：一个索引 37MB，而它的语料 6.6MB 且随包，那索引就该现建。反过来，语料本身若要从几十 GB 上游拼，就该随包或走 `ptdbench_fetch_url`。

派生脚本要写成**自包含**的一份，不要调 `workspace/` 里的构建器，原因是三条，都写进脚本注释：

1. `workspace/` 是 agent 可改的树。用 agent 能改的代码去做资产，资产质量就跟着 agent 走
2. 上游构建器常常无条件 `.cuda()`，而 prepare 阶段可能没有 GPU（harbor 后端把 prepare 跑成 build-time 层）。自包含脚本要能退回 CPU
3. 上游构建器可能还要拉一堆 train 侧的依赖（`datasets`、`tqdm`、memmap 写入），准备阶段不需要

派生资产的校验只做**与缺陷无关的那一侧**。这是准备阶段唯一容易踩的坑：如果 agent 能读 `/ptdbench/environment`，在那里断言「查询侧检索质量」就等于告诉他缺陷在哪。文档向量在两侧通常完全相同，所以断言条数、维度、有限性、单位范数这些就够了。断言写得再严也不能碰查询侧。

产物落点与缓存：

```bash
# $PTDBENCH_DATASETS 在没有配缓存时是空串，会落到 / 根下。给个回退。
CACHE="${PTDBENCH_DATASETS:-$PTDBENCH_WORKSPACE/.index-cache}/<name>"
MARKER="$CACHE/.complete"

if [ ! -f "$MARKER" ] || [ ! -s "$CACHE/<file>" ]; then
    rm -rf "$CACHE"; mkdir -p "$CACHE"
    python3 "$PTDBENCH_ENVIRONMENT/build_index.py" --out "$CACHE/<file>"
    touch "$MARKER"
fi
rm -f "$PTDBENCH_WORKSPACE/<dest>/<file>"
cp -al "$CACHE/<file>" "$PTDBENCH_WORKSPACE/<dest>/<file>" 2>/dev/null \
    || cp -a "$CACHE/<file>" "$PTDBENCH_WORKSPACE/<dest>/<file>"
```

三点：用 `.complete` 标记而不是认产物文件在不在（上次被 kill 的 prepare 会留下半截产物，认文件会把它当成做好了）；`cp -al` 硬链，同一文件系统上第二个 run 不花字节，跨文件系统退化成真拷贝；脚本自己写文件时用「写临时名再 `os.replace`」，半截的索引比没有索引糟得多——`prepare.sh` 的守卫和 `run.py` 的资产检查都会把它当成已就绪，然后在计时的重跑里炸。

派生产物**不逐字节可复现**（batch 组成、fp16/fp32、cuBLAS kernel 都会动最后几位）。所以不要给它写 `--expect`，改在 `data_manifest.json` 里记下原件的 sha256 并说明「重建的是同一个索引，不是同一次字节」。

## 冻结证据的瘦身与脱敏

交付包里的 `results/<run>/` 常常几十 MB（逐步 rollout、wandb 二进制），而且**带着答案**。两件事都要做。

瘦身按「诊断需不需要」取舍，不是按大小：

| 留 | 丢 |
| --- | --- |
| `train.log`、`launch.log`、服务日志、`experiment.json` | 大部分 `rollouts/step_*.jsonl`（留首、次、中间、末步做样本） |
| 全部 `rollouts/metrics_*.json`（逐步曲线，每个几 KB） | wandb 的 `*.wandb` 二进制（几 MB，同样的曲线 log 里也有） |
| 输入侧的 jsonl 记录，**裁剪成十几条** | wandb `latest-run/`（离线 run 目录的重复拷贝） |
| 离线 wandb 的 `files/`、`logs/` | 原始数据分片（出处记进 manifest） |
| 新加一份机器可读的曲线汇总（对应 `outputs/metrics.json`） | checkpoint、权重 |

脱敏是硬要求，因为交付包里两侧的目录名就叫 `broken` / `fixed`，这几个词会散落在日志、`experiment.json` 和 wandb 元数据里。要清的东西：

- 目录名里的 `broken` / `fixed`，以及 `experiment.json` 的 `"mode"` / `"validation_retriever"` 这类直接写着哪一侧的字段
- 绝对路径（`/mnt/data/<user>/...`）——换成容器里的 `/workspace/...`
- 环境里的用户名、主机名、内网 IP（`wandb-metadata.json` 里有）
- 端口与卡号：两侧 `config.json` 里的 `gpu` / `port` 常常正好是对照的（4/8004 对 5/8005），这本身就是个 tell
- `source_sha256` 里指向对照侧修复文件的那一行（`maintainer/fixed_*.py` 之类）
- 文件头/启动横幅里的 `mode=broken` 这种字样

**不要动任何指标数值。** 脱敏只改标签与路径。

每一条替换都记进 `environment/source_manifest.json`，带改前改后的 sha256：

```json
{
  "schema_version": 1,
  "source": "<交付包路径>",
  "staged_side": "有缺陷的那一侧",
  "unchanged": { "<目录>/": { "note": "逐字节复制，含缺陷", "sha256": "..." } },
  "edited_for_packaging": [
    { "path": "workspace/run.py", "action": "packaging-adjusted",
      "why": "为什么必须改，以及不改会怎样",
      "sha256_before": "...", "sha256_after": "..." }
  ],
  "dropped_from_the_delivery": [ { "path": "...", "why": "去了哪里，为什么" } ],
  "frozen_evidence": { "substitutions": [], "files": [] }
}
```

`work/` 里放一份能重跑的打包脚本（瘦身 + 替换 + 出报告），并把报告喂给写 manifest 的那一步。脱敏靠人眼过一遍必漏，靠脚本才能把「改了哪几处、每处几次」数出来。

脱敏完扫一遍，命中的必须是**上游原本就有的词**，不是我们的替换漏项：

```bash
# 期望：只剩上游源码里的无关命中，以及语料正文里的自然语言命中
grep -rIl 'broken\|fixed\|<原路径片段>\|<原用户名>' workspace/
```

扫完要把剩下那几处命中的**原因**写进 `source_manifest.json`（"这是上游 `verl/utils/hdfs_io.py` 的原有措辞"），不然下一个人会以为没扫干净。

对账：`experiment.json` 里记着的 `source_sha256` / `input_sha256` 是最好的自检。随包文件应当逐条对上；prepare 阶段才有的（索引、权重）对不上是正常的，在那份 JSON 里加一条 `input_sha256_note` 说明，别让后来者对着一棵正确的树得出「数据坏了」的结论。

## 目录

```text
<task_id>/
├── environment/prepare.sh
├── environment/build_index.py               # 派生资产由本地脚本生成时才有
├── environment/data_manifest.json           # 数据随包走时才有
├── environment/source_manifest.json         # 打包对来源做过改动时才有
├── environment/docker-compose.yaml          # 需要多卡或大 shm 且要走 harbor 时才有
├── evaluations/issue/instruction.md
├── evaluations/issue/oracle.md
├── evaluations/issue/verifier/judge_prompt.md
├── evaluations/issue/verifier/judge.py
├── evaluations/issue/verifier/test.sh
├── evaluations/issue/verifier/README.md
├── evaluations/issue/verifier/gold.yaml     # 可选，只给 reviewer 看
├── evaluations/solution/instruction.md
├── evaluations/solution/oracle/README.md
├── evaluations/solution/oracle/task.yaml
├── evaluations/solution/oracle/classification.json
├── evaluations/solution/oracle/standard_answer.txt
├── evaluations/solution/oracle/historical_evidence_manifest.json
├── evaluations/solution/oracle/dirty_reference/historical_metrics.json
├── evaluations/solution/oracle/recorded_comparison/   # 对照侧的原始记录，交付包来源时才有
├── evaluations/solution/oracle/{code_target,format_gold}/  # 密封的 baseline 实现
├── evaluations/solution/verifier/test.sh
├── evaluations/solution/verifier/README.md
├── evaluations/solution/verifier/resources/grader_spec.json
├── evaluations/solution/verifier/resources/README.md
├── evaluations/solution/verifier/tests/grade_submission.py
├── validation/status.json                   # 验过什么、没验什么
├── validation/README.md                     # 打包脚本说明，随之一起留下时才写
├── workspace/
├── metadata.yaml
└── readme.md
```

没有 `environment/Dockerfile`、`environment/requirements.txt`，也没有 `environment/hydrate-assets.sh`。`tasks/` 下已经没有任何任务留着它们（`tasks/reward_design/*` 那两个最后的 legacy 任务已于 2026-09-23 迁完）；新任务不要照抄。

`validation/` 不是框架要求的目录，是打包留痕：`status.json` 记本机验过什么（validate、grader 回放、哈希、泄漏扫描）、没验什么（镜像构建、prepare 全流程、真实重跑）以及每一项的下一步。写它是因为打包机常常没有 GPU，`readme.md` 里那些没跑过的数字要有人替它们标注来源。没有 GPU 的机器上尤其要写。

## 流程

1. 认定来源形态（见上）。形态 1 跑 `git diff <buggy> <baseline>`，形态 2 跑 `diff -rq workspace_broken workspace_fixed` 再逐文件 `diff -u`，形态 3 用用户给出的 `bug_root_cause`。无论哪种，根因只取代码与启动脚本（含它们直接引用的训练设定、超参、奖励与数据处理）。排除 `logs/`、`wandb/`、`outputs/`、`*.log`、checkpoint、权重与数据文件的不同。用户标成无关的 hunk 也排除。过滤后没有实质变更就停。
2. 原样复制 issue 的 `instruction.md`、`judge_prompt.md`、`judge.py`、`test.sh`，以及 solution 的 `verifier/test.sh`。来源见 [reference.md](reference.md)。复制后用 `cmp` 确认与来源一致，并保持 LF。
3. 用过滤后的 diff 重写 `evaluations/issue/oracle.md`、`evaluations/solution/oracle/README.md`、`task.yaml`、`standard_answer.txt`。`metadata.yaml` 的 `failure_reason` 与 `readme.md` 的 Canonical failure reason 也按这份 diff 写成几句话，并写上用户给出的指标现象。用户若另写了失败原因，不作为根因依据。
4. 用用户本次输入重写 `solution/instruction.md`、`grader_spec.json`、`resources/README.md`，并把失败运行的指标序列落成 `oracle/historical_evidence_manifest.json`。`grade_submission.py` 先按下面「grader 能不能复用」判断要新写还是照抄。
5. 组 workspace（形态 2 要按「只发布有缺陷的那一侧」分好去处）、冻结证据的瘦身与脱敏、`environment/prepare.sh`、`metadata.yaml`、`readme.md`。`readme.md` 按 [readme-format.md](readme-format.md) 的六节格式写，空白骨架见 [readme-template.md](readme-template.md)。
6. 用用户给的 log 样例跑一次提取，确认能抽出所填的键与数字。抽不到就改 `grade_submission.py`，直到样例通过。来源带对照侧时，**两侧的历史 log 各回放一次**：缺陷侧必须不过，对照侧必须过。这是门槛唯一的离线自检。
7. 跑 `python3 -m autorlbench validate <task_dir>`，必须零 error。有 docker 和网络的机器上再用 `./docker/warm-cache.sh` 跑一遍 `prepare.sh`（它会按需构建镜像，并把全部任务的 prepare 各跑一次）。
8. 本机没有 GPU 时，把没验过的东西逐条写进 `validation/status.json`：镜像构建、prepare 全流程、派生资产真的建得出来、一次真实重跑、门槛的方差。每一项写清楚「为什么没验」和「下一步怎么验」。

## Issue 归因

根因只来自第 1 步的 diff。每一项写路径或符号、kind、observed（buggy）、expected（baseline）。`failure_reason` 由该 diff 写成一句话，不照抄用户自拟的摘要。

log、wandb、指标曲线只写进 Observable impact，不写成根因。

多于一个实质变更时，`oracle.md` 必须写上：

```text
Listed causes are all essential. Identifying only part of them must receive partial root_cause_score.
```

这句对齐 `judge_prompt.md`（多个根因只点出一部分要给部分分）。权重维持来源文件的 0.2 / 0.6 / 0.2，及格线 0.8。不要写「点出任一项即满分」。

`gold.yaml` 只是给 reviewer 的对照，机器评分读 `oracle.md`。写了就必须和 `oracle.md` 的 verdict、root_cause 一致。

## 阈值怎么定

门槛写在 `grader_spec.json`，但它的依据要写进 `verifier/README.md`、`resources/README.md`、`solution/oracle/task.yaml` 和 `readme.md`。有对照侧时这是纯算术，把两侧的曲线摆出来：

| step | 缺陷侧 | 对照侧 |
| ---: | ---: | ---: |
| 0 | 0.03125 | 0.13281 |
| … | … | … |
| 32 | 0.15625 | 0.32813 |

取门槛的三条规则：

1. **两侧都留余量。** 门槛要在缺陷侧上限之上、对照侧下限之下，并报出两个余量（上例 0.084 / 0.088）。哪边余量薄要说哪边。
2. **压过对照侧的起步点。** 对照侧 step 0（没训练的那一点）如果在门槛附近，那就等于「改对代码不用跑完也能过」。门槛要高于它，让「修好」和「跑完」都得做到。
3. **上界挡编造。** `required_final_score_max` 按指标空间取（准确率就是 `1.0`），作用只是不认离谱的值，不是评优线。

**说清门槛是从几次运行上量的。** 这是这个 skill 里最容易被含糊过去的一件事：

- 每侧多次（`slime_search_r1_nomask` 是四次）：可以报最薄余量，可以给方差
- **每侧只有一次**：可以说余量，不能说方差。就在 `verifier/README.md` 和 `readme.md` 里明写「本门槛只有每侧一次运行作为标定」，把两个余量写出来，再写清后果——落在门槛附近的结果属于未测区间，读日志再判；补一对重跑是下一步

不要在只有单次标定时把话说得像有多次。审这份任务的人会照着 `verifier/README.md` 决定要不要重跑，含糊等于把风险留给别人。

缺陷侧方差大的任务（同一份代码两次跑能差出一大截）要额外注意：定门槛的水平位置要选在两次缺陷侧 run **都完整跑过**的步数上。在更远的步数上缺陷侧可能还没跑完就被杀了，那段没有上限可比，门槛在那一段无据可依。

## Solution

`instruction.md` 保留「在 `/workspace` 修好、前台跑完、不用网络、不 detach、数据与模型固定、log 只认 `$PTDBENCH_RESULTS/experiment_logs`、禁止编造指标」。启动脚本、参数、模型路径、训练集路径、验证集路径改成用户本次输入。步数与评测间隔被固定时，一并写进「The effective run must use」。

修好之后那个「改哪里」也在这一档说明：入口脚本要写得让 agent 不必猜调用方式。上面「门槛与重跑的对齐」给的命令行就是这一节的主体。

## 门槛与重跑的对齐

solution 档的重跑预算常常和冻结证据的步数不一样（交付包可能跑了 200 步，而重跑只给 100 步）。这时：

- `grader_spec.json` 的 `final_step` 写**重跑**的最后一步，不是冻结证据的最后一步
- 门槛只在两者共有的区间上校准，并在 `verifier/README.md` 里说明为什么停在更近的那一步
- `historical_evidence_manifest.json` 记完整的历史曲线，标注哪一段是超出的

另一件要对齐的是**重跑怎么落地**。workspace 里的入口脚本常常自己管着一堆事（拉起并发服务、等就绪、结束时清理）。这时 `instruction.md` 要：

- 给出**一条完整命令行**，用 `--runs-dir "$PTDBENCH_RESULTS/experiment_logs"` 之类把日志指到计分目录
- 写明这个入口自己会管那个服务，不要另起、也不要绕开
- 顺手指明重跑参数从哪读（`config.json` 之类），并写「不要改」

如果入口脚本把 checkpoint 也写在日志目录下，改掉它：几 GB 的权重放进计分目录既没人读也不该进结果。让权重留在 workspace 里（run 名的子目录），日志才进 `$PTDBENCH_RESULTS`。

`grader_spec.json` 写入用户本次给的 `task_id`、评测集键、`final_step`、`historical_final_baseline`、`required_final_score`、`log_roots: ["experiment_logs"]`，需要时加 `score_mode` 与 `required_final_score_max`。用户指定了评分方式时，参考分按扁平的 `<name>_key` / `required_<name>` / `<name>_weight` 三件套写，不要用 `components` / `pass_rule`。

未指定评分方式时，分数全部看评测集：按 `score_mode` 取该键的最高有限值（`best`）或 `final_step` 那一点（`final`），达到 `required_final_score` 则 score 为 1，否则为 0。用户指定了参考分时，按该输入的权重与及格规则计算，不另行假设必须两项都过线。

`grade_submission.py` 从这份 spec 读键与门槛，按用户给的 log 样例撰写提取。键出现在文字 log、离线 wandb，或两者，以用户输入为准。

### grader 能不能复用

先看仓库里现成的 `grade_submission.py` 能不能直接吃这个任务的日志。它们做的是同一件事：在 `$PTDBENCH_RESULTS/experiment_logs` 下扫文本日志，正则抽出「`step:N - ... <键>:<数字>`」或「`eval N: {...}`」两种形态，按 `score_mode` 取一个值，与门槛比。**新任务的日志形态和某个现成 grader 的两种模式之一对得上时，照抄它，不要重写**：

```bash
cp tasks/<现成任务>/evaluations/solution/verifier/tests/grade_submission.py \
   tasks/<task_id>/evaluations/solution/verifier/tests/
```

`spec` 是唯一需要按任务改的地方（键、门槛、`final_step`、`historical_final_baseline`）。照抄的收益不只是省事：正则、ANSI 剥离、符号链接拒绝、`MAX_FILE_BYTES`、原子写 `result.json` 这些都是踩过坑才长出来的，重写一遍等于把那些坑再踩一次。照抄的代价要写清楚：两个任务的 `grade_submission.py` 从此是同一份，将来改要一起改；在 `verifier/README.md` 里写明「本文件与 `tasks/<来源任务>/...` 逐字节相同」。

复用前必须验证形态，不能想当然——拿两侧的真实 log 各跑一次（第 6 步）。常见的一处不高明但不影响判分的地方：它的正则带 `re.DOTALL`，匹配可能从更早的一个 `step:N` 标签跨行接到后面的键，于是 `best_step` 报的不是最高分所在的那一步。`score_mode: best` 下判分只看数值，所以这不影响 pass/fail，但要在 `verifier/README.md` 里写明，免得读报告的人拿 `best_step` 当「最好那一步」。

对不上时才自己写，并且按用户给的 log 样例写，不要照抄旧任务的 log 格式分类。

输出 `experiment_metrics.json`、`metrics.json`、`result.json`（含 `passed` 与 `score`）。`resources/README.md` 用同一组键与门槛说明上述规则；`verifier/README.md` 再写一遍门槛怎么来的（对着历史曲线的哪几个点校准）。

`solution/oracle/README.md`、`task.yaml`、`standard_answer.txt` 按同一份 diff 重写。修复动作是把那些变更改回 baseline。`historical_evidence_manifest.json` 记那次失败运行的指标序列，是 reviewer 复核门槛的依据。来源有对照侧时再加一份 `oracle/dirty_reference/historical_metrics.json`，两侧的曲线放在一起，门槛的余量就是这两条线的间距。

`task.yaml` 里另外记两件 reviewer 会问的事：这次标定是每侧几次（`calibration_runs_per_side`），以及哪些文件是随包的、哪些由 `prepare.sh` 交付。`classification.json` 记分类名与门槛来源。

## Workspace

放入有缺陷的那一侧的完整代码与启动脚本。保留用户指出的失败运行 logs；来源里有 wandb 就一并放入。

排除 `.git`、`__pycache__`、`agent_trace.json`、权重、checkpoint，以及将由 `prepare.sh` 下载的训练集与验证集文件。workspace 里没有那次失败运行的 log 就停。

启动脚本在 workspace 里的路径必须等于 `metadata.training.script`，否则 `validate` 报 error。

来源是交付包时，对照侧的代码（核心文件）进 `evaluations/solution/oracle/{code_target,format_gold}/`，对照侧的运行结果与指标曲线进 `oracle/dirty_reference/` 与 `oracle/recorded_comparison/`。冻结证据要按「冻结证据的瘦身与脱敏」处理后再进 workspace。

## Environment

`environment.image` 从 `docker/images/` 现有的镜像里挑，按框架与版本对上任务录制时的栈。镜像是多个任务共享的，内容就是它们的 ground truth 环境：不要为新任务改已有的 Dockerfile。现有镜像都不合适时，新加一个 `docker/images/<name>.Dockerfile`（`FROM` 钉 digest，末尾照抄其他 Dockerfile 的 AutoRLBench tooling 段），做法见 `docker/README.zh.md`。只缺一两个纯 Python 包的，在 `prepare.sh` 里 `pip install --target` 到 `$PTDBENCH_WORKSPACE` 下，再往 `$PTDBENCH_PREPARED/env` 写一行 `PYTHONPATH=...`（照 `tasks/reward_design/task_evaluator` 的做法；prepare 容器对镜像 site-packages 的改动带不到后两个阶段）。

来源的栈没有现成镜像时，需要自建的也写进 skill 流程，并标注「镜像未构建」。`metadata.yaml` 里写的名字必须是 `docker/images/` 下已存在的 `<name>.Dockerfile` 里对应的名字；`autorlbench validate` 会拒掉不存在的名字。因此，**哪怕镜像还没构建，Dockerfile 也要先写出来**，留待后续在有 GPU 的机器上实际构建，并在构建时把 `FROM` 的标签换成 digest。

现有的镜像名：`verl`、`slime-rc0`、`slime-v0.3.2`、`rank-distillation-cpu`（CPU 专用的 torch 环境，无 CUDA）。**不再有 `python3.11-slim`**——它从来没有对应的 Dockerfile，是 legacy 字段遗留的错误名字，不要在新任务里用它。

`prepare.sh` 跑在独立的 prepare 阶段容器里：root、镜像自带的解释器、**唯一有网的阶段**，`$PTDBENCH_WORKSPACE` 可写。用 `ptdbench_fetch_hf` / `ptdbench_fetch_url` / `ptdbench_dataset` 三个命令取资产，每一份都要 `--expect` 与 `--expect-size` 硬校验，HF revision 必须是 40 位 commit sha。`--dest` 写到 `$PTDBENCH_WORKSPACE/<启动脚本实际读的路径>`，不要让任务脚本去引用 `$PTDBENCH_MODELS` / `$PTDBENCH_DATASETS`——那两个只是缓存，harbor 后端跑完 prepare 就删。

`prepare.sh` 里**不要写 `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`**。断网这件事由框架按阶段设（`autorlbench/runtimes/base.py` 的 `_network_policy_env`），agent 和 verifier 阶段才会带上这两个变量；写在 `metadata.environment.env` 里是三个阶段一视同仁，会把 prepare 自己的下载也堵死。

要用的卡写在 `metadata.environment.env` 里，值必须是带引号的字符串。不在 `prepare.sh` 里设 `CUDA_VISIBLE_DEVICES`，也不写 `PTDBENCH_*`（validator 会拒掉）。

需要多卡或大 `/dev/shm` 且要走 harbor 后端时才加 `environment/docker-compose.yaml`；它对原生 docker 路径无效，照抄现有 slime 任务那份即可。

校验值或 revision 未给齐时，`prepare.sh` 保留明确待填字段，并在 `readme.md` 标成未发布。

## metadata.yaml 与 readme.md

字段结构见 [reference.md](reference.md)。`metadata.yaml` 的所有字段值都必须是简单标量（字符串、数字、布尔）、列表或嵌套 mapping。**`autorlbench/utils/yaml.py` 是一个轻量解析器，不支持 `>-`、`|` 块标量**（即多行折叠或原文缩进文字）：写了 `>-` 会在 `autorlbench validate` 时报 `unexpected indentation`，必须改成单行字符串。如果 `failure_reason` 很长，用引号包裹整行，不要用 `>-`。

`training.script` 写相对 `workspace/` 的启动脚本路径，需要位置参数时加 `script_args`（列表，一项一个参数，全部带引号）。`environment` 写 `image` 与 `env`；旧的 `conda` / `requirements` / `dockerfile` 字段会被 validator 拒掉。

`readme.md` 的 Environment requirement 节写明用哪个环境镜像、为什么是它、`prepare.sh` 交付了哪几份资产、要几张卡。不要再写 External dataset asset 那种镜像下载表：资产大小与 SHA-256 只出现在 `prepare.sh`。

来源是交付包且镜像未构建时，在 `readme.md` 里明写「镜像未构建」（例如 `UNVERIFIED: this image has never been built.`），并在 `validation/status.json` 里把「镜像构建」列为未验项，给出下一步操作。

## 完成前检查

- 骨架齐全，且没有 `agent_trace.json`、没有 `environment/Dockerfile`、没有 `requirements.txt`、没有 `hydrate-assets.sh`
- issue 四档与 solution `test.sh` 与来源 `cmp` 一致
- oracle 根因与过滤后的 diff 一致；多根因时含部分分那句；写了 `gold.yaml` 则与 `oracle.md` 一致
- `grade_submission.py`：要么与来源逐字节相同（在 `verifier/README.md` 里注明），要么能从样例抽出用户给的键
- 两侧历史 log 各回放一次 grader：缺陷侧 exit 1，对照侧 exit 0（来源无对照侧时只回放缺陷侧）
- workspace 含启动脚本与 logs；有 wandb 来源时 workspace 内也有；冻结证据已脱敏
- `grep -rIl 'broken|fixed|<原路径片段>|<原用户名>' workspace/` 只剩上游无关命中
- `metadata.training.script` 指向的文件在 `workspace/` 里真的存在
- `metadata.environment.image` 是 `docker/images/` 下已有的名字
- `metadata.yaml` 没有 `>-` 或 `|` 块标量；所有 `env` 值都是字符串（`"0"` 而不是 `0`）；没有 `PTDBENCH_*` 也没有 `HF_HUB_OFFLINE`
- `prepare.sh` 里每份 HF 资产都有 `--expect` 与 `--expect-size`，revision 都是 40 位 sha；派生资产用 `.complete` 标记守卫
- `python3 -m autorlbench validate <task_dir>` 零 error
- `validation/status.json` 里诚实地列出本机没验过的东西和原因
