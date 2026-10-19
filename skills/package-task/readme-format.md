# Task README format

每个任务一份 README，`raw_tasks/<task>/README.md` 与 `tasks/<family>/<task_id>/readme.md`
用同一套骨架 —— 第 2 步的转换因此主要是填空与收紧，而不是重写。

读者是**维护者与审稿人**，不是 agent。agent 看到的是 `evaluations/<档>/instruction.md`。
这份文档可以写根因、写门槛、写对照侧；但它不能被挂进 agent 阶段。

照着 [raw_tasks/raw_task_example/README.md](../../raw_tasks/raw_task_example/README.md)
写，那是填好的一份。要空白骨架时复制
[readme-template.md](readme-template.md)，占位符与注释都在里面。

## 骨架

```text
# <task-id>

<身份表>

## Task abstract     缺陷一段话，末尾给实测数字
## The defect        根因 + 代码 + 为什么它不报错
## Evidence          随包的产物，以及它们显示了什么
## Reference answer  issue 档的判词 + solution 档的补丁
## Verification      判据、门槛、门槛的校准依据
## Running it        怎么跑起来，以及缺哪些东西
```

六节，顺序不要动。审稿人按这个顺序读：**坏在哪 → 为什么没被发现 → 证据在哪 →
正确答案是什么 → 怎么判 → 怎么复现**。

**不要加节。** 环境、来源、局限这些不单列 —— 环境进身份表，来源进身份表的
`Recorded runs`，局限写在它所属的那一节里（门槛的局限写在 Verification，
环境的局限写在 Running it）。单列成节只会把一份两百行的文档撑成五百行。

## 各节的判据

### 身份表

紧跟标题，两列，不要 bullet 列表。字段照抄 `metadata.yaml`，不要另立说法。

| 字段 | 内容 |
| --- | --- |
| **Author** | 贡献者 |
| **Domain** | 任务族（`environment`、`reward`、`algorithm`…） |
| **Tags** | 反引号包的小写短词，便于检索 |
| **Framework** / **Model** / **Dataset** | 各一行 |
| **Environment** | 环境文件 + 关键 pin，一行写完 |
| **Resources** | 几张什么卡 |
| **Recorded runs** | **每侧几次**、什么时候跑的 |
| **Recorded runtime** | 每侧跑了多久 |

`Recorded runs` 这一行是整张表里最要紧的：审稿人扫一眼就知道这个门槛是几次运行
标定出来的。不要省。

### Task abstract

一段。第一句说缺陷，最后一句给**实测后果的数字**。不要写"显著下降"这种词。

### The defect

根因写成路径 + 符号 + 一小段代码。然后用一个 `###` 子节回答**为什么它没有当场炸** ——
这通常才是任务成立的原因。形状合法、dtype 合法、异常没抛，正是 agent 要翻过的那道坎。

**似是而非的诊断就放在这里反驳**，不要单开一节。最有力的那条证据（例如"差距在
step 0 就已经拉开"）要明写出来，它往往一句话就排掉一整类错误归因。

### Evidence

表格列随包的产物，一列文件名，一列**它能证明什么**。只列文件名没有用。

然后放实测曲线。曲线下面写清楚两侧共享了什么（同一份数据、同一个 seed、同一个
配方），否则读者没有理由相信这个对照。

### Reference answer

两档分开。`issue` 档是**判词**：必须点出哪几件事，多根因时写明只点出一部分要给
部分分，并列出哪些归因必须判不过。`solution` 档是**补丁**：最小的那个 diff，
以及为什么它充分。

两侧之间**不属于修复**的差异也写在这里（端口、卡号这类），它们是打包时要抹掉的 tell。

### Verification

判据键、门槛数值，以及**门槛是从几次运行上量的**。有对照侧就报出两侧余量，
说明哪边薄，并检查门槛是否压过了对照侧的起步点 —— 压不过就意味着"改对代码不用跑完
也能过"。

每侧只有一次运行时，明写"本门槛只有每侧一次运行作为标定"，只报余量、不报方差。
含糊过去等于把风险留给下一个人。

raw task 阶段还没有 verifier，这一节写成提议，标明是提议。

### Running it

启动命令、入口脚本自己管了什么、**没有随包的东西**（模型、索引、checkpoint）
以及怎么取。能从随包内容算出来的东西不要收，但要写清楚怎么重建。

环境的局限写在这一节末尾，不要单开 Environment 节。

## 不要照搬 ref 的地方

参考的是 terminal-bench-science 的
`tasks/earth-sciences/atmospheric-sciences/sparse-network-assimilation/README.md`。
骨架可用，有几处不要学：

- **不要写两千词的单段落。** 那份的 Difficulty 和 Verification 各是一整段，读不下去。
- **不要有 managed header。** 我们没有 `task.toml`，也没有生成器。身份表手写，
  与 `metadata.yaml` 对齐即可。
- **不要把 reference solution 写成流程说明。** 它们的参考解是一条真实的求解管线；
  我们的通常是一次 revert，智力含量在根因与门槛上，不在补丁里。
- **不要为每件事单开一节。** 那份有 Author's relevant experience、Difficulty、
  Reference solution、Verification 四个大节，各自冗长。我们六节封顶。
