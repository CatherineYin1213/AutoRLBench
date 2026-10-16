# task_dapo_math_021 — GRPO loses its group baseline

Maintainer-facing. Not exposed to the agent.

## Case overview

| | |
| --- | --- |
| evaluation key | `val-core/math_dapo/acc/mean@1` |
| family | `verl_implementation/llama_dapo_implementation` |
| Lark 分类 | ③ Algorithm |
| schema version | 2 |
| levels | `issue` only (see "Why issue-only for now") |
| model | Llama-3.2-1B-Instruct |
| dataset | DAPO-Math-17k |
| framework | verl `0.9.0.dev` @ `6f2aa24abfc01cc72811b376aeb3936503885096` |
| algo | GRPO |
| launch script | `examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh` |
| GPUs | 2 |
| budget | 60 steps, eval every 20 |

## Canonical failure reason

The per-prompt `uid` assignment in
`verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit` was moved below
`batch = batch.repeat(repeat_times=rollout.n, interleave=True)`, so each of
the `train_batch_size * rollout_n` sampled responses receives its own fresh
UUID instead of inheriting its prompt's. `uid` is the grouping key that
`compute_advantage` hands to `compute_grpo_outcome_advantage` as `index`, so
every GRPO group becomes a singleton. That function's `len(group) == 1` branch
silently substitutes `id2mean = 0.0` and `id2std = 1.0` rather than raising,
so the group baseline is never subtracted and the per-group standard
deviation never divides: GRPO degenerates into REINFORCE with no baseline over
the raw 0/1 reward.

## Workspace and hidden evaluations

`workspace/` is the defect-side snapshot: upstream verl at the pinned commit
with one file modified, plus the recipe's launch script and reward wrapper.
`evaluations/issue/oracle.md` is hidden from the agent and is what the LLM
judge actually reads.

**`environment/` is mounted read-only into the agent container** at
`/ptdbench/environment` (`ptdbench/agents/base.py`), so it holds only
`prepare.sh`. `validation/source_manifest.json` — which spells out the
injected defect — deliberately lives under `validation/` instead of the
`environment/` location that `skills/package-task/SKILL.md` suggests.
`validation/` is never mounted. Do not move it back.

## Source and preserved context

Base commit: `6f2aa24abfc01cc72811b376aeb3936503885096` (2026-07-15), the
same base as the sibling task `task_monkey_patch_011`, whose recorded
2026-07-20 run executed on the `verl` PTDBench image. That is the evidence
this commit is compatible with the image; the two trees differ only by each
task's own injected defect plus the sibling's bench scaffolding.

Filtered root-cause diff — exactly one file, one relocation:

```text
verl/trainer/ppo/ray_trainer.py
  -  uid assignment above  `gen_batch = self._get_gen_batch(batch)`
  +  uid assignment below  `batch = batch.union(gen_batch_output)`
```

The move is verbatim, comment included, so the staged tree still has exactly
one uid assignment in the training loop and two `uuid.uuid4()` call sites in
the file, as upstream does. An agent cannot find this by counting
assignments or by noticing a stray extra statement; only the position
relative to `batch.repeat` is wrong. Full provenance, including what was
deliberately *not* carried over from the sibling, is in
`validation/source_manifest.json`.

## Original failure evidence

**Not yet recorded.** `workspace/logs/` holds only a `README.txt` at the time
of packaging. The defect-arm run is being produced separately on GPU
hardware and will be added to `workspace/logs/`. Until then every number in
the next section is a derivation, not a measurement — `validation/status.json`
says so explicitly.

## Canonical expected outcome

Verdict: **unhealthy**. Root cause as above. The fix is to move the uid
assignment back above `self._get_gen_batch(batch)`.

The signature is unusually sharp, and most of it is checkable from the defect
log alone without a control arm:

| metric | healthy GRPO | this defect |
| --- | --- | --- |
| `critic/advantages/mean` | ≈ 0 (group mean removed) | **equals `critic/rewards/mean`** |
| `critic/advantages/min` | negative | **exactly 0.0** |
| `critic/advantages/max` | 2.4749 for a 1-of-8 group | **exactly 1.0** |
| `critic/returns/*` | tracks advantages | tracks advantages |
| `actor/pg_loss` | order `1e-4` | order `1e-2` |

Both the healthy and the defect columns were reproduced on CPU against the
real `compute_grpo_outcome_advantage` logic at the pinned commit. The healthy
`2.4749` is not a guess: it is `(1 - 1/8) / 0.35355`, and the sibling task's
real frozen log records `critic/advantages/max:2.4748666286468506` on the same
recipe. That cross-check is what makes the derived table trustworthy ahead of
the recorded run.

## Evaluation design

Issue level only. The shared LLM judge compares
`$PTDBENCH_RESULTS/agent_answer.md` against the hidden `oracle.md` on verdict
/ root cause / evidence with weights 0.2 / 0.6 / 0.2 and a pass bar of 0.8.
There is a single root cause, so `oracle.md` carries no partial-credit clause.
It does carry the opposite requirement: "group size is 1" or "advantages are
not normalised", stated without locating the uid assignment relative to
`batch.repeat`, is incomplete and must not get full `root_cause_score`. The
symptom is one grep away in the log; the cause is one indirection further, in
the silent singleton-group branch.

`oracle.md` also lists the wrong attributions to reject: `train_batch_size`,
`rollout.n`, the learning rate, the reward function, the absence of KL loss,
and a bare "held-out accuracy is low". All of those match the healthy recipe.

### Why issue-only for now

A solution level needs a line saying what the metric should reach once
repaired, and that line has to be measured off a real repaired run. Neither
arm has been run yet. Adding solution later is cheap — the repair is a
one-file revert and the control arm uses the same launch script unchanged —
but the threshold must come from the repaired run's own logs, not from a
conversion.

### Recording the two arms

Defect arm, from `workspace/`:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh
```

Control arm: in a *separate* copy of the workspace, move the four-line
`batch.non_tensor_batch["uid"] = ...` block in
`verl/trainer/ppo/ray_trainer.py` back above
`gen_batch = self._get_gen_batch(batch)`, then run the identical command. No
launch-script or config change is involved, so the two logs differ only by
the defect. Keep the control arm's code and logs out of `workspace/`.

The script writes to `logs/llama3_2_1b_instruct_dapo_math_fsdp_<timestamp>.log`
and offline W&B to `logs/wandb/`. Check the first logged step before letting
either run finish: `critic/advantages/min` is the whole experiment, and it is
decided by step 1.

## Environment requirement

The task runs on the `verl` PTDBench image (`environment: image: verl`, built
from `docker/images/verl.Dockerfile`) — verl + vLLM 0.17, torch 2.10 /
CUDA 12.9 on Ubuntu 22.04. It is the image the sibling task's recorded run
used with this same verl commit, which is why the staged tree is known to
import and launch there. The CPU-only `rank-distillation-cpu` image cannot
host a vLLM rollout, and the slime images are a different framework.

`environment/prepare.sh` delivers the Llama-3.2-1B-Instruct weights to
`workspace/verl/Llama-3.2-1B-instruct` (note the lowercase `i`, which is what
the launch script looks for) and the DAPO-Math-17k train/test parquet pair to
`workspace/DAPO-Math-17k_verl`, both pinned by commit sha and checksummed.
Its non-comment lines are byte-identical to the sibling task's `prepare.sh`,
so the two tasks share one dataset cache entry.

The recipe needs **2 GPUs**; `metadata.yaml` declares them through
`environment.env`.
