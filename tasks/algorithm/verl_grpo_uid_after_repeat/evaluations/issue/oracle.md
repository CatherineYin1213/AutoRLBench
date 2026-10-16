# Canonical issue oracle

Reconciled from the task's frozen evidence and the diff between the staged
workspace and upstream verl `6f2aa24abfc01cc72811b376aeb3936503885096`.

Expected verdict: **unhealthy**.

## Root cause

- `verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit`
  - kind: `symbol`
  - observed: the `batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)` assignment was moved to **after** `batch = batch.repeat(repeat_times=rollout.n, interleave=True)` and `batch.union(gen_batch_output)`. The batch at that point holds one row per sampled response, so every one of the `train_batch_size * rollout_n` rows receives its own fresh UUID.
  - expected: the uid is assigned **before** `self._get_gen_batch(batch)`, while the batch still holds one row per prompt, so the later `repeat(interleave=True)` duplicates each uid `rollout_n` times and all samples drawn from one prompt share a group key.

The uid is the grouping key that `compute_advantage` passes to
`compute_grpo_outcome_advantage` as `index` (`ray_trainer.py:243`,
`ray_trainer.py:256-257`). One uid per row makes every GRPO group a singleton.
`verl/trainer/ppo/core_algos.py::compute_grpo_outcome_advantage` handles
`len(id2score[idx]) == 1` by silently substituting `id2mean = 0.0` and
`id2std = 1.0` instead of raising, so the group-relative baseline is never
subtracted and the per-group standard deviation never divides. The advantage
of each response therefore degenerates to its own raw outcome reward:
GRPO becomes REINFORCE with no baseline and no variance normalisation.

## Observable impact

- Primary source evaluation metric: `val-core/math_dapo/acc/mean@1`.
- `critic/advantages/mean` equals `critic/rewards/mean` digit for digit at
  every logged step, instead of sitting near zero. Group-mean subtraction is
  what forces a healthy GRPO batch's advantages to average to zero.
- `critic/advantages/min` is exactly `0.0` and `critic/advantages/max` is
  exactly `1.0` at every step, matching the 0/1 reward range of
  `reward_math_dapo_01.py::compute_score`. A healthy run on the same recipe
  (`train_batch_size` 8, `rollout.n` 8, 0/1 reward) shows a negative minimum
  and a discrete positive maximum set by the within-group standard deviation —
  for a group with exactly one correct response out of eight that maximum is
  `(1 - 1/8) / 0.35355 = 2.4749`.
- `critic/returns/*` tracks `critic/advantages/*` exactly, because the GRPO
  estimator returns the advantage tensor for both.
- `actor/pg_loss` sits around the negative of the mean reward (order `1e-2`)
  rather than the order `1e-4` of a baseline-subtracted batch, because every
  token carries a non-negative advantage and nothing pushes any sampled
  response down.
- No group is ever filtered or skipped for having zero advantage: with
  singleton groups a correct response always yields advantage 1.0, so the run
  looks busier than a healthy one while optimising a higher-variance,
  positively biased objective.
- Held-out `val-core/math_dapo/acc/mean@1` fails to improve over the run and
  `actor/entropy` falls faster than the healthy arm.

## Trigger conditions

- `algorithm.adv_estimator=grpo` (or any estimator that reads
  `non_tensor_batch["uid"]` as its grouping index). The defect is inert for
  estimators that do not group, such as GAE.
- `actor_rollout_ref.rollout.n > 1`. At `rollout.n == 1` the moved assignment
  is behaviourally equivalent to the baseline, because every group is a
  singleton either way.
- The run does not crash, warn, or change tensor shapes: the silent
  singleton-group branch in `compute_grpo_outcome_advantage` absorbs the
  defect, so the only evidence is the numeric signature of the advantages.

## Documentation reconciliation

- `ray_trainer.py:1510` still carries the upstream comment "this usually
  changes the order of data in the `batch`, which won't affect the advantage
  calculation (since it's based on uid)". That comment is true of the baseline
  and remains in the staged tree; it describes the invariant the moved
  assignment breaks, and is not itself a defect.
- The `# add uid to batch` comment moved with the assignment, so the staged
  tree has exactly one uid assignment in the training loop, as upstream does.
  Counting assignments does not reveal the defect; their position relative to
  `repeat` does.
- A low absolute held-out accuracy is a symptom, not the root cause. The
  recipe's base accuracy on DAPO-Math is only a few percent, so the held-out
  curve alone does not separate this defect from a healthy short run. The
  advantage statistics do.
- Reporting `train_batch_size`, `rollout.n`, the learning rate, the reward
  function, or the absence of KL loss as the root cause is a wrong attribution:
  all of them match the healthy recipe for this task.
- Reporting "GRPO group size is 1" or "advantages are not normalised" without
  tying it to where the uid is assigned relative to `batch.repeat` is an
  incomplete attribution and must not receive full `root_cause_score`.
