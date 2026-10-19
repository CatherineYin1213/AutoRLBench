# Missing flash-attention hook

Training Llama-3.2-1B-Instruct with GRPO on DAPO-Math-17k. A rewritten `apply_monkey_patch` no longer installs the baseline flash-attention hook.

| | |
| --- | --- |
| **Author** | AutoRLBench |
| **Domain** | infrastructure |
| **Tags** | `verl` `grpo` `llama` `flash-attn` `monkey-patch` |
| **Model** | Llama-3.2-1B-Instruct |
| **Dataset** | DAPO-Math-17k (test split 1807 rows) |
| **Framework** | verl, `examples/grpo_trainer/run_llama3_2_1b_instruct_dapo_math_fsdp.sh` |
| **Environment** | `verl` · 2 GPUs |
| **Evaluations** | issue · solution |
| **Recorded runs** | 1 defective + 1 repaired run, same node, 200 steps, ~29 min each |

## Difficulty

`verl/models/transformers/monkey_patch.py::apply_monkey_patch` drops the flash-attention hook used under `use_remove_padding`. The model still runs forward, nothing errors, and the held-out curve sits at 3–4%, which reads like "small model, hard task".

A second cause is entangled with the first: `TRAIN_BATCH_SIZE=4` with sparse 0/1 rewards leaves most GRPO groups all-wrong, and 70.5% of steps have zero advantage. Each cause alone explains the curve, so the diagnosis has to separate them. `repeat_kv` is not on the execution path (SP=1).

| arm | step 200 | peak |
| --- | ---: | ---: |
| defective (no hook) | 0.0476 | 0.0686 @ 100 |
| repaired (hook restored) | 0.0736 | 0.0747 @ 180 |

## Reference solution

- **issue**: either cause counts: the missing hook in `apply_monkey_patch`, or the too-small `TRAIN_BATCH_SIZE`.
- **solution**: restore the full baseline `apply_monkey_patch`. That alone clears the threshold; the calibration run did not touch the batch size.

## Verification

- **issue**: an LLM judge scores verdict / root cause / evidence, weighted 0.2 / 0.6 / 0.2. Passing requires ≥ 0.8.
- **solution**: rerun the launch script and require `val-core/math_dapo/acc/mean@1` at step 200 ≥ 0.060. Only step 200 counts, so extending the run or raising eval frequency cannot lift an unrepaired score. The threshold is the midpoint of the two arms (~2.2 SE above defective, ~2.4 SE below repaired), calibrated on one run per side.
