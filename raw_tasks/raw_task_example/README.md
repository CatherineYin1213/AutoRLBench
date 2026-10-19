# search-r1-zero-query-embedding

| | |
| --- | --- |
| **Author** | Shouzheng Huang |
| **Domain** | `environment` |
| **Tags** | `search-r1` `grpo` `retrieval` `faiss` `e5` `hotpotqa` `device-transfer` `silent-corruption` `multi-hop-qa` |
| **Framework** | Search-R1 with veRL vendored |
| **Model** | Qwen2.5-1.5B-Instruct |
| **Dataset** | HotpotQA distractor — 1024 train / 128 held-out, corpus of 12468 passages |
| **Environment** | `Dockerfile`, self-contained — Python 3.10, CUDA 12.1, torch 2.4.0, vLLM 0.6.3, faiss-cpu 1.14.2 |
| **Resources** | 1 × A100-SXM4-40GB |
| **Recorded runtime** | ~62 min (buggy), ~56 min (fixed) |

## Task abstract

The retrieval encoder returns an all-zero query vector. Under an inner-product
index every document scores exactly `0.0`, so FAISS hands back the same
arbitrary top-3 for every question ever asked — valid shape, valid dtype, no
exception, no warning. The policy trains to completion on retrieved text that
has no relationship to what it asked for, and held-out EM stalls at **15.62%**
at the 32-step horizon against **32.81%** for the identical recipe with the
embedding copied to the host.

This is the **worked example of a raw task** — the four things a contributor
prepares before any packaging happens. `Dockerfile` is the environment both
runs were recorded on; the two workspaces each hold their own code and their
own run's artifacts. See [README.md](../../README.md#contributing-a-task) for
the flow and
[skills/package-task/readme-format.md](../../skills/package-task/readme-format.md)
for this document's format.

## The defect

`Search-R1/search_r1/search/retrieval_server.py`, in `Encoder.encode()`:

```python
response_buffer = np.zeros(tuple(query_emb.shape), dtype=np.float32)
writer = torch.as_tensor(response_buffer, device=query_emb.device)
writer.copy_(query_emb)
query_emb = response_buffer                      # <-- still all zeros
```

`response_buffer` is a zero-filled array in host memory. `query_emb` is on the
GPU. `torch.as_tensor(response_buffer, device=query_emb.device)` does **not**
alias `response_buffer` — asking for a CUDA tensor from CPU memory is a copy,
so `writer` is a fresh GPU tensor sharing no storage with it. `writer.copy_()`
then faithfully writes the embedding into that GPU scratch tensor, which is
discarded on the next line, and the function returns the untouched zero array.

Had the buffer been on the same device, `as_tensor` would have aliased it and
the code would have worked. That is what makes it survive review: **correct on
CPU, silently wrong on GPU.**

### Why nothing catches it

FAISS is built with `METRIC_INNER_PRODUCT`. The inner product of a zero query
with every document is `0.0` — a perfectly valid score — so the index returns a
well-formed top-3, the same one every time, since there is nothing to rank by.
No exception, no shape change, no NaN.

The defect sits on the **evaluation path as well as the rollout path**; both go
through the same encoder. So the buggy arm is not a model that cannot search.
It searches on every single question (`val/searched` is `1.0` at all five
validation points), its answers are well-formed (`val/format` ≥ 0.977), and it
still improves with training. Only the retrieved content is meaningless.

## Evidence

Each workspace carries its own run's artifacts under `results/baseline32/`,
where the run wrote them:

| Artifact | What it answers |
| --- | --- |
| `rollouts/validation_metrics_NNNN.json` | the decider: `val/em` and `val/f1` at steps 0, 10, 20, 30, 32 |
| `rollouts/metrics_N.json` | per-step training curve, all 32 steps |
| `retrieval.jsonl` | the symptom itself — every returned document scoring `0.0` (first 8 rows kept) |
| `retrieval_server.log` | the retriever was up and serving; this is not an outage |
| `train.log`, `launch.log` | complete command-line output of the run |
| `experiment.json` | the run's declared config, plus SHA-256 of every input and source file |
| `wandb/wandb/offline-run-*/` | the W&B offline run, including the `run-*.wandb` binary — without it `wandb sync` cannot replay the run |
| `rollouts/step_N.jsonl` | full rollout dumps, steps 1 and 32 as samples |

`retrieval.jsonl` ships trimmed to its first 8 rows, which is enough to see the
symptom. Counted over the two complete traces: the buggy arm's 132 records held
13959 document scores, **all zero**, across exactly **one** distinct document
combination; the fixed arm's 116 records held 14193 nonzero scores across 3285
combinations.

The same thing is measurable before training, scoring each arm's retriever on
the held-out questions with the original question as the query:

| held-out retrieval quality | buggy | fixed |
| --- | ---: | ---: |
| `support_title_recall_at_3` | **0.000** | 0.809 |
| `any_support_hit_at_3` | **0.000** | 0.977 |
| `all_scores_zero` | **true** | false |

### Held-out results

Readable out of each side's `rollouts/validation_metrics_*.json`:

| Step | buggy EM | fixed EM | buggy F1 | fixed F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 3.12% | 13.28% | 8.97% | 20.13% |
| 10 | 4.69% | 23.44% | 13.51% | 32.34% |
| 20 | 11.72% | 31.25% | 18.20% | 41.74% |
| 30 | 13.28% | 32.03% | 21.02% | 40.65% |
| **32** | **15.62%** | **32.81%** | **24.70%** | **41.43%** |

Both arms improve — +12.50 points for the buggy arm, +19.53 for the fixed one.
Both trained on byte-identical `data/`, the same seed and the same recipe; the
only code difference is the four lines above.

The gap is already open at **step 0**, before either arm has taken a gradient
step, because the defect is in inference and not in learning. That one fact
rules out every learning-side explanation — the optimizer, the reward, the
budget, the model size — and it is readable straight out of the shipped
artifacts.

## Reference answer

### `issue` — the diagnosis

A passing answer must identify **all** of:

1. The query embedding returned by `Encoder.encode()` is all zeros.
2. The cause is the device mismatch: `torch.as_tensor(cpu_array, device=cuda)`
   copies rather than aliases, so `writer.copy_()` fills a discarded GPU
   tensor and `response_buffer` is returned untouched.
3. It is silent because a zero query yields a valid `0.0` inner-product score
   against every document, so FAISS returns a well-formed constant top-3.

Identifying only part of these must receive partial root-cause credit.
An answer attributing the failure to the checkpoint, the reward, the data, the
model size, or to "training not working" must not pass — the last of these
contradicts the measured curve.

### `solution` — the patch

```diff
-        response_buffer = np.zeros(tuple(query_emb.shape), dtype=np.float32)
-        writer = torch.as_tensor(response_buffer, device=query_emb.device)
-        writer.copy_(query_emb)
-        query_emb = response_buffer
+        query_emb = query_emb.detach().cpu().numpy()
+        query_emb = query_emb.astype(np.float32, order="C")
```

`.cpu()` is what repairs the transfer. `.detach()` drops the autograd
association and `.astype(..., order="C")` guarantees the float32 C-contiguous
layout FAISS requires; neither is the fix, both are there so the result is
unambiguously what the index expects.

### One difference that is not the fix

`config.json` differs between the two arms:

| | buggy | fixed |
| --- | ---: | ---: |
| `gpu` | 4 | 5 |
| `port` | 8004 | 8005 |

The arms ran concurrently on one host, so they needed separate devices and
separate retriever ports. Placement parameters, not reproduction conditions —
and a tell that packaging must neutralize.

## Verification

Proposed, since a raw task has no verifier yet.

- **Decider:** `val/em` at step 32, read from
  `rollouts/validation_metrics_0032.json`.
- **Threshold:** between the two measured endpoints, 15.62% and 32.81%. A bar
  at **25%** leaves margins of 9.4 points below and 7.8 points above; the upper
  margin is the thinner one.
- **The bar must clear the fixed arm's step 0** (13.28%), otherwise repairing
  the code without running would pass. 25% clears it by 11.7 points, so both
  fixing and running are required.
- **Upper bound:** `1.0`, the metric's own ceiling — it rejects fabricated
  values and is not a quality line.

**This threshold has one run per side as its calibration.** Margins are
reportable; variance is not. A result landing near 25% is in untested
territory and has to be judged by reading the log. A matched pair of reruns is
what would earn a variance claim.

Neither evaluation should gate on the precheck numbers above: they measure
retrieval quality directly, and asserting on the query side anywhere the agent
can read it would hand over the location of the defect.

## Running it

```bash
python3 buggy_workspace/run.py --run-name baseline32
python3 fixed_workspace/run.py  --run-name baseline32
```

`run.py` runs in the foreground, starts and reaps the retriever itself, refuses
to overwrite an existing run name, and refuses to start on a GPU that already
has memory in use. W&B is forced offline. `--check` validates inputs, row
counts and local imports without a GPU.

Two things each workspace needs and does not ship. Both models, ~3.5 GB —
`run.py --check` fails until they are in `models/`:

```bash
hf download Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --include '*.json' '*.txt' model.safetensors \
  --local-dir <workspace>/models/Qwen2.5-1.5B-Instruct
hf download intfloat/e5-base-v2 \
  --include config.json model.safetensors special_tokens_map.json \
           tokenizer.json tokenizer_config.json vocab.txt \
  --local-dir <workspace>/models/e5-base-v2
```

The Qwen revision is the commit the experiment used; the E5 copy's revision was
never recorded, so `main` is the only pointer available. And `data/e5_Flat.index`
(38 MB), left out because it is a pure function of the shipped `corpus.jsonl`
plus that E5 checkpoint — rebuild it rather than version it.

vLLM 0.6.3 is the one pin that is not negotiable: `Search-R1/verl/third_party`
is written against that rollout API. The `Dockerfile`'s last layer asserts five
of its pins against what `experiment.json` recorded, so drift fails the build
rather than the run. One caveat it states itself: the recording host built from
a `uv.lock` that also pinned every transitive dependency, and that lock is not
reproduced here, so the direct pins are exact and the closure is whatever pip
resolves.
