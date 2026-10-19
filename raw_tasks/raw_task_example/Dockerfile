# Environment for the Search-R1 zero-query-embedding case.
#
# Self-contained: no build context is needed, so either of these works.
#
#     docker build -t search-r1-raw -f Dockerfile .
#     docker build -t search-r1-raw - < Dockerfile
#
# The versions below are the recorded ones. results/baseline32/experiment.json
# in both workspaces names the six that run.py captured at launch:
#
#     torch 2.4.0          vLLM 0.6.3       flash-attn 2.6.3
#     transformers 4.46.3  ray 2.40.0       faiss-cpu 1.14.2
#
# vLLM 0.6.3 is not negotiable: Search-R1/verl/third_party is written against
# that rollout API and does not run on a newer one. torch 2.4.0 and flash-attn
# 2.6.3 are what that vLLM resolves with, and xformers is pinned because
# VLLM_ATTENTION_BACKEND below selects it.
#
# The recording host built this with `uv sync --frozen` from a uv.lock, which
# additionally pinned every transitive dependency. That lock is not reproduced
# here -- a raw task should not need a build context to be legible -- so the
# direct pins below are exact and the transitive closure is whatever pip
# resolves. If a rebuild ever disagrees with the recorded curves, the
# transitive set is the first place to look.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# python3.10: the recorded stack's interpreter (Ubuntu 22.04 ships it as
# default). build-essential + ninja: flash-attn 2.6.3 publishes no wheel for
# every torch/ABI combination and falls back to compiling. git: sources that
# read repo metadata at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        ninja-build \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/opt/search-r1-runtime/.venv/bin:/usr/local/cuda/bin:$PATH \
    MAX_JOBS=4 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

# Isolated venv, so nothing here touches Ubuntu's apt-managed pip. run.py
# defaults its training interpreter to exactly this path.
RUN python3.10 -m venv /opt/search-r1-runtime/.venv \
    && pip install --upgrade pip setuptools==75.6.0 wheel==0.45.1

# torch before anything else: flash-attn's build imports it, and vLLM 0.6.3
# pins against it. This is the same ordering the recording host's setup_env.sh
# used (it ran `uv sync` twice for the same reason).
RUN pip install torch==2.4.0

RUN pip install vllm==0.6.3 xformers==0.0.27.post2

# --no-build-isolation so the torch above is visible to flash-attn's setup.py
# instead of being downloaded again into a build sandbox.
RUN pip install --no-build-isolation flash-attn==2.6.3

RUN pip install -r /dev/stdin <<'REQS'
accelerate==1.1.1
codetiming==1.4.0
datasets==3.1.0
dill==0.3.8
faiss-cpu==1.14.2
fastapi==0.115.5
httpx==0.27.2
hydra-core==1.3.2
ipython==8.29.0
matplotlib==3.9.2
ninja==1.11.1.1
numpy==1.26.4
openai==1.54.4
packaging==24.2
pandas==2.2.3
pyarrow==17.0.0
pybind11==2.13.6
ray==2.40.0
tensordict==0.5.0
transformers==4.46.3
uvicorn==0.32.1
wandb==0.18.7
REQS

# vLLM 0.6.3 + qwen2 on the flash_attn backend has a known upstream issue, so
# Search-R1/train_grpo.sh exports this itself. Setting it here too means it
# holds for `run.py --check` and for any manual probing.
ENV VLLM_ATTENTION_BACKEND=XFORMERS

# W&B must never reach the network: both recorded runs are offline, and run.py
# forces this as well.
ENV WANDB_MODE=offline

# Build-time consistency check against what experiment.json recorded.
RUN python - <<'PY'
import faiss, flash_attn, ray, torch, transformers, vllm
for name, got, want in [
    ("torch", torch.__version__, "2.4.0"),
    ("vllm", vllm.__version__, "0.6.3"),
    ("flash_attn", flash_attn.__version__, "2.6.3"),
    ("transformers", transformers.__version__, "4.46.3"),
    ("ray", ray.__version__, "2.40.0"),
]:
    print(f"{name:13s} = {got}")
    assert got.startswith(want), f"{name}: got {got}, expected {want}"
print("torch cuda    =", torch.version.cuda)
assert torch.version.cuda == "12.1"
PY

# A workspace is mounted here. Each one carries its own data/ (train.parquet,
# test.parquet, corpus.jsonl, manifest.json); run.py and serve_retriever.py
# resolve both data/ and models/ relative to the workspace root. Neither model
# nor the FAISS index ships -- see README.md, "What is not in this bundle".
WORKDIR /workspace

CMD ["/bin/bash"]
