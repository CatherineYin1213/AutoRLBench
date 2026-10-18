FROM slimerl/slime:v0.3.2-cu129@sha256:39be6cbb00f9b6770e664ace0c7b9f5ecff2977a1a205e7926a720f906fbc62c

# Nothing is installed on top of the upstream slime image on purpose.
#
# The reproduction runs entirely on what that image already ships:
#   - Megatron-LM, SGLang, Ray, PyTorch  (the training stack under test)
#   - Python 3.12 + pytest               (the Endless Terminals verifiers)
#   - sqlite3, coreutils, bash           (what the terminal tasks use)
#
# Keeping this image a pure digest pin of the upstream one means the case adds
# no new image to the benchmark, and it is also the only thing that builds on a
# host without a package mirror: the workspace needs no `pip install` and no
# network access at any point, including during the rollout.

ENV PYTHONUNBUFFERED=1

# Mount points that the workspace's run.sh expects (all overridable by env var):
#
#   /workspace                 buggy_workspace/ or fixed_workspace/
#   /models/Qwen3-1.7B         Qwen3-1.7B HuggingFace weights       (read only)
#   /datasets/endless-terminals  raw Endless Terminals task_* dirs  (read only)
#   /ckpt                      writable scratch for Megatron checkpoints
#
# /dev/shm is mounted noexec in this image, so the workspace deliberately keeps
# its episode sandboxes on /tmp -- see et/sandbox.py:check_exec_capable.
WORKDIR /workspace
