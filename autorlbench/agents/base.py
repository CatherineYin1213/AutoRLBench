from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..core.types import AgentSpec, CommandResult
from ..runtimes import (
    EnvironmentArtifact,
    SandboxInstance,
    SandboxRequest,
    SandboxRuntime,
)
from ..utils.filesystem import write_text


@dataclass(frozen=True)
class AgentExecutionContext:
    # The agent runtime sees only this context, with no direct dependency on the evaluator or a specific runtime.
    task_id: str
    instruction: str
    metadata: Dict[str, Any]
    timeout_s: Optional[int]
    task_root: Path
    workspace: Path
    visible_task: Path
    results: Path
    evaluation_level: str = "solution"
    log_level: str = "quiet"
    debian_mirror: Optional[str] = None
    # The task's environment/ directory and the prepare-stage output directory are both mounted read-only for the agent.
    # Letting the agent see prepare.sh is intentional: it is changing training code,
    # and "what is installed in the environment" is a premise for its decisions, not part of the answer.
    environment_dir: Optional[Path] = None
    prepared_dir: Optional[Path] = None
    # The two download cache roots, mounted read-only for the agent. Visible so debugging can reconcile them; delivery
    # is still the copy prepare writes into workspace via --dest, and task scripts should not reference these directly.
    models_dir: Optional[Path] = None
    datasets_dir: Optional[Path] = None
    # metadata.environment.env: per-task variables (e.g. CUDA_VISIBLE_DEVICES).
    task_env: Dict[str, str] = field(default_factory=dict)

    def sandbox_request(
        self,
        runtime: SandboxRuntime,
        *,
        env: Optional[Dict[str, str]] = None,
        network: str = "bridge",
        workdir: str = "workspace",
    ) -> SandboxRequest:
        """Build a uniform run request for the agent; the runtime resolves path details."""
        return SandboxRequest(
            stage="agent",
            task_id=self.task_id,
            host_task_root=self.task_root,
            host_visible_task=self.visible_task,
            host_workspace=self.workspace,
            host_results=self.results,
            evaluation_level=self.evaluation_level,
            env=env or {},
            network=network,
            timeout_s=self.timeout_s,
            workdir=workdir,
            host_environment=self.environment_dir,
            host_prepared=self.prepared_dir,
            host_models=self.models_dir,
            host_datasets=self.datasets_dir,
            task_env=self.task_env,
        )


class AgentRunner(ABC):
    # Every concrete agent implements this interface; the evaluator depends only on it.
    def __init__(self, spec: AgentSpec):
        self.spec = spec

    @property
    def kind(self) -> str:
        return self.spec.kind

    def _write_agent_trace(self, results_dir: Path, payload: Dict[str, Any]) -> Path:
        # Collapse the agent's final trajectory into one JSON file to reduce clutter in results.
        path = results_dir / "agent_trace.json"
        write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))
        return path

    def _write_agent_trace_jsonl(
        self,
        results_dir: Path,
        messages: Sequence[Dict[str, str]],
    ) -> Path:
        """Save the full agent trajectory line by line in OpenAI message style."""
        lines = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("trace message needs string role and content")
            lines.append(json.dumps({"role": role, "content": content}, ensure_ascii=False))
        path = results_dir / "agent_trace.jsonl"
        write_text(path, "\n".join(lines) + "\n")
        return path

    @staticmethod
    def _redact_env(env: Dict[str, str]) -> Dict[str, str]:
        redacted: Dict[str, str] = {}
        for key, value in env.items():
            if any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
                redacted[key] = "<redacted>"
            else:
                redacted[key] = value
        return redacted

    @abstractmethod
    def setup(
        self,
        context: AgentExecutionContext,
        *,
        runtime: SandboxRuntime,
        sandbox: SandboxInstance,
    ) -> CommandResult:
        """Install/prep this agent in the already-created sandbox."""
        raise NotImplementedError

    @abstractmethod
    def run(
        self,
        context: AgentExecutionContext,
        *,
        runtime: SandboxRuntime,
        artifact: Optional[EnvironmentArtifact],
        sandbox: Optional[SandboxInstance] = None,
    ) -> CommandResult:
        # Concrete agents only translate the spec into a command; the runtime provides execution.
        raise NotImplementedError
