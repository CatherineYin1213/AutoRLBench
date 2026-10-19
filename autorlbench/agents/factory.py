from __future__ import annotations

from typing import Callable, Dict, Optional, Type

from ..core.types import AgentSpec
from .base import AgentRunner

_REGISTRY: Dict[str, Type[AgentRunner]] = {}


def register_agent_runner(kind: str) -> Callable[[Type[AgentRunner]], Type[AgentRunner]]:
    # A new agent only needs to register itself; the evaluator does not change.
    def decorator(cls: Type[AgentRunner]) -> Type[AgentRunner]:
        _REGISTRY[kind] = cls
        return cls

    return decorator


def resolve_agent_spec(agent: Optional[AgentSpec], legacy_cmd: Optional[str]) -> Optional[AgentSpec]:
    # Support the legacy --agent-cmd first; if a new-style AgentSpec is already given, use it directly.
    if agent is not None:
        return agent
    if legacy_cmd:
        return AgentSpec(kind="shell", command=legacy_cmd)
    return None


def create_agent_runner(spec: AgentSpec) -> AgentRunner:
    runner_cls = _REGISTRY.get(spec.kind)
    if runner_cls is None:
        raise ValueError(f"unsupported agent kind: {spec.kind}")
    return runner_cls(spec)


# Import installed agents to register them.
from . import installed as _installed  # noqa: E402,F401
