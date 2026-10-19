from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ...core.types import CommandResult
from ...runtimes import CommandSpec, EnvironmentArtifact, SandboxRuntime
from ..base import AgentExecutionContext, AgentRunner
from ..factory import register_agent_runner


@register_agent_runner("opencode")
class OpenCodeAgentRunner(AgentRunner):
    """Call an OpenAI-compatible model through the OpenCode CLI."""

    # Use a fixed provider name so tasks need not supply an OpenCode config file.
    provider_name = "autorlbench"

    def run(
        self,
        context: AgentExecutionContext,
        *,
        runtime: SandboxRuntime,
        artifact: Optional[EnvironmentArtifact],
    ) -> CommandResult:
        if not self.spec.base_url:
            raise ValueError("opencode agent needs --base-url")

        config = self._build_config()
        env = dict(self.spec.env)
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config, ensure_ascii=False)
        request = context.sandbox_request(
            runtime,
            env=env,
            network=self.spec.network,
        )
        workspace_path = runtime.path(request, "workspace")
        prompt = self._build_prompt(context, workspace_path)
        result = runtime.run(
            request,
            CommandSpec(argv=self._build_argv(prompt, workspace_path)),
            artifact=artifact,
        )
        self._write_opencode_trace(
            context.results,
            prompt,
            config,
            result,
            backend=runtime.name,
        )
        return result

    def _build_config(self) -> Dict[str, Any]:
        return {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                self.provider_name: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "AutoRLBench OpenAI Compatible",
                    "options": {
                        "baseURL": self.spec.base_url,
                        "apiKey": self.spec.api_key or "EMPTY",
                    },
                    "models": {
                        self._model_name(): {
                            "name": self._model_name(),
                            "limit": {
                                "context": 32768,
                                "output": 8192,
                            },
                        }
                    },
                }
            },
        }

    def _build_argv(self, prompt: str, workdir: str) -> list[str]:
        argv = [self.spec.command or "opencode", "run", "--pure"]
        argv.extend(["--dir", workdir, "--format", "json", "--model", self._model_selector()])
        argv.extend(list(self.spec.extra_args))
        argv.append(prompt)
        return argv

    def _model_name(self) -> str:
        # SGLang currently returns /model by default; keeping the slash makes the selector
        # autorlbench//model. If the server uses --served-model-name, that name is used directly.
        return self.spec.model or "/model"

    def _model_selector(self) -> str:
        return f"{self.provider_name}/{self._model_name()}"

    def _build_prompt(self, context: AgentExecutionContext, workspace_path: str) -> str:
        return "\n".join(
            [
                "You are solving a AutoRLBench task.",
                (
                    "Only modify files under the workspace, except for deliverables "
                    "that the task explicitly requires under $AUTORLBENCH_RESULTS."
                ),
                "Do not touch oracle/ or verifier/.",
                "Task instruction:",
                context.instruction.strip(),
                "",
                f"Workspace absolute path: {workspace_path}",
                "Use $AUTORLBENCH_RESULTS only for deliverables required by the task.",
                "Use that exact absolute path when calling file tools.",
                "Do not guess or construct a different workspace path.",
                "The task instruction above is the complete public task description.",
            ]
        )

    def _write_opencode_trace(
        self,
        results_dir: Path,
        prompt: str,
        config: Dict[str, Any],
        result: CommandResult,
        *,
        backend: str,
    ) -> None:
        """Convert the OpenCode JSON event stream into line-by-line role/content messages."""
        metadata = {
            "type": "run_metadata",
            "agent": self.kind,
            "backend": backend,
            "provider": self.provider_name,
            "model": self._model_name(),
            "config": self._redact_config(config),
        }
        messages = [
            self._message("system", metadata),
            {"role": "user", "content": prompt},
        ]

        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                messages.append(self._message("system", {"type": "raw_stdout", "text": line}))
                continue
            if not isinstance(event, dict):
                messages.append(self._message("system", {"type": "raw_event", "event": event}))
                continue
            messages.extend(self._event_messages(event))

        execution = result.to_dict()
        execution.pop("stdout", None)
        messages.append(self._message("system", {"type": "execution_result", **execution}))
        self._write_agent_trace_jsonl(results_dir, messages)

    def _event_messages(self, event: Dict[str, Any]) -> list[Dict[str, str]]:
        """Convert one OpenCode event into one or more standard messages."""
        event_type = event.get("type")
        part = event.get("part")
        if not isinstance(part, dict):
            return [self._message("system", {"type": "raw_event", "event": event})]

        if event_type == "text":
            text = part.get("text", "")
            return [{"role": "assistant", "content": str(text)}]

        if event_type == "tool_use":
            state = part.get("state")
            state = state if isinstance(state, dict) else {}
            call_id = part.get("callID")
            tool_name = part.get("tool")
            call = {
                "type": "tool_call",
                "tool_call_id": call_id,
                "name": tool_name,
                "arguments": state.get("input", {}),
                "timestamp": event.get("timestamp"),
            }
            tool_result = {
                "type": "tool_result",
                "tool_call_id": call_id,
                "name": tool_name,
                "status": state.get("status"),
                "output": state.get("output"),
                "error": state.get("error"),
                "metadata": state.get("metadata"),
                "title": state.get("title"),
                "time": state.get("time"),
            }
            return [
                self._message("assistant", call),
                self._message("tool", tool_result),
            ]

        # step_start/step_finish carry the snapshot, token usage, and stop reason.
        # They are not natural-language messages but belong to the full trajectory, so keep them as system events.
        return [self._message("system", {"type": event_type, "event": event})]

    @staticmethod
    def _message(role: str, content: Any) -> Dict[str, str]:
        if isinstance(content, str):
            text = content
        else:
            text = json.dumps(content, ensure_ascii=False)
        return {"role": role, "content": text}

    @staticmethod
    def _redact_config(config: Dict[str, Any]) -> Dict[str, Any]:
        # The trace keeps the endpoint and model name but never the API key.
        copied = json.loads(json.dumps(config))
        provider = copied.get("provider", {}).get("autorlbench", {})
        options = provider.get("options", {})
        if "apiKey" in options:
            options["apiKey"] = "<redacted>"
        return copied
