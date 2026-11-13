from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def fake_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[str], None]:
    """Make ``ffmpeg`` on PATH a shell script with the given body, and nothing else."""

    def install(script: str) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        ffmpeg = bin_dir / "ffmpeg"
        ffmpeg.write_text(f"#!/bin/sh\n{script}\n")
        ffmpeg.chmod(0o755)
        monkeypatch.setenv("PATH", str(bin_dir))

    return install


class ScriptedModel:
    """A model that gives canned structured answers, one per call, in order.

    Every call costs 100 input and 10 output tokens.
    """

    def __init__(self, answers: tuple[dict[str, object], ...]) -> None:
        self._answers = list(answers)
        self.calls: list[list[ModelMessage]] = []
        self.model = FunctionModel(self._respond)

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.calls.append(messages)
        answer = self._answers.pop(0)
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, answer)],
            usage=RequestUsage(input_tokens=100, output_tokens=10),
        )

    def prompt(self, call: int = 0) -> str:
        """The user prompt text the model received on the given call."""
        return "\n".join(
            str(part.content)
            for message in self.calls[call]
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        )


@pytest.fixture
def scripted_model() -> Callable[..., ScriptedModel]:
    def make(*answers: dict[str, object]) -> ScriptedModel:
        return ScriptedModel(answers)

    return make
